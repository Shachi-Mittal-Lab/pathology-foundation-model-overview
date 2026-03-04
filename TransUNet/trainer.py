import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms

def trainer_he(args, model, snapshot_path):
    from datasets.dataset_dcishe import dcisHE_dataset, RandomGeneratorRGB
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = dcisHE_dataset(
            base_dir=args.root_path,
            list_dir=args.list_dir,
            split="train",
            transform=transforms.Compose(
                [RandomGeneratorRGB(output_size=[args.img_size, args.img_size])]
            )
        )
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    # model.train()

    # Put the whole model in eval mode to freeze BN stats + disable dropout everywhere
    model.eval()

    # Put only the segmentation head back into train mode (prevent batch norm from running updates)
    head = model.module.segmentation_head if hasattr(model, "module") else model.segmentation_head # TODO
    head.train()
        
    IGNORE_INDEX = 255
    ce_loss = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    dice_loss = DiceLoss(num_classes, ignore_index=IGNORE_INDEX)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters()) # make sure only trainable parameters are used by optimizer
    optimizer = optim.SGD(trainable_params, lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)

            # TODO: DEBUG - check label range
            lb = label_batch
            bad = (lb != 255) & ((lb < 0) | (lb >= args.num_classes))
            if bad.any():
                vals = torch.unique(lb[bad]).detach().cpu().tolist()
                all_vals = torch.unique(lb).detach().cpu().tolist()
                print("Unique labels in batch:", all_vals)
                print("Invalid labels (non-255):", vals)
                raise ValueError("Found invalid labels in batch")

            # --- this is dense per-pixel multi-class cross entropy (semantic segmentation), 
            # --- not “masked loss” unless you explicitly tell it to ignore certain label values.
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                # image = image_batch[1, 0:1, :, :] # will only show the R channel for an RGB image
                image = image_batch[1]  # (3,H,W)
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"