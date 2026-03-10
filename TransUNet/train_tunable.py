import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling_r50CCL import VisionTransformer as ViT_seg
from networks.vit_seg_modeling_r50CCL import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling_r50CCL import load_weights_selectively
from trainer import trainer_he

parser = argparse.ArgumentParser()
# parser.add_argument('--pretrained_model_path', type=str, default='',
#                     help='path to a checkpoint to load (optional)')
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str, 
                    default='dcisHE', help='dataset name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')

# models to load weights from
parser.add_argument('--load_selective', action='store_true',
                    help='Use selective weight loading from multiple sources')
parser.add_argument('--backbone_path', type=str, default=None,
                    help='Path to backbone checkpoint (e.g., RetCCL)')
parser.add_argument('--transunet_path', type=str, default=None,
                    help='Path to TransUNet checkpoint')
parser.add_argument('--pretrained_model_path', type=str, default=None,
                    help='Path to complete pretrained model')

# loading flags
parser.add_argument('--load_backbone', action='store_true', default=True,
                    help='Load backbone weights (default: True)')
parser.add_argument('--no_load_backbone', dest='load_backbone', action='store_false',
                    help='Do not load backbone weights')
parser.add_argument('--load_transformer', action='store_true', default=True,
                    help='Load transformer weights (default: True)')
parser.add_argument('--no_load_transformer', dest='load_transformer', action='store_false',
                    help='Do not load transformer weights')
parser.add_argument('--load_decoder', action='store_true', default=True,
                    help='Load decoder weights (default: True)')
parser.add_argument('--no_load_decoder', dest='load_decoder', action='store_false',
                    help='Do not load decoder weights')
parser.add_argument('--load_seg_head', action='store_true', default=True,
                    help='Load segmentation head weights (default: True)')
parser.add_argument('--no_load_seg_head', dest='load_seg_head', action='store_false',
                    help='Do not load segmentation head weights')

# freezing options
parser.add_argument('--freeze_all_except_head', action='store_true',
                    help='Freeze all parameters except segmentation head')
parser.add_argument('--freeze_backbone', action='store_true',
                    help='Freeze ResNet50 backbone')
parser.add_argument('--freeze_transformer', action='store_true',
                    help='Freeze transformer encoder')
parser.add_argument('--freeze_decoder', action='store_true',
                    help='Freeze decoder')

args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '../data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        'dcisHE': { # TODO
            # set these defaults however you want
            'root_path': r"E:\PROJ_DCIS\training_patches\annotation_scheme-0\as0_all_tiles", 
            'list_dir': r"E:\PROJ_DCIS\lists\lists_dcisHE_TU_retrain_R50-ViT-B_16_skip3_epo150_bs8_224_4class",
            'num_classes': 4,
        }
    }
    
    # Only override args if user didn't pass them explicitly
    if dataset_name in dataset_config:
        if args.root_path == parser.get_default('root_path'):
            args.root_path = dataset_config[dataset_name]['root_path']
        if args.list_dir == parser.get_default('list_dir'):
            args.list_dir = dataset_config[dataset_name]['list_dir']
        if args.num_classes == parser.get_default('num_classes'):
            args.num_classes = dataset_config[dataset_name]['num_classes']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip

    if args.vit_name.find('R50') != -1:
        if 'CCL' not in args.vit_name: # don't calculate patch grid if replacing backbone
            config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), 
                                    int(args.img_size / args.vit_patches_size))

    # ADD THIS DEBUG PRINT:
    print(f"DEBUG: patches.grid = {config_vit.patches.grid}")  # Should be (7, 7)

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    
    # Load ImageNet pretrained weights
    # net.load_from(weights=np.load(config_vit.pretrained_path))

    # # TODO Load weights from model trained on CT data (9 classes)
    # if args.pretrained_model_path: 
    #     print(f"Loading 3-class pretrained weights from {args.pretrained_model_path}")
    #     checkpoint = torch.load(args.pretrained_model_path)
    #     net.load_state_dict(checkpoint)
    # else:
    #     # Fall back to ImageNet pretrained
    #     net.load_from(weights=np.load(config_vit.pretrained_path))

    # # TODO Freeze everything
    # for p in net.parameters():
    #     p.requires_grad = False

    # # TODO Unfreeze only the segmentation head
    # for p in net.segmentation_head.parameters():
    #     p.requires_grad = True

    # ============================================================
    # LOAD IN WEIGHTS 
    # ============================================================

    print("=" * 60)
    print("WEIGHT LOADING STRATEGY")
    print("=" * 60)

    if args.load_selective: # selective weight loading from different sources
        load_weights_selectively(
            model=net,
            backbone_path=args.backbone_path,
            transunet_path=args.transunet_path,
            load_backbone=args.load_backbone,
            load_transformer=args.load_transformer,
            load_decoder=args.load_decoder,
            load_seg_head=args.load_seg_head
        )

    elif args.pretrained_model_path: # load complete pretrained TransUNet
        print(f"Loading complete model from: {args.pretrained_model_path}")
        checkpoint = torch.load(args.pretrained_model_path)
        
        if 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            net.load_state_dict(checkpoint['state_dict'])
        else:
            net.load_state_dict(checkpoint)

    else:
        # load ImageNet pretrained ViT weights
        if hasattr(config_vit, 'pretrained_path') and config_vit.pretrained_path:
            print(f"Loading ImageNet ViT from: {config_vit.pretrained_path}")
            net.load_from(weights=np.load(config_vit.pretrained_path))
        else:
            print("WARNING: Training from scratch (no pretrained weights)")

    print("=" * 60 + "\n")

    # ============================================================
    # FREEZING WEIGHTS
    # ============================================================

    print("=" * 60)
    print("PARAMETER FREEZING STRATEGY")
    print("=" * 60)

    if args.freeze_all_except_head: # freeze everything except segmentation head
        print("Freezing all parameters except segmentation head...")
        for param in net.parameters():
            param.requires_grad = False
        for param in net.segmentation_head.parameters():
            param.requires_grad = True

    else: # selective freezing
        if args.freeze_backbone:
            print("Freezing ResNet50 backbone...")
            for name, param in net.named_parameters():
                if 'transformer.embeddings.hybrid_model' in name:
                    param.requires_grad = False
        
        if args.freeze_transformer:
            print("Freezing transformer encoder...")
            for name, param in net.named_parameters():
                if 'transformer.encoder' in name or 'transformer.embeddings.patch_embeddings' in name:
                    param.requires_grad = False
        
        if args.freeze_decoder:
            print("Freezing decoder...")
            for name, param in net.named_parameters():
                if 'decoder' in name:
                    param.requires_grad = False

    # Print trainable parameters summary
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"\nTrainable: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)")

    # Print breakdown by component
    components = {
        'Backbone': 'transformer.embeddings.hybrid_model',
        'Transformer': 'transformer.encoder',
        'Decoder': 'decoder',
        'Seg Head': 'segmentation_head'
    }

    for comp_name, prefix in components.items():
        comp_params = sum(p.numel() for n, p in net.named_parameters() 
                        if prefix in n)
        comp_trainable = sum(p.numel() for n, p in net.named_parameters() 
                            if prefix in n and p.requires_grad)
        if comp_params > 0:
            print(f"  {comp_name}: {comp_trainable:,} / {comp_params:,} trainable")

    print("=" * 60 + "\n")

    trainer = {'dcisHE': trainer_he,}
    trainer[dataset_name](args, net, snapshot_path)