## ADAPTED FROM TRANSUNET

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    ### OLD DICE LOSS
    # def forward(self, inputs, target, weight=None, softmax=False):
    #     if softmax:
    #         inputs = torch.softmax(inputs, dim=1)
    #     target = self._one_hot_encoder(target)
    #     if weight is None:
    #         weight = [1] * self.n_classes
    #     assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
    #     class_wise_dice = []
    #     loss = 0.0
    #     for i in range(0, self.n_classes):
    #         dice = self._dice_loss(inputs[:, i], target[:, i])
    #         class_wise_dice.append(1.0 - dice.item())
    #         loss += dice * weight[i]
    #     return loss / self.n_classes
    
    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        if weight is None:
            weight = [1] * self.n_classes

        # mask ignored pixels (so they contribute 0 to both prediction + target)
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()  # (B,H,W)
        else:
            valid_mask = torch.ones_like(target, dtype=torch.float32)

        target_1h = self._one_hot_encoder(target)  # (B,C,H,W), ignore pixels = 0 since 255 /= 1,2,3

        # apply mask to both prediction and target so ignore pixels contribute nothing
        valid_mask = valid_mask.unsqueeze(1)  # (B,1,H,W)
        inputs = inputs * valid_mask
        target_1h = target_1h * valid_mask # one hot target
    
        assert inputs.size() == target_1h.size(), \
            f'predict {inputs.size()} & target {target_1h.size()} shape do not match'

        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target_1h[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes