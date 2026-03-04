import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage
from scipy.ndimage import zoom

IGNORE_INDEX = 255

def random_rot_flip(image, label): # same as synapse
    # image: (H,W,3), label: (H,W)
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0, 1))
    label = np.rot90(label, k, axes=(0, 1))

    axis = np.random.randint(0, 2)  # 0=y or 1=x
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label): # (basically) the same as synapse
    angle = np.random.randint(-20, 20)
    # For RGB, rotate each channel; order=3 for image, 0 for label
    image = ndimage.rotate(image, angle, axes=(0, 1), order=3, reshape=False, mode="reflect")
    label = ndimage.rotate(label, angle, axes=(0, 1), order=0, reshape=False, mode="nearest")
    return image, label

class RandomGeneratorRGB(object):
    def __init__(self, output_size):
        self.output_size = output_size  # (H,W)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]  # image (H,W,3), label (H,W)

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        h, w = image.shape[0], image.shape[1]
        oh, ow = self.output_size # output dimensions

        if (h != oh) or (w != ow):
            image = zoom(image, (oh / h, ow / w, 1), order=3)   # keep channels
            label = zoom(label, (oh / h, ow / w), order=0)

        # HWC -> CHW
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)

        # normalize if needed (your images look like 0..255 floats)
        if image.max() > 1.0:
            image = image / 255.0

        label = torch.from_numpy(label.astype(np.int64))

        return {"image": image, "label": label}
    
class dcisHE_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split="train", transform=None,
                 image_key="image", label_key="label"):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, split + ".txt")).readlines()
        self.data_dir = base_dir
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx].strip("\n")

        # allow list entries to be either "tile_001" or "tile_001.npz"
        if name.endswith(".npz"):
            path = os.path.join(self.data_dir, name)
            case_name = name[:-4]
        else:
            path = os.path.join(self.data_dir, name + ".npz")
            case_name = name

        data = np.load(path)
        image = data[self.image_key]   # expected (224,224,3) float32
        label = data[self.label_key]   # expected (224,224) uint8/int, values 0..2 plus 255 ignore

        sample = {"image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)

        sample["case_name"] = case_name
        return sample