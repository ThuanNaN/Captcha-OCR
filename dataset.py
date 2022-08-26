import os
from pathlib import Path
from torch import utils

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

from utils import label2int

def get_chars():
    chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    return chars

def get_vocab(chars):
    vocab = {c: i for i, c in enumerate(sorted(chars), 1)}
    return vocab

class Captcha(Dataset):
    def __init__(self, root, transform=None, sampler=None, binarize=False):
        self.sampler = sampler
        if self.sampler:
            image_dir = sorted(list(map(str, list(Path(root).glob("*.png")))))
            self.images = [image_dir[i] for i in self.sampler.indices]
        else:
            self.images = sorted(list(map(str, list(Path(root).glob("*.png")))))

        self.labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in self.images]
        self.transform = transform
        self.binarize = binarize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        img = img.convert('RGB')
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        elif self.binarize:
            img = transforms.ToTensor()(img)
            img = (img > torch.tensor([128/255], dtype=torch.float32)) * 1.0
            
        else:
            img = transforms.ToTensor()(img)

        chars = get_chars()
        vocab = get_vocab(chars)
        label = torch.as_tensor(label2int(label, vocab), dtype=torch.float32)
        return img, label