import pandas as pd
import os
import torch
import time
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.metrics import *


# Choose Dataset
dataset1 = 'cifar10'
# Assign GPU for process
gpuCore = 2
batch_size = 64


def addLine(name, line):
    with open(name, 'a') as f:
       f.write(line)
       f.write("\n")
    f.close()

def getTrainCIFAR10():
    data_dir = './data/' + dataset1

    # Data transforms (normalization & data augmentation)
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tfms = tt.Compose([ tt.ToTensor(), 
                            tt.Normalize(*stats,inplace=True)])
    valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

    data_dir_test = './data/' + dataset1
    # PyTorch datasets
    train_ds = ImageFolder(data_dir+'/train', train_tfms)
    valid_ds = ImageFolder(data_dir_test+'/test', valid_tfms)

    # PyTorch data loaders
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)

    return train_dl, valid_dl

def getTrainCIFAR100():
    train_data = torchvision.datasets.CIFAR100('./', train=True, download=True)

    # Stick all the images together to form a 1600000 X 32 X 3 array
    x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])

    # calculate the mean and std along the (0, 1) axes
    mean = np.mean(x, axis=(0, 1))/255
    std = np.std(x, axis=(0, 1))/255
    # the the mean and std
    mean=mean.tolist()
    std=std.tolist()

    # Data transforms (normalization & data augmentation)
    # stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tfms = tt.Compose([tt.RandomCrop(32, padding=4,padding_mode='reflect'), 
                            tt.RandomHorizontalFlip(), 
                            tt.ToTensor(), 
                            tt.Normalize(mean,std,inplace=True)])

    valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(mean,std)])

# data_dir_test = './data/' + dataset1

    # PyTorch datasets CIFAR100
    train_ds = torchvision.datasets.CIFAR100("./", train=True, download=True,transform=train_tfms)
    valid_ds = torchvision.datasets.CIFAR100("./", train=False,download=True,transform=valid_tfms)


    # PyTorch data loaders
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)

    return train_dl, valid_dl



def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0))
        break


def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)



def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda:'+str(gpuCore))
    else:
        return torch.device('cpu')
    

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def getTrainTestLoaderCIFAR10():
    device = get_default_device()
    train_dl, valid_dl = getTrainCIFAR10()
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)
    return train_dl, valid_dl, device


def getTrainTestLoaderCIFAR100():
    device = get_default_device()
    train_dl, valid_dl = getTrainCIFAR100()
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)
    return train_dl, valid_dl, device
