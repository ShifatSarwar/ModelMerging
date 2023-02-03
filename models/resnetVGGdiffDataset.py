import os
import torch
import time
import torchvision
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from memory_profiler import profile
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Assign GPU for process
gpuCore = 2
# Assign Number of epochs
epochs = 100
# Choose Dataset
dataset1 = 'cifar10'
dataset2 = 'cifar5'
dataset3 = ''
dataset4 = ''
dataset5 = ''
# Mode 1 for entire Dataset
# Mode 2 for half the Dataset
# Mode 3 for Similar Dataset
mode = 1
data_dir = './data/' + dataset1
data_dir2 = './data/' + dataset2

# Data transforms (normalization & data augmentation)
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = tt.Compose([ tt.ToTensor(), 
                         tt.Normalize(*stats,inplace=True)])
valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

data_dir_test = './data/' + dataset1
data_dir_test2 = './data/' + dataset2
# PyTorch datasets
train_ds = ImageFolder(data_dir+'/train', train_tfms)
valid_ds = ImageFolder(data_dir_test+'/test', valid_tfms)

# PyTorch datasets
train_ds2 = ImageFolder(data_dir2+'/train', train_tfms)
valid_ds2 = ImageFolder(data_dir_test2+'/test', valid_tfms)

batch_size = 64

# PyTorch data loaders
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)

# PyTorch data loaders
train_dl2 = DataLoader(train_ds2, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valid_dl2 = DataLoader(valid_ds2, batch_size*2, num_workers=3, pin_memory=True)


def addLine(name, line):
    with open(name, 'a') as f:
       f.write(line)
       f.write("\n")
    f.close()
    
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

device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
train_dl2 = DeviceDataLoader(train_dl2, device)
valid_dl2 = DeviceDataLoader(valid_dl2, device)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step_resnet(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss_resnet = F.cross_entropy(out[0], labels) # Calculate loss
        return loss_resnet
    
    def training_step_vgg(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss_vgg = F.cross_entropy(out[1], labels) # Calculate loss
        return loss_vgg
    
    def validation_step_resnet(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss_resnet = F.cross_entropy(out[0], labels)   # Calculate loss Resnet
        acc_resnet = accuracy(out[0], labels)           # Calculate accuracy
        return [{'val_loss_resnet': loss_resnet.detach(), 'val_acc_resnet': acc_resnet}]
    
    def validation_step_vgg(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss_vgg = F.cross_entropy(out[1], labels)   # Calculate loss Resnet
        acc_vgg = accuracy(out[1], labels)
        return [{'val_loss_vgg': loss_vgg.detach(), 'val_acc_vgg': acc_vgg}]
    

    def validation_epoch_end_resnet(self, outputs):
        batch_losses_resnet = [x[0]['val_loss_resnet'] for x in outputs]
        epoch_loss_resnet = torch.stack(batch_losses_resnet).mean()   # Combine losses
        batch_accs_resnet = [x[0]['val_acc_resnet'] for x in outputs]
        epoch_acc_resnet = torch.stack(batch_accs_resnet).mean()      # Combine accuracies
        return [{'val_loss_resnet': epoch_loss_resnet.item(), 'val_acc_resnet': epoch_acc_resnet.item()}]

    def validation_epoch_end_vgg(self, outputs):
        batch_losses_vgg = [x[0]['val_loss_vgg'] for x in outputs]
        epoch_loss_vgg = torch.stack(batch_losses_vgg).mean()   # Combine losses
        batch_accs_vgg= [x[0]['val_acc_vgg'] for x in outputs]
        epoch_acc_vgg= torch.stack(batch_accs_vgg).mean()      # Combine accuracies
        return [{'val_loss_vgg': epoch_loss_vgg.item(), 'val_acc_vgg': epoch_acc_vgg.item()}]

    def epoch_end(self, epoch, results, val, start_time):
        timeSet = [25,50,75,100,150]
        if epoch in timeSet:
            timeTaken = time.time()-start_time
            dataLine = val+','+str(epoch)+','+str(timeTaken)
            addLine('dataList/list.csv',dataLine)
        result = results[0][0]
        print("Epoch [{}], last_lr_resnet: {:.5f}, train_loss_resnet: {:.4f}, val_loss_resnet: {:.4f}, val_acc_resnet: {:.4f}".format(
            epoch, result['lrs_resnet'][-1], result['train_loss_resnet'], result['val_loss_resnet'], result['val_acc_resnet']))
        result = results[1][0]
        print("Epoch [{}], last_lr_vgg: {:.5f}, train_loss_vgg: {:.4f}, val_loss_vgg: {:.4f}, val_acc_vgg: {:.4f}".format(
            epoch, result['lrs_vgg'][-1], result['train_loss_vgg'], result['val_loss_vgg'], result['val_acc_vgg']))


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class HybridModel(ImageClassificationBase):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10, num_classes2=5, num_channel=3, vgg_name='VGG16'):
        super(HybridModel, self).__init__()
        
        # ResNet Model Definition
        self.in_planes_resnet = 64
        self.conv1_resnet = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_resnet = nn.BatchNorm2d(64)
        self.layer1_resnet = self._make_layer_resnet(block, 64, num_blocks[0], stride=1)
        self.layer2_resnet = self._make_layer_resnet(block, 128, num_blocks[1], stride=2)
        self.layer3_resnet = self._make_layer_resnet(block, 256, num_blocks[2], stride=2)
        self.layer4_resnet = self._make_layer_resnet(block, 512, num_blocks[3], stride=2)
        self.linear_resnet = nn.Linear(512*block.expansion, num_classes)
        
        # VGG Model Definition
        self.features_vgg = self._make_layers_vgg(cfg[vgg_name], num_channel)
        self.classifier_vgg = nn.Linear(512, num_classes2)

    def _make_layer_resnet(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes_resnet, planes, stride))
            self.in_planes_resnet = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _make_layers_vgg(self, cfg, num_channels):
        layers = []
        in_channels = num_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out_resnet = F.relu(self.bn1_resnet(self.conv1_resnet(x)))
        out_resnet = self.layer1_resnet(out_resnet)
        out_resnet = self.layer2_resnet(out_resnet)
        out_resnet = self.layer3_resnet(out_resnet)
        out_resnet = self.layer4_resnet(out_resnet)
        out_resnet = F.avg_pool2d(out_resnet, 4)
        out_resnet = out_resnet.view(out_resnet.size(0), -1)
        out_resnet = self.linear_resnet(out_resnet)
        
        out_vgg = self.features_vgg(x)
        out_vgg = out_vgg.view(out_vgg.size(0), -1)
        out_vgg = self.classifier_vgg(out_vgg)
        
        return out_resnet, out_vgg

model = to_device(HybridModel(num_classes=10, num_classes2=5, num_channel=3), device)


# Main evaluator
@torch.no_grad()
def evaluate(model, valid_dl, valid_dl2):
    model.eval()
    outputs_resnet = [model.validation_step_resnet(batch) for batch in valid_dl]
    outputs_vgg = [model.validation_step_vgg(batch) for batch in valid_dl2]
    return model.validation_epoch_end_resnet(outputs_resnet), model.validation_epoch_end_vgg(outputs_vgg)

# def evaluate2(model, valid_dl):
#     model.eval()
#     outputs = [model.validation_step2(batch) for batch in valid_dl]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, val, start_time, max_lr, model, train_dl, valid_dl, train_dl2, valid_dl2,
                  weight_decay=0, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    resnet_params = []
    vgg_params = []
    # print(type(model.parameters()))
    # print(model.parameters())

    for name, params in model.named_parameters():
        if '_resnet' in name:
            resnet_params.append(params)
        else:
            vgg_params.append(params)

    # model_params=[] 
    # for x in model.parameters():
    #     model_params.append(x)
    
        
#         # Set up cutom optimizer with weight decay
    optimizerResnet = opt_func(resnet_params, max_lr, weight_decay=weight_decay)
    optimizerVGG = opt_func(vgg_params, max_lr, weight_decay=weight_decay)

    # Set up one-cycle learning rate scheduler
    schedResnet = torch.optim.lr_scheduler.OneCycleLR(optimizerResnet, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    schedVGG = torch.optim.lr_scheduler.OneCycleLR(optimizerVGG, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses_resnet = []
        train_losses_vgg = []
        lrs_resnet = []
        lrs_vgg = []

        for batch in train_dl:
            loss_resnet = model.training_step_resnet(batch)
            train_losses_resnet.append(loss_resnet)
            loss_resnet.backward()
            optimizerResnet.step()
            optimizerResnet.zero_grad()
            # Record & update learning rate
            lrs_resnet.append(get_lr(optimizerResnet))
            schedResnet.step()

        for batch in train_dl2:
            loss_vgg = model.training_step_vgg(batch)
            train_losses_vgg.append(loss_vgg)
            loss_vgg.backward()
            optimizerVGG.step()
            optimizerVGG.zero_grad()
            # Record & update learning rate
            lrs_vgg.append(get_lr(optimizerVGG))
            schedVGG.step()
        
        # Validation phase
        result_resnet, result_vgg = evaluate(model, valid_dl, valid_dl2)

        result_resnet[0]['train_loss_resnet'] = torch.stack(train_losses_resnet).mean().item()
        result_resnet[0]['lrs_resnet'] = lrs_resnet

        result_vgg[0]['train_loss_vgg'] = torch.stack(train_losses_vgg).mean().item()
        result_vgg[0]['lrs_vgg'] = lrs_vgg

        result = [result_resnet, result_vgg]
        model.epoch_end(epoch, result, val, start_time)
        history.append(result)

    return history

max_lr = 0.01
# grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam

def getHistory(val,start_time):
    history = [evaluate(model, valid_dl, valid_dl2)]
    history += fit_one_cycle(epochs,val, start_time, max_lr, model, train_dl, valid_dl, train_dl2, valid_dl2,
                            #  grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
    
    
    # evaluate2(model, valid_dl)
    torch.save(model.state_dict(), 'saved_models/model_'+val+'.sav')
    return history


