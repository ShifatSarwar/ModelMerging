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
import torch.multiprocessing as mp

# Assign GPU for process
gpuCore = 0
# Assign Number of epochs
epochs = 100
# Choose Dataset
dataset1 = 'cifar10'
dataset2 = 'emnist_bymerge'
dataset3 = ''
dataset4 = ''
dataset5 = ''
# Mode 1 for entire Dataset
# Mode 2 for half the Dataset
# Mode 3 for Similar Dataset
mode = 1
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

batch_size = 64

# PyTorch data loaders
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)


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

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss_resnet50 = F.cross_entropy(out[0], labels) # Calculate loss
        loss_resnet101 = F.cross_entropy(out[1], labels) # Calculate loss
        loss_resnet152 = F.cross_entropy(out[2], labels) # Calculate loss
        loss_vgg19 = F.cross_entropy(out[3], labels) # Calculate loss
        return loss_resnet50, loss_resnet101, loss_resnet152, loss_vgg19
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss_resnet50 = F.cross_entropy(out[0], labels) # Calculate loss
        loss_resnet101 = F.cross_entropy(out[1], labels) # Calculate loss
        loss_resnet152 = F.cross_entropy(out[2], labels) # Calculate loss
        loss_vgg19 = F.cross_entropy(out[3], labels) # Calculate loss
        acc_resnet50 = accuracy(out[0], labels)           # Calculate accuracy
        acc_resnet101 = accuracy(out[1], labels)           # Calculate accuracy
        acc_resnet152 = accuracy(out[2], labels)
        acc_vgg19 = accuracy(out[3], labels)

        return [{'val_loss_resnet50': loss_resnet50.detach(), 'val_acc_resnet50': acc_resnet50},
                   {'val_loss_resnet101': loss_resnet101.detach(), 'val_acc_resnet101': acc_resnet101},
                   {'val_loss_resnet152': loss_resnet152.detach(), 'val_acc_resnet152': acc_resnet152},
                   {'val_loss_vgg19': loss_vgg19.detach(), 'val_acc_vgg19': acc_vgg19}]
    

    def validation_epoch_end(self, outputs):
        batch_losses_resnet50 = [x[0]['val_loss_resnet50'] for x in outputs]
        epoch_loss_resnet50 = torch.stack(batch_losses_resnet50).mean()   # Combine losses
        batch_accs_resnet50 = [x[0]['val_acc_resnet50'] for x in outputs]
        epoch_acc_resnet50 = torch.stack(batch_accs_resnet50).mean()      # Combine accuracies

        batch_losses_resnet101 = [x[1]['val_loss_resnet101'] for x in outputs]
        epoch_loss_resnet101 = torch.stack(batch_losses_resnet101).mean()   # Combine losses
        batch_accs_resnet101 = [x[1]['val_acc_resnet101'] for x in outputs]
        epoch_acc_resnet101 = torch.stack(batch_accs_resnet101).mean()      # Combine accuracies

        batch_losses_resnet152 = [x[2]['val_loss_resnet152'] for x in outputs]
        epoch_loss_resnet152 = torch.stack(batch_losses_resnet152).mean()   # Combine losses
        batch_accs_resnet152 = [x[2]['val_acc_resnet152'] for x in outputs]
        epoch_acc_resnet152 = torch.stack(batch_accs_resnet152).mean()      # Combine accuracies

        batch_losses_vgg19 = [x[3]['val_loss_vgg19'] for x in outputs]
        epoch_loss_vgg19 = torch.stack(batch_losses_vgg19).mean()   # Combine losses
        batch_accs_vgg19 = [x[3]['val_acc_vgg19'] for x in outputs]
        epoch_acc_vgg19 = torch.stack(batch_accs_vgg19).mean()      # Combine accuracies

        return [{'val_loss_resnet50': epoch_loss_resnet50.item(), 'val_acc_resnet50': epoch_acc_resnet50.item()},
                  {'val_loss_resnet101': epoch_loss_resnet101.item(), 'val_acc_resnet101': epoch_acc_resnet101.item()},
                  {'val_loss_resnet152': epoch_loss_resnet152.item(), 'val_acc_resnet152': epoch_acc_resnet152.item()},
                  {'val_loss_vgg19': epoch_loss_vgg19.item(), 'val_acc_vgg19': epoch_acc_vgg19.item()}]

    def epoch_end(self, epoch, results, val, start_time):
        timeSet = [25,50,75,100,150]
        if epoch in timeSet:
            timeTaken = time.time()-start_time
            dataLine = val+','+str(epoch)+','+str(timeTaken)
            addLine('dataList/list.csv',dataLine)
        
        result = results[0]
        print("Epoch [{}], last_lr_resnet50: {:.5f}, train_loss_resnet50: {:.4f}, val_loss_resnet50: {:.4f}, val_acc_resnet50: {:.4f}".format(
            epoch, result['lrs_resnet50'][-1], result['train_loss_resnet50'], result['val_loss_resnet50'], result['val_acc_resnet50']))
        print("---------------------------------------------------------------------------------------------------")
        result = results[1]
        print("Epoch [{}], last_lr_resnet101: {:.5f}, train_loss_resnet101: {:.4f}, val_loss_resnet101: {:.4f}, val_acc_resnet101: {:.4f}".format(
            epoch, result['lrs_resnet101'][-1], result['train_loss_resnet101'], result['val_loss_resnet101'], result['val_acc_resnet101']))
        print("---------------------------------------------------------------------------------------------------")
        result = results[2]
        print("Epoch [{}], last_lr_resnet152: {:.5f}, train_loss_resnet152: {:.4f}, val_loss_resnet152: {:.4f}, val_acc_resnet152: {:.4f}".format(
            epoch, result['lrs_resnet152'][-1], result['train_loss_resnet152'], result['val_loss_resnet152'], result['val_acc_resnet152']))
        print("---------------------------------------------------------------------------------------------------")
        result = results[3]
        print("Epoch [{}], last_lr_vgg19: {:.5f}, train_loss_vgg19: {:.4f}, val_loss_vgg19: {:.4f}, val_acc_vgg19: {:.4f}".format(
            epoch, result['lrs_vgg19'][-1], result['train_loss_vgg19'], result['val_loss_vgg19'], result['val_acc_vgg19']))

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
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10, num_channel=3, vgg_name='VGG16'):
        super(HybridModel, self).__init__()
        
        # ResNet-18 Model Definition
        num_blocks = [3, 4, 6, 3]
        self.in_planes_resnet = 64
        self.conv1_resnet = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_resnet = nn.BatchNorm2d(64)
        self.layer1_resnet = self._make_layer_resnet(block, 64, num_blocks[0], stride=1)
        self.layer2_resnet = self._make_layer_resnet(block, 128, num_blocks[1], stride=2)
        self.layer3_resnet = self._make_layer_resnet(block, 256, num_blocks[2], stride=2)
        self.layer4_resnet = self._make_layer_resnet(block, 512, num_blocks[3], stride=2)
        self.linear_resnet = nn.Linear(512*block.expansion, num_classes)
        
        # ResNet-101 Model Definition
        num_blocks = [3, 4, 23, 3]
        self.in_planes_resnet = 64
        self.conv1_resnet_2 = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_resnet_2 = nn.BatchNorm2d(64)
        self.layer1_resnet_2 = self._make_layer_resnet(block, 64, num_blocks[0], stride=1)
        self.layer2_resnet_2 = self._make_layer_resnet(block, 128, num_blocks[1], stride=2)
        self.layer3_resnet_2 = self._make_layer_resnet(block, 256, num_blocks[2], stride=2)
        self.layer4_resnet_2 = self._make_layer_resnet(block, 512, num_blocks[3], stride=2)
        self.linear_resnet_2 = nn.Linear(512*block.expansion, num_classes)
        
        # resnet152 Model Definition
        num_blocks = [3, 8, 36, 3]
        self.in_planes_resnet = 64
        self.conv1_resnet_3 = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_resnet_3 = nn.BatchNorm2d(64)
        self.layer1_resnet_3 = self._make_layer_resnet(block, 64, num_blocks[0], stride=1)
        self.layer2_resnet_3 = self._make_layer_resnet(block, 128, num_blocks[1], stride=2)
        self.layer3_resnet_3 = self._make_layer_resnet(block, 256, num_blocks[2], stride=2)
        self.layer4_resnet_3 = self._make_layer_resnet(block, 512, num_blocks[3], stride=2)
        self.linear_resnet_3 = nn.Linear(512*block.expansion, num_classes)
        
        # VGG19 Model Definition
        vgg_name = 'VGG19'
        self.features_vgg_2 = self._make_layers_vgg(cfg[vgg_name], num_channel)
        self.classifier_vgg_2 = nn.Linear(512, num_classes)

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
        out_resnet_50 = F.relu(self.bn1_resnet(self.conv1_resnet(x)))
        out_resnet_50 = self.layer1_resnet(out_resnet_50)
        out_resnet_50 = self.layer2_resnet(out_resnet_50)
        out_resnet_50 = self.layer3_resnet(out_resnet_50)
        out_resnet_50 = self.layer4_resnet(out_resnet_50)
        out_resnet_50 = F.avg_pool2d(out_resnet_50, 4)
        out_resnet_50 = out_resnet_50.view(out_resnet_50.size(0), -1)
        out_resnet_50 = self.linear_resnet(out_resnet_50)
        
        out_resnet_101 = F.relu(self.bn1_resnet_2(self.conv1_resnet_2(x)))
        out_resnet_101 = self.layer1_resnet_2(out_resnet_101)
        out_resnet_101 = self.layer2_resnet_2(out_resnet_101)
        out_resnet_101 = self.layer3_resnet_2(out_resnet_101)
        out_resnet_101 = self.layer4_resnet_2(out_resnet_101)
        out_resnet_101 = F.avg_pool2d(out_resnet_101, 4)
        out_resnet_101 = out_resnet_101.view(out_resnet_101.size(0), -1)
        out_resnet_101 = self.linear_resnet_2(out_resnet_101)
        
        out_resnet_152 = F.relu(self.bn1_resnet_3(self.conv1_resnet_3(x)))
        out_resnet_152 = self.layer1_resnet_3(out_resnet_152)
        out_resnet_152 = self.layer2_resnet_3(out_resnet_152)
        out_resnet_152 = self.layer3_resnet_3(out_resnet_152)
        out_resnet_152 = self.layer4_resnet_3(out_resnet_152)
        out_resnet_152 = F.avg_pool2d(out_resnet_152, 4)
        out_resnet_152 = out_resnet_152.view(out_resnet_152.size(0), -1)
        out_resnet_152 = self.linear_resnet_3(out_resnet_152)
        
        out_vgg_19 = self.features_vgg_2(x)
        out_vgg_19 = out_vgg_19.view(out_vgg_19.size(0), -1)
        out_vgg_19 = self.classifier_vgg_2(out_vgg_19)
        
        return out_resnet_50, out_resnet_101, out_resnet_152, out_vgg_19

model = to_device(HybridModel(num_classes=10, num_channel=3), device)

# Main evaluator
@torch.no_grad()
def evaluate(model, valid_dl):
    model.eval()
    outputs = [model.validation_step(batch) for batch in valid_dl]
    return model.validation_epoch_end(outputs)

# def evaluate2(model, valid_dl):
#     model.eval()
#     outputs = [model.validation_step2(batch) for batch in valid_dl]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, val, start_time, max_lr, model, train_dl, valid_dl, 
                  weight_decay=0, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    resnet50_params = []
    resnet101_params = []
    resnet152_params = []
    vgg19_params = []
    # print(type(model.parameters()))
    # print(model.parameters())

    for name, params in model.named_parameters():
        if '_resnet_2' in name:
            resnet101_params.append(params)
        elif '_resnet_3' in name:
            resnet152_params.append(params)
        elif '_vgg_2' in name:
            vgg19_params.append(params)
        else:
            resnet50_params.append(params)

    # model_params=[] 
    # for x in model.parameters():
    #     model_params.append(x)
    
        
#         # Set up cutom optimizer with weight decay
    optimizerResnet50 = opt_func(resnet50_params, max_lr, weight_decay=weight_decay)
    optimizerResnet101 = opt_func(resnet101_params, max_lr, weight_decay=weight_decay)
    optimizerResnet152 = opt_func(resnet152_params, max_lr, weight_decay=weight_decay)
    optimizerVGG19 = opt_func(vgg19_params, max_lr, weight_decay=weight_decay)

    # Set up one-cycle learning rate scheduler
    schedResnet50 = torch.optim.lr_scheduler.OneCycleLR(optimizerResnet50, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    schedResnet101 = torch.optim.lr_scheduler.OneCycleLR(optimizerResnet101, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    schedResnet152 = torch.optim.lr_scheduler.OneCycleLR(optimizerResnet152, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    schedVGG19 = torch.optim.lr_scheduler.OneCycleLR(optimizerVGG19, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses_resnet50 = []
        train_losses_resnet152 = []
        train_losses_resnet101 = []
        train_losses_vgg19 = []
        lrs_resnet50 = []
        lrs_resnet152 = []
        lrs_resnet101 = []
        lrs_vgg19 = []

        for batch in train_dl:

            loss_resnet50, loss_resnet101, loss_resnet152, loss_vgg19 = model.training_step(batch)

            train_losses_resnet50.append(loss_resnet50)
            loss_resnet50.backward()

            train_losses_resnet101.append(loss_resnet101)
            loss_resnet101.backward()

            train_losses_resnet152.append(loss_resnet152)
            loss_resnet152.backward()

            train_losses_vgg19.append(loss_vgg19)
            loss_vgg19.backward()
            
            # Gradient clipping
            # if grad_clip: 
            #     nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizerResnet50.step()
            optimizerResnet50.zero_grad()

            optimizerResnet101.step()
            optimizerResnet101.zero_grad()

            optimizerResnet152.step()
            optimizerResnet152.zero_grad()

            optimizerVGG19.step()
            optimizerVGG19.zero_grad()
            
            # Record & update learning rate
            lrs_resnet50.append(get_lr(optimizerResnet50))
            schedResnet50.step()

            lrs_resnet101.append(get_lr(optimizerResnet101))
            schedResnet101.step()

            # Record & update learning rate
            lrs_resnet152.append(get_lr(optimizerResnet152))
            schedResnet152.step()

            lrs_vgg19.append(get_lr(optimizerVGG19))
            schedVGG19.step()
        
        # Validation phase
        result = evaluate(model, valid_dl)
        result_resnet50 = result[0]
        result_resnet101 = result[1]
        result_resnet152 = result[2]
        result_vgg19 = result[3]

        result_resnet50['train_loss_resnet50'] = torch.stack(train_losses_resnet50).mean().item()
        result_resnet50['lrs_resnet50'] = lrs_resnet50

        result_resnet101['train_loss_resnet101'] = torch.stack(train_losses_resnet101).mean().item()
        result_resnet101['lrs_resnet101'] = lrs_resnet101

        result_resnet152['train_loss_resnet152'] = torch.stack(train_losses_resnet152).mean().item()
        result_resnet152['lrs_resnet152'] = lrs_resnet152

        result_vgg19['train_loss_vgg19'] = torch.stack(train_losses_vgg19).mean().item()
        result_vgg19['lrs_vgg19'] = lrs_vgg19

        result = [result_resnet50, result_resnet101, result_resnet152, result_vgg19]
        model.epoch_end(epoch, result, val, start_time)
        history.append(result)

    return history

max_lr = 0.01
# grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam

def getHistory(val,start_time):
    history = [evaluate(model, valid_dl)]
    history += fit_one_cycle(epochs,val, start_time, max_lr, model, train_dl, valid_dl, 
                            #  grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
    
    # evaluate2(model, valid_dl)
    torch.save(model.state_dict(), 'saved_models/model_'+val+'.sav')
    return history