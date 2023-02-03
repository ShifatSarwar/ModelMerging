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
gpuCore = 1
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
        loss_resnet18 = F.cross_entropy(out[0], labels) # Calculate loss
        loss_vgg16 = F.cross_entropy(out[1], labels) # Calculate loss
        loss_vgg13 = F.cross_entropy(out[2], labels) # Calculate loss
        loss_vgg11 = F.cross_entropy(out[3], labels) # Calculate loss
        return loss_resnet18, loss_vgg16, loss_vgg13, loss_vgg11
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions

        loss_resnet18 = F.cross_entropy(out[0], labels) # Calculate loss
        loss_vgg16 = F.cross_entropy(out[1], labels) # Calculate loss
        loss_vgg13 = F.cross_entropy(out[2], labels) # Calculate loss
        loss_vgg11 = F.cross_entropy(out[3], labels) # Calculate loss

        acc_resnet18 = accuracy(out[0], labels)           # Calculate accuracy
        acc_vgg16 = accuracy(out[1], labels)
        acc_vgg13 = accuracy(out[2], labels)
        acc_vgg11 = accuracy(out[3], labels)

        return [{'val_loss_resnet18': loss_resnet18.detach(), 'val_acc_resnet18': acc_resnet18},
                   {'val_loss_vgg16': loss_vgg16.detach(), 'val_acc_vgg16': acc_vgg16},
                   {'val_loss_vgg13': loss_vgg13.detach(), 'val_acc_vgg13': acc_vgg13},
                   {'val_loss_vgg11': loss_vgg11.detach(), 'val_acc_vgg11': acc_vgg11}]
    

    def validation_epoch_end(self, outputs):
        batch_losses_resnet18 = [x[0]['val_loss_resnet18'] for x in outputs]
        epoch_loss_resnet18 = torch.stack(batch_losses_resnet18).mean()   # Combine losses
        batch_accs_resnet18 = [x[0]['val_acc_resnet18'] for x in outputs]
        epoch_acc_resnet18 = torch.stack(batch_accs_resnet18).mean()      # Combine accuracies

        batch_losses_vgg16 = [x[1]['val_loss_vgg16'] for x in outputs]
        epoch_loss_vgg16 = torch.stack(batch_losses_vgg16).mean()   # Combine losses
        batch_accs_vgg16 = [x[1]['val_acc_vgg16'] for x in outputs]
        epoch_acc_vgg16 = torch.stack(batch_accs_vgg16).mean()      # Combine accuracies

        batch_losses_vgg13 = [x[2]['val_loss_vgg13'] for x in outputs]
        epoch_loss_vgg13 = torch.stack(batch_losses_vgg13).mean()   # Combine losses
        batch_accs_vgg13 = [x[2]['val_acc_vgg13'] for x in outputs]
        epoch_acc_vgg13 = torch.stack(batch_accs_vgg13).mean()      # Combine accuracies

        batch_losses_vgg11 = [x[3]['val_loss_vgg11'] for x in outputs]
        epoch_loss_vgg11 = torch.stack(batch_losses_vgg11).mean()   # Combine losses
        batch_accs_vgg11 = [x[3]['val_acc_vgg11'] for x in outputs]
        epoch_acc_vgg11 = torch.stack(batch_accs_vgg11).mean()      # Combine accuracies

        return [{'val_loss_resnet18': epoch_loss_resnet18.item(), 'val_acc_resnet18': epoch_acc_resnet18.item()},
                  {'val_loss_vgg16': epoch_loss_vgg16.item(), 'val_acc_vgg16': epoch_acc_vgg16.item()},
                  {'val_loss_vgg13': epoch_loss_vgg13.item(), 'val_acc_vgg13': epoch_acc_vgg13.item()},
                  {'val_loss_vgg11': epoch_loss_vgg11.item(), 'val_acc_vgg11': epoch_acc_vgg11.item()}]

    def epoch_end(self, epoch, results, val, start_time):
        timeSet = [25,50,75,100,150]
        if epoch in timeSet:
            timeTaken = time.time()-start_time
            dataLine = val+','+str(epoch)+','+str(timeTaken)
            addLine('dataList/list.csv',dataLine)
        
        result = results[0]
        print("Epoch [{}], last_lr_resnet18: {:.5f}, train_loss_resnet18: {:.4f}, val_loss_resnet18: {:.4f}, val_acc_resnet18: {:.4f}".format(
            epoch, result['lrs_resnet18'][-1], result['train_loss_resnet18'], result['val_loss_resnet18'], result['val_acc_resnet18']))
        print("---------------------------------------------------------------------------------------------------")
        
        result = results[1]
        print("Epoch [{}], last_lr_vgg16: {:.5f}, train_loss_vgg16: {:.4f}, val_loss_vgg16: {:.4f}, val_acc_vgg16: {:.4f}".format(
            epoch, result['lrs_vgg16'][-1], result['train_loss_vgg16'], result['val_loss_vgg16'], result['val_acc_vgg16']))
        print("---------------------------------------------------------------------------------------------------")
        
        result = results[2]
        print("Epoch [{}], last_lr_vgg13: {:.5f}, train_loss_vgg13: {:.4f}, val_loss_vgg13: {:.4f}, val_acc_vgg13: {:.4f}".format(
            epoch, result['lrs_vgg13'][-1], result['train_loss_vgg13'], result['val_loss_vgg13'], result['val_acc_vgg13']))
        print("---------------------------------------------------------------------------------------------------")

        result = results[3]
        print("Epoch [{}], last_lr_vgg11: {:.5f}, train_loss_vgg11: {:.4f}, val_loss_vgg11: {:.4f}, val_acc_vgg11: {:.4f}".format(
            epoch, result['lrs_vgg11'][-1], result['train_loss_vgg11'], result['val_loss_vgg11'], result['val_acc_vgg11']))
        print("---------------------------------------------------------------------------------------------------")
        

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
        self.in_planes_resnet = 64
        self.conv1_resnet = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_resnet = nn.BatchNorm2d(64)
        self.layer1_resnet = self._make_layer_resnet(block, 64, num_blocks[0], stride=1)
        self.layer2_resnet = self._make_layer_resnet(block, 128, num_blocks[1], stride=2)
        self.layer3_resnet = self._make_layer_resnet(block, 256, num_blocks[2], stride=2)
        self.layer4_resnet = self._make_layer_resnet(block, 512, num_blocks[3], stride=2)
        self.linear_resnet = nn.Linear(512*block.expansion, num_classes)
        
        # VGG16 Model Definition
        self.features_vgg = self._make_layers_vgg(cfg[vgg_name], num_channel)
        self.classifier_vgg = nn.Linear(512, num_classes)
        
        # VGG13 Model Definition
        vgg_name = 'VGG13'
        self.features_vgg_3 = self._make_layers_vgg(cfg[vgg_name], num_channel)
        self.classifier_vgg_3 = nn.Linear(512, num_classes)
        
        # VGG11 Model Definition
        vgg_name = 'VGG11'
        self.features_vgg_4 = self._make_layers_vgg(cfg[vgg_name], num_channel)
        self.classifier_vgg_4 = nn.Linear(512, num_classes)
        

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
        out_resnet_18 = F.relu(self.bn1_resnet(self.conv1_resnet(x)))
        out_resnet_18 = self.layer1_resnet(out_resnet_18)
        out_resnet_18 = self.layer2_resnet(out_resnet_18)
        out_resnet_18 = self.layer3_resnet(out_resnet_18)
        out_resnet_18 = self.layer4_resnet(out_resnet_18)
        out_resnet_18 = F.avg_pool2d(out_resnet_18, 4)
        out_resnet_18 = out_resnet_18.view(out_resnet_18.size(0), -1)
        out_resnet_18 = self.linear_resnet(out_resnet_18)
        
        out_vgg_16 = self.features_vgg(x)
        out_vgg_16 = out_vgg_16.view(out_vgg_16.size(0), -1)
        out_vgg_16 = self.classifier_vgg(out_vgg_16)
        
        out_vgg_13 = self.features_vgg_3(x)
        out_vgg_13 = out_vgg_13.view(out_vgg_13.size(0), -1)
        out_vgg_13 = self.classifier_vgg_3(out_vgg_13)
        
        out_vgg_11 = self.features_vgg_4(x)
        out_vgg_11 = out_vgg_11.view(out_vgg_11.size(0), -1)
        out_vgg_11 = self.classifier_vgg_4(out_vgg_11)
        
        return out_resnet_18, out_vgg_16, out_vgg_13, out_vgg_11

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

    resnet18_params = []
    vgg16_params = []
    vgg13_params = []
    vgg11_params = []
    # print(type(model.parameters()))
    # print(model.parameters())

    for name, params in model.named_parameters():
        if '_resnet' in name:
            resnet18_params.append(params)
        elif '_vgg_3' in name:
            vgg13_params.append(params)
        elif '_vgg_4' in name:
            vgg11_params.append(params)
        else:
            vgg16_params.append(params)

    # model_params=[] 
    # for x in model.parameters():
    #     model_params.append(x)
    
        
#         # Set up cutom optimizer with weight decay
    optimizerResnet18 = opt_func(resnet18_params, max_lr, weight_decay=weight_decay)
    optimizerVGG11 = opt_func(vgg11_params, max_lr, weight_decay=weight_decay)
    optimizerVGG13 = opt_func(vgg13_params, max_lr, weight_decay=weight_decay)
    optimizerVGG16 = opt_func(vgg16_params, max_lr, weight_decay=weight_decay)
    
    # Set up one-cycle learning rate scheduler
    schedResnet18 = torch.optim.lr_scheduler.OneCycleLR(optimizerResnet18, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    schedVGG11 = torch.optim.lr_scheduler.OneCycleLR(optimizerVGG11, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    schedVGG13 = torch.optim.lr_scheduler.OneCycleLR(optimizerVGG13, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    schedVGG16 = torch.optim.lr_scheduler.OneCycleLR(optimizerVGG16, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses_resnet18 = []
        train_losses_vgg16 = []
        train_losses_vgg11 = []
        train_losses_vgg13 = []
        lrs_resnet18 = []
        lrs_vgg16 = []
        lrs_vgg11 = []
        lrs_vgg13 = []
    
        for batch in train_dl:

            loss_resnet18, loss_vgg16, loss_vgg13, loss_vgg11  = model.training_step(batch)

            train_losses_resnet18.append(loss_resnet18)
            loss_resnet18.backward()

            train_losses_vgg11.append(loss_vgg11)
            loss_vgg11.backward()

            train_losses_vgg13.append(loss_vgg13)
            loss_vgg13.backward()

            train_losses_vgg16.append(loss_vgg16)
            loss_vgg16.backward()
            
            optimizerResnet18.step()
            optimizerResnet18.zero_grad()

            optimizerVGG11.step()
            optimizerVGG11.zero_grad()

            optimizerVGG13.step()
            optimizerVGG13.zero_grad()

            optimizerVGG16.step()
            optimizerVGG16.zero_grad()

            # Record & update learning rate
            lrs_resnet18.append(get_lr(optimizerResnet18))
            schedResnet18.step()

            # Record & update learning rate
            lrs_vgg11.append(get_lr(optimizerVGG11))
            schedVGG11.step()

            lrs_vgg13.append(get_lr(optimizerVGG13))
            schedVGG13.step()

            lrs_vgg16.append(get_lr(optimizerVGG16))
            schedVGG16.step()
        
        # Validation phase
        result = evaluate(model, valid_dl)
        result_resnet18 = result[0]
        result_vgg16 = result[1]
        result_vgg13 = result[2]
        result_vgg11 = result[3]

        result_resnet18['train_loss_resnet18'] = torch.stack(train_losses_resnet18).mean().item()
        result_resnet18['lrs_resnet18'] = lrs_resnet18

        result_vgg11['train_loss_vgg11'] = torch.stack(train_losses_vgg11).mean().item()
        result_vgg11['lrs_vgg11'] = lrs_vgg11

        result_vgg13['train_loss_vgg13'] = torch.stack(train_losses_vgg13).mean().item()
        result_vgg13['lrs_vgg13'] = lrs_vgg13

        result_vgg16['train_loss_vgg16'] = torch.stack(train_losses_vgg16).mean().item()
        result_vgg16['lrs_vgg16'] = lrs_vgg16

        result = [result_resnet18, result_vgg16, result_vgg13, result_vgg11]
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