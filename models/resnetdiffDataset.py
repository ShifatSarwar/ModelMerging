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
from models.dataLoader import *

# Assign Number of epochs
epochs = 100



y_pred = []
y_true = []

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def accuracy2(outputs, labels):
    _, preds = torch.max(outputs, dim=1) 
    output = preds
    label = labels
    output = output.data.cpu().numpy()
    y_pred.extend(output)
    label = label.data.cpu().numpy()
    y_true.extend(label) # Save Truth
        


class ImageClassificationBase(nn.Module):
    def training_step_resnet(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss_resnet = F.cross_entropy(out[0], labels) # Calculate loss
        return loss_resnet
    
    def training_step_resnet_2(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss_resnet_2 = F.cross_entropy(out[1], labels) # Calculate loss
        return loss_resnet_2
    
    def validation_step_resnet(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss_resnet = F.cross_entropy(out[0], labels)   # Calculate loss Resnet
        acc_resnet = accuracy(out[0], labels)           # Calculate accuracy
        return [{'val_loss_resnet': loss_resnet.detach(), 'val_acc_resnet': acc_resnet}]
    
    def validation_step_resnet_2(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss_resnet_2 = F.cross_entropy(out[1], labels)   # Calculate loss Resnet
        acc_resnet_2 = accuracy(out[1], labels)
        return [{'val_loss_resnet_2': loss_resnet_2.detach(), 'val_acc_resnet_2': acc_resnet_2}]
    

    def validation_epoch_end_resnet(self, outputs):
        batch_losses_resnet = [x[0]['val_loss_resnet'] for x in outputs]
        epoch_loss_resnet = torch.stack(batch_losses_resnet).mean()   # Combine losses
        batch_accs_resnet = [x[0]['val_acc_resnet'] for x in outputs]
        epoch_acc_resnet = torch.stack(batch_accs_resnet).mean()      # Combine accuracies
        return [{'val_loss_resnet': epoch_loss_resnet.item(), 'val_acc_resnet': epoch_acc_resnet.item()}]

    def validation_epoch_end_resnet_2(self, outputs):
        batch_losses_resnet_2 = [x[0]['val_loss_resnet_2'] for x in outputs]
        epoch_loss_resnet_2 = torch.stack(batch_losses_resnet_2).mean()   # Combine losses
        batch_accs_resnet_2= [x[0]['val_acc_resnet_2'] for x in outputs]
        epoch_acc_resnet_2= torch.stack(batch_accs_resnet_2).mean()      # Combine accuracies
        return [{'val_loss_resnet_2': epoch_loss_resnet_2.item(), 'val_acc_resnet_2': epoch_acc_resnet_2.item()}]

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
        print("Epoch [{}], last_lr_resnet_2: {:.5f}, train_loss_resnet_2: {:.4f}, val_loss_resnet_2: {:.4f}, val_acc_resnet_2: {:.4f}".format(
            epoch, result['lrs_resnet_2'][-1], result['train_loss_resnet_2'], result['val_loss_resnet_2'], result['val_acc_resnet_2']))


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
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=100, num_classes2=10, num_channel=3, vgg_name='VGG16'):
        super(HybridModel, self).__init__()
        
        self.in_planes_resnet = 64   
        self.conv1_resnet = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_resnet = nn.BatchNorm2d(64)
        self.layer1_resnet = self._make_layer_resnet(block, 64, num_blocks[0], stride=1)
        self.layer2_resnet = self._make_layer_resnet(block, 128, num_blocks[1], stride=2)
        self.layer3_resnet = self._make_layer_resnet(block, 256, num_blocks[2], stride=2)
        self.layer4_resnet = self._make_layer_resnet(block, 512, num_blocks[3], stride=2)
        self.linear_resnet = nn.Linear(512*block.expansion, num_classes)
        
        
        # # ResNet-50 Model Definition
        num_blocks = [3, 4, 6, 3]
        block = Bottleneck
        self.in_planes_resnet = 64
        self.conv1_resnet_2 = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_resnet_2 = nn.BatchNorm2d(64)
        self.layer1_resnet_2 = self._make_layer_resnet(block, 64, num_blocks[0], stride=1)
        self.layer2_resnet_2 = self._make_layer_resnet(block, 128, num_blocks[1], stride=2)
        self.layer3_resnet_2 = self._make_layer_resnet(block, 256, num_blocks[2], stride=2)
        self.layer4_resnet_2 = self._make_layer_resnet(block, 512, num_blocks[3], stride=2)
        self.linear_resnet_2 = nn.Linear(512*block.expansion, num_classes2)

    def _make_layer_resnet(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes_resnet, planes, stride))
            self.in_planes_resnet = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _make_layers_resnet_2(self, cfg, num_channels):
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
        
        out_resnet_50 = F.relu(self.bn1_resnet_2(self.conv1_resnet_2(x)))
        out_resnet_50 = self.layer1_resnet_2(out_resnet_50)
        out_resnet_50 = self.layer2_resnet_2(out_resnet_50)
        out_resnet_50 = self.layer3_resnet_2(out_resnet_50)
        out_resnet_50 = self.layer4_resnet_2(out_resnet_50)
        out_resnet_50 = F.avg_pool2d(out_resnet_50, 4)
        out_resnet_50 = out_resnet_50.view(out_resnet_50.size(0), -1)
        out_resnet_50 = self.linear_resnet_2(out_resnet_50)
        
        
        return out_resnet_18, out_resnet_50

train_dl, valid_dl, device = getTrainTestLoaderCIFAR100()
train_dl2, valid_dl2, device = getTrainTestLoaderCIFAR10()

model = to_device(HybridModel(num_classes=100, num_classes2=10, num_channel=3), device)


# Main evaluator
@torch.no_grad()
def evaluate(model, valid_dl, valid_dl2):
    model.eval()
    outputs_resnet = [model.validation_step_resnet(batch) for batch in valid_dl]
    outputs_resnet_2 = [model.validation_step_resnet_2(batch) for batch in valid_dl2]
    return model.validation_epoch_end_resnet(outputs_resnet), model.validation_epoch_end_resnet_2(outputs_resnet_2)

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
    resnet_2_params = []
    # print(type(model.parameters()))
    # print(model.parameters())

    for name, params in model.named_parameters():
        if '_resnet_2' in name:
            resnet_2_params.append(params)
        else:
            resnet_params.append(params)

    # model_params=[] 
    # for x in model.parameters():
    #     model_params.append(x)
    
        
#         # Set up cutom optimizer with weight decay
    optimizerResnet = opt_func(resnet_params, max_lr, weight_decay=weight_decay)
    optimizerResnet2 = opt_func(resnet_2_params, max_lr, weight_decay=weight_decay)

    # Set up one-cycle learning rate scheduler
    schedResnet = torch.optim.lr_scheduler.OneCycleLR(optimizerResnet, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    schedResnet2 = torch.optim.lr_scheduler.OneCycleLR(optimizerResnet2, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses_resnet = []
        train_losses_resnet_2 = []
        lrs_resnet = []
        lrs_resnet_2 = []

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
            loss_resnet_2 = model.training_step_resnet_2(batch)
            train_losses_resnet_2.append(loss_resnet_2)
            loss_resnet_2.backward()
            optimizerResnet2.step()
            optimizerResnet2.zero_grad()
            # Record & update learning rate
            lrs_resnet_2.append(get_lr(optimizerResnet2))
            schedResnet2.step()
        
        # Validation phase
        result_resnet, result_resnet_2 = evaluate(model, valid_dl, valid_dl2)

        result_resnet[0]['train_loss_resnet'] = torch.stack(train_losses_resnet).mean().item()
        result_resnet[0]['lrs_resnet'] = lrs_resnet

        result_resnet_2[0]['train_loss_resnet_2'] = torch.stack(train_losses_resnet_2).mean().item()
        result_resnet_2[0]['lrs_resnet_2'] = lrs_resnet_2

        result = [result_resnet, result_resnet_2]
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


