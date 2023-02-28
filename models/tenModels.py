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
        loss_model1 = F.cross_entropy(out[0], labels) # Calculate loss
        loss_model2 = F.cross_entropy(out[1], labels) # Calculate loss
        loss_model3 = F.cross_entropy(out[2], labels) # Calculate loss
        loss_model4 = F.cross_entropy(out[3], labels) # Calculate loss
        loss_model5 = F.cross_entropy(out[4], labels) # Calculate loss
        loss_model6 = F.cross_entropy(out[5], labels) # Calculate loss
        loss_model7 = F.cross_entropy(out[6], labels) # Calculate loss
        loss_model8 = F.cross_entropy(out[7], labels) # Calculate loss
        loss_model9 = F.cross_entropy(out[8], labels) # Calculate loss
        loss_model10 = F.cross_entropy(out[9], labels) # Calculate loss
        return loss_model1, loss_model2, loss_model3, loss_model4, loss_model5, loss_model6, loss_model7, loss_model8, loss_model9, loss_model10
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions

        loss_model1 = F.cross_entropy(out[0], labels) # Calculate loss
        loss_model2 = F.cross_entropy(out[1], labels) # Calculate loss
        loss_model3 = F.cross_entropy(out[2], labels) # Calculate loss
        loss_model4 = F.cross_entropy(out[3], labels) # Calculate loss
        loss_model5 = F.cross_entropy(out[4], labels) # Calculate loss
        loss_model6 = F.cross_entropy(out[5], labels) # Calculate loss
        loss_model7 = F.cross_entropy(out[6], labels) # Calculate loss
        loss_model8 = F.cross_entropy(out[7], labels) # Calculate loss
        loss_model9 = F.cross_entropy(out[8], labels) # Calculate loss
        loss_model10 = F.cross_entropy(out[9], labels) # Calculate loss

        acc_model1 = accuracy(out[0], labels)           # Calculate accuracy
        acc_model2 = accuracy(out[1], labels)           # Calculate accuracy
        acc_model3 = accuracy(out[2], labels)           # Calculate accuracy
        acc_model4 = accuracy(out[3], labels)           # Calculate accuracy
        acc_model5 = accuracy(out[4], labels)
        acc_model6 = accuracy(out[5], labels)
        acc_model7 = accuracy(out[6], labels)
        acc_model8 = accuracy(out[7], labels)
        acc_model9 = accuracy(out[8], labels)
        acc_model10 = accuracy(out[9], labels)


        return [{'val_loss_model1': loss_model1.detach(), 'val_acc_model1': acc_model1},
                   {'val_loss_model2': loss_model2.detach(), 'val_acc_model2': acc_model2},
                   {'val_loss_model3': loss_model3.detach(), 'val_acc_model3': acc_model3},
                   {'val_loss_model4': loss_model4.detach(), 'val_acc_model4': acc_model4},
                   {'val_loss_model5': loss_model5.detach(), 'val_acc_model5': acc_model5},
                   {'val_loss_model6': loss_model6.detach(), 'val_acc_model6': acc_model6},
                   {'val_loss_model7': loss_model7.detach(), 'val_acc_model7': acc_model7},
                   {'val_loss_model8': loss_model8.detach(), 'val_acc_model8': acc_model8},
                   {'val_loss_model9': loss_model9.detach(), 'val_acc_model9': acc_model9},
                   {'val_loss_model10': loss_model10.detach(), 'val_acc_model10': acc_model10}]
    

    def validation_epoch_end(self, outputs):
        batch_losses_model1 = [x[0]['val_loss_model1'] for x in outputs]
        epoch_loss_model1 = torch.stack(batch_losses_model1).mean()   # Combine losses
        batch_accs_model1 = [x[0]['val_acc_model1'] for x in outputs]
        epoch_acc_model1 = torch.stack(batch_accs_model1).mean()      # Combine accuracies

        batch_losses_model2 = [x[1]['val_loss_model2'] for x in outputs]
        epoch_loss_model2 = torch.stack(batch_losses_model2).mean()   # Combine losses
        batch_accs_model2 = [x[1]['val_acc_model2'] for x in outputs]
        epoch_acc_model2 = torch.stack(batch_accs_model2).mean()      # Combine accuracies

        batch_losses_model3 = [x[2]['val_loss_model3'] for x in outputs]
        epoch_loss_model3 = torch.stack(batch_losses_model3).mean()   # Combine losses
        batch_accs_model3 = [x[2]['val_acc_model3'] for x in outputs]
        epoch_acc_model3 = torch.stack(batch_accs_model3).mean()      # Combine accuracies

        batch_losses_model4 = [x[3]['val_loss_model4'] for x in outputs]
        epoch_loss_model4 = torch.stack(batch_losses_model4).mean()   # Combine losses
        batch_accs_model4 = [x[3]['val_acc_model4'] for x in outputs]
        epoch_acc_model4 = torch.stack(batch_accs_model4).mean()      # Combine accuracies

        batch_losses_model5 = [x[4]['val_loss_model5'] for x in outputs]
        epoch_loss_model5 = torch.stack(batch_losses_model5).mean()   # Combine losses
        batch_accs_model5 = [x[4]['val_acc_model5'] for x in outputs]
        epoch_acc_model5 = torch.stack(batch_accs_model5).mean()      # Combine accuracies

        batch_losses_model6 = [x[5]['val_loss_model6'] for x in outputs]
        epoch_loss_model6 = torch.stack(batch_losses_model6).mean()   # Combine losses
        batch_accs_model6 = [x[5]['val_acc_model6'] for x in outputs]
        epoch_acc_model6 = torch.stack(batch_accs_model6).mean()      # Combine accuracies

        batch_losses_model7 = [x[6]['val_loss_model7'] for x in outputs]
        epoch_loss_model7 = torch.stack(batch_losses_model7).mean()   # Combine losses
        batch_accs_model7 = [x[6]['val_acc_model7'] for x in outputs]
        epoch_acc_model7 = torch.stack(batch_accs_model7).mean()      # Combine accuracies

        batch_losses_model8 = [x[7]['val_loss_model8'] for x in outputs]
        epoch_loss_model8 = torch.stack(batch_losses_model8).mean()   # Combine losses
        batch_accs_model8 = [x[7]['val_acc_model8'] for x in outputs]
        epoch_acc_model8 = torch.stack(batch_accs_model8).mean()      # Combine accuracies

        batch_losses_model9 = [x[8]['val_loss_model9'] for x in outputs]
        epoch_loss_model9 = torch.stack(batch_losses_model9).mean()   # Combine losses
        batch_accs_model9 = [x[8]['val_acc_model9'] for x in outputs]
        epoch_acc_model9 = torch.stack(batch_accs_model9).mean()      # Combine accuracies

        batch_losses_model10 = [x[9]['val_loss_model10'] for x in outputs]
        epoch_loss_model10 = torch.stack(batch_losses_model10).mean()   # Combine losses
        batch_accs_model10 = [x[9]['val_acc_model10'] for x in outputs]
        epoch_acc_model10 = torch.stack(batch_accs_model10).mean()      # Combine accuracies


        return [{'val_loss_model1': epoch_loss_model1.item(), 'val_acc_model1': epoch_acc_model1.item()},
                  {'val_loss_model2': epoch_loss_model2.item(), 'val_acc_model2': epoch_acc_model2.item()},
                  {'val_loss_model3': epoch_loss_model3.item(), 'val_acc_model3': epoch_acc_model3.item()},
                  {'val_loss_model4': epoch_loss_model4.item(), 'val_acc_model4': epoch_acc_model4.item()},
                  {'val_loss_model5': epoch_loss_model5.item(), 'val_acc_model5': epoch_acc_model5.item()},
                  {'val_loss_model6': epoch_loss_model6.item(), 'val_acc_model6': epoch_acc_model6.item()},
                  {'val_loss_model7': epoch_loss_model7.item(), 'val_acc_model7': epoch_acc_model7.item()},
                  {'val_loss_model8': epoch_loss_model8.item(), 'val_acc_model8': epoch_acc_model8.item()},
                  {'val_loss_model9': epoch_loss_model9.item(), 'val_acc_model9': epoch_acc_model9.item()},
                  {'val_loss_model10': epoch_loss_model10.item(), 'val_acc_model10': epoch_acc_model10.item()}]

    def epoch_end(self, epoch, results, val, start_time):
        timeSet = [25,50,75,100,150]
        if epoch in timeSet:
            timeTaken = time.time()-start_time
            dataLine = val+','+str(epoch)+','+str(timeTaken)
            addLine('dataList/list.csv',dataLine)
        
        result = results[0]
        print("Epoch [{}], last_lr_model1: {:.5f}, train_loss_model1: {:.4f}, val_loss_model1: {:.4f}, val_acc_model1: {:.4f}".format(
            epoch, result['lrs_model1'][-1], result['train_loss_model1'], result['val_loss_model1'], result['val_acc_model1']))
        print("---------------------------------------------------------------------------------------------------")
        
        result = results[1]
        print("Epoch [{}], last_lr_model2: {:.5f}, train_loss_model2: {:.4f}, val_loss_model2: {:.4f}, val_acc_model2: {:.4f}".format(
            epoch, result['lrs_model2'][-1], result['train_loss_model2'], result['val_loss_model2'], result['val_acc_model2']))
        print("---------------------------------------------------------------------------------------------------")

        result = results[2]
        print("Epoch [{}], last_lr_model3: {:.5f}, train_loss_model3: {:.4f}, val_loss_model3: {:.4f}, val_acc_model3: {:.4f}".format(
            epoch, result['lrs_model3'][-1], result['train_loss_model3'], result['val_loss_model3'], result['val_acc_model3']))
        print("---------------------------------------------------------------------------------------------------")

        result = results[3]
        print("Epoch [{}], last_lr_model4: {:.5f}, train_loss_model4: {:.4f}, val_loss_model4: {:.4f}, val_acc_model4: {:.4f}".format(
            epoch, result['lrs_model4'][-1], result['train_loss_model4'], result['val_loss_model4'], result['val_acc_model4']))
        print("---------------------------------------------------------------------------------------------------")
        
        result = results[4]
        print("Epoch [{}], last_lr_model5: {:.5f}, train_loss_model5: {:.4f}, val_loss_model5: {:.4f}, val_acc_model5: {:.4f}".format(
            epoch, result['lrs_model5'][-1], result['train_loss_model5'], result['val_loss_model5'], result['val_acc_model5']))
        print("---------------------------------------------------------------------------------------------------")
        
        result = results[5]
        print("Epoch [{}], last_lr_model6: {:.5f}, train_loss_model6: {:.4f}, val_loss_model6: {:.4f}, val_acc_model6: {:.4f}".format(
            epoch, result['lrs_model6'][-1], result['train_loss_model6'], result['val_loss_model6'], result['val_acc_model6']))
        print("---------------------------------------------------------------------------------------------------")
        
        result = results[6]
        print("Epoch [{}], last_lr_model7: {:.5f}, train_loss_model7: {:.4f}, val_loss_model7: {:.4f}, val_acc_model7: {:.4f}".format(
            epoch, result['lrs_model7'][-1], result['train_loss_model7'], result['val_loss_model7'], result['val_acc_model7']))
        print("---------------------------------------------------------------------------------------------------")

        result = results[7]
        print("Epoch [{}], last_lr_model8: {:.5f}, train_loss_model8: {:.4f}, val_loss_model8: {:.4f}, val_acc_model8: {:.4f}".format(
            epoch, result['lrs_model8'][-1], result['train_loss_model8'], result['val_loss_model8'], result['val_acc_model8']))
        print("---------------------------------------------------------------------------------------------------")
        
        result = results[8]
        print("Epoch [{}], last_lr_model9: {:.5f}, train_loss_model9: {:.4f}, val_loss_model9: {:.4f}, val_acc_model9: {:.4f}".format(
            epoch, result['lrs_model9'][-1], result['train_loss_model9'], result['val_loss_model9'], result['val_acc_model9']))
        print("---------------------------------------------------------------------------------------------------")
        
        result = results[9]
        print("Epoch [{}], last_lr_model10: {:.5f}, train_loss_model10: {:.4f}, val_loss_model10: {:.4f}, val_acc_model10: {:.4f}".format(
            epoch, result['lrs_model10'][-1], result['train_loss_model10'], result['val_loss_model10'], result['val_acc_model10']))
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
    def __init__(self, bnum_blocks=[2, 2, 2, 2], num_classes=10, num_channel=3, vgg_name='VGG16'):
        super(HybridModel, self).__init__()
        
        block=BasicBlock 
        # resnet-18 Model Definition
        self.in_planes_resnet = 64
        self.conv1_model_1 = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_model_1 = nn.BatchNorm2d(64)
        self.layer1_model_1 = self._make_layer_resnet(block, 64, num_blocks[0], stride=1)
        self.layer2_model_1 = self._make_layer_resnet(block, 128, num_blocks[1], stride=2)
        self.layer3_model_1 = self._make_layer_resnet(block, 256, num_blocks[2], stride=2)
        self.layer4_model_1 = self._make_layer_resnet(block, 512, num_blocks[3], stride=2)
        self.linear_model_1 = nn.Linear(512*block.expansion, num_classes)


        # ResNet-50 Model Definition
        block=Bottleneck
        num_blocks = [3, 4, 6, 3]
        self.in_planes_resnet = 64
        self.conv1_model_2 = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_model_2 = nn.BatchNorm2d(64)
        self.layer1_model_2 = self._make_layer_resnet(block, 64, num_blocks[0], stride=1)
        self.layer2_model_2 = self._make_layer_resnet(block, 128, num_blocks[1], stride=2)
        self.layer3_model_2 = self._make_layer_resnet(block, 256, num_blocks[2], stride=2)
        self.layer4_model_2 = self._make_layer_resnet(block, 512, num_blocks[3], stride=2)
        self.linear_model_2 = nn.Linear(512*block.expansion, num_classes)
        
        # ResNet-101 Model Definition
        block=Bottleneck
        num_blocks = [3, 4, 23, 3]
        self.in_planes_resnet = 64
        self.conv1_model_3 = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_model_3 = nn.BatchNorm2d(64)
        self.layer1_model_3 = self._make_layer_resnet(block, 64, num_blocks[0], stride=1)
        self.layer2_model_3 = self._make_layer_resnet(block, 128, num_blocks[1], stride=2)
        self.layer3_model_3 = self._make_layer_resnet(block, 256, num_blocks[2], stride=2)
        self.layer4_model_3 = self._make_layer_resnet(block, 512, num_blocks[3], stride=2)
        self.linear_model_3 = nn.Linear(512*block.expansion, num_classes)
        
        # ResNet-152 Model Definition
        block=Bottleneck
        num_blocks = [3, 8, 36, 3]
        self.in_planes_resnet = 64
        self.conv1_model_4 = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_model_4 = nn.BatchNorm2d(64)
        self.layer1_model_4 = self._make_layer_resnet(block, 64, num_blocks[0], stride=1)
        self.layer2_model_4 = self._make_layer_resnet(block, 128, num_blocks[1], stride=2)
        self.layer3_model_4 = self._make_layer_resnet(block, 256, num_blocks[2], stride=2)
        self.layer4_model_4 = self._make_layer_resnet(block, 512, num_blocks[3], stride=2)
        self.linear_model_4 = nn.Linear(512*block.expansion, num_classes)
        
        # VGG16 Model Definition
        self.features_model_5 = self._make_layers_vgg(cfg[vgg_name], num_channel)
        self.classifier_model_5 = nn.Linear(512, num_classes)
        
        # VGG19 Model Definition
        vgg_name = 'VGG19'
        self.features_model_6 = self._make_layers_vgg(cfg[vgg_name], num_channel)
        self.classifier_model_6 = nn.Linear(512, num_classes)
        
        # VGG13 Model Definition
        vgg_name = 'VGG13'
        self.features_model_7 = self._make_layers_vgg(cfg[vgg_name], num_channel)
        self.classifier_model_7 = nn.Linear(512, num_classes)
        
        # VGG11 Model Definition
        vgg_name = 'VGG11'
        self.features_model_8 = self._make_layers_vgg(cfg[vgg_name], num_channel)
        self.classifier_model_8 = nn.Linear(512, num_classes)

        # VGG11 Model Definition
        vgg_name = 'VGG11'
        self.features_model_9 = self._make_layers_vgg(cfg[vgg_name], num_channel)
        self.classifier_model_9 = nn.Linear(512, num_classes)

        # VGG13 Model Definition
        vgg_name = 'VGG13'
        self.features_model_10 = self._make_layers_vgg(cfg[vgg_name], num_channel)
        self.classifier_model_10 = nn.Linear(512, num_classes)

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
        out_resnet_18 = F.relu(self.bn1_model_1(self.conv1_model_1(x)))
        out_resnet_18 = self.layer1_model_1(out_resnet_18)
        out_resnet_18 = self.layer2_model_1(out_resnet_18)
        out_resnet_18 = self.layer3_model_1(out_resnet_18)
        out_resnet_18 = self.layer4_model_1(out_resnet_18)
        out_resnet_18 = F.avg_pool2d(out_resnet_18, 4)
        out_resnet_18 = out_resnet_18.view(out_resnet_18.size(0), -1)
        out_resnet_18 = self.linear_model_1(out_resnet_18)
        
        out_resnet_50 = F.relu(self.bn1_model_2(self.conv1_model_2(x)))
        out_resnet_50 = self.layer1_model_2(out_resnet_50)
        out_resnet_50 = self.layer2_model_2(out_resnet_50)
        out_resnet_50 = self.layer3_model_2(out_resnet_50)
        out_resnet_50 = self.layer4_model_2(out_resnet_50)
        out_resnet_50 = F.avg_pool2d(out_resnet_50, 4)
        out_resnet_50 = out_resnet_50.view(out_resnet_50.size(0), -1)
        out_resnet_50 = self.linear_model_2(out_resnet_50)
        
        out_resnet_101 = F.relu(self.bn1_model_3(self.conv1_model_3(x)))
        out_resnet_101 = self.layer1_model_3(out_resnet_101)
        out_resnet_101 = self.layer2_model_3(out_resnet_101)
        out_resnet_101 = self.layer3_model_3(out_resnet_101)
        out_resnet_101 = self.layer4_model_3(out_resnet_101)
        out_resnet_101 = F.avg_pool2d(out_resnet_101, 4)
        out_resnet_101 = out_resnet_101.view(out_resnet_101.size(0), -1)
        out_resnet_101 = self.linear_model_3(out_resnet_101)
        
        out_resnet_152 = F.relu(self.bn1_model_4(self.conv1_model_4(x)))
        out_resnet_152 = self.layer1_model_4(out_resnet_152)
        out_resnet_152 = self.layer2_model_4(out_resnet_152)
        out_resnet_152 = self.layer3_model_4(out_resnet_152)
        out_resnet_152 = self.layer4_model_4(out_resnet_152)
        out_resnet_152 = F.avg_pool2d(out_resnet_152, 4)
        out_resnet_152 = out_resnet_152.view(out_resnet_152.size(0), -1)
        out_resnet_152 = self.linear_model_4(out_resnet_152)
        
        out_vgg_16 = self.features_model_5(x)
        out_vgg_16 = out_vgg_16.view(out_vgg_16.size(0), -1)
        out_vgg_16 = self.classifier_model_5(out_vgg_16)
        
        out_vgg_19 = self.features_model_6(x)
        out_vgg_19 = out_vgg_19.view(out_vgg_19.size(0), -1)
        out_vgg_19 = self.classifier_model_6(out_vgg_19)
        
        out_vgg_13 = self.features_model_7(x)
        out_vgg_13 = out_vgg_13.view(out_vgg_13.size(0), -1)
        out_vgg_13 = self.classifier_model_7(out_vgg_13)
        
        out_vgg_11 = self.features_model_8(x)
        out_vgg_11 = out_vgg_11.view(out_vgg_11.size(0), -1)
        out_vgg_11 = self.classifier_model_8(out_vgg_11)

        out_vgg_11_2 = self.features_model_9(x)
        out_vgg_11_2 = out_vgg_11_2.view(out_vgg_11_2.size(0), -1)
        out_vgg_11_2 = self.classifier_model_9(out_vgg_11_2)

        out_vgg_13_2 = self.features_model_10(x)
        out_vgg_13_2 = out_vgg_13_2.view(out_vgg_13_2.size(0), -1)
        out_vgg_13_2 = self.classifier_model_10(out_vgg_13_2)

        return out_resnet_18, out_resnet_50, out_resnet_101, out_resnet_152, out_vgg_16, out_vgg_19, out_vgg_13, out_vgg_11, out_vgg_11_2, out_vgg_13_2

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

    model1_params = []
    model2_params = []
    model3_params = []
    model4_params = []
    model5_params = []
    model6_params = []
    model7_params = []
    model8_params = []
    model9_params = []
    model10_params = []
    # print(type(model.parameters()))
    # print(model.parameters())

    for name, params in model.named_parameters():
        if '_model_2' in name:
            model2_params.append(params)
        elif '_model_3' in name:
            model3_params.append(params)
        elif '_model_4' in name:
            model4_params.append(params)
        elif '_model_5' in name:
            model5_params.append(params)
        elif '_model_6' in name:
            model6_params.append(params)
        elif '_model_7' in name:
            model7_params.append(params)
        elif '_model_8' in name:
            model8_params.append(params)
        elif '_model_9' in name:
            model9_params.append(params)
        elif '_model_10' in name:
            model10_params.append(params)
        else:
            model1_params.append(params)

    # model_params=[] 
    # for x in model.parameters():
    #     model_params.append(x)
    
        
#         # Set up cutom optimizer with weight decay
    optimizerModel1 = opt_func(model1_params, max_lr, weight_decay=weight_decay)
    optimizerModel2 = opt_func(model2_params, max_lr, weight_decay=weight_decay)
    optimizerModel3 = opt_func(model3_params, max_lr, weight_decay=weight_decay)
    optimizerModel4 = opt_func(model4_params, max_lr, weight_decay=weight_decay)
    optimizerModel5 = opt_func(model5_params, max_lr, weight_decay=weight_decay)
    optimizerModel6 = opt_func(model6_params, max_lr, weight_decay=weight_decay)
    optimizerModel7 = opt_func(model7_params, max_lr, weight_decay=weight_decay)
    optimizerModel8 = opt_func(model8_params, max_lr, weight_decay=weight_decay)
    optimizerModel9 = opt_func(model9_params, max_lr, weight_decay=weight_decay)
    optimizerModel10 = opt_func(model10_params, max_lr, weight_decay=weight_decay)
    

    # Set up one-cycle learning rate scheduler
    schedModel1 = torch.optim.lr_scheduler.OneCycleLR(optimizerModel1, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    schedModel2 = torch.optim.lr_scheduler.OneCycleLR(optimizerModel2, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    schedModel3 = torch.optim.lr_scheduler.OneCycleLR(optimizerModel3, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    schedModel4 = torch.optim.lr_scheduler.OneCycleLR(optimizerModel4, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    schedModel5 = torch.optim.lr_scheduler.OneCycleLR(optimizerModel5, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    schedModel6 = torch.optim.lr_scheduler.OneCycleLR(optimizerModel6, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    schedModel7 = torch.optim.lr_scheduler.OneCycleLR(optimizerModel7, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    schedModel8 = torch.optim.lr_scheduler.OneCycleLR(optimizerModel8, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    schedModel9 = torch.optim.lr_scheduler.OneCycleLR(optimizerModel9, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    schedModel10 = torch.optim.lr_scheduler.OneCycleLR(optimizerModel10, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    


    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses_model1 = []
        train_losses_model2 = []
        train_losses_model3 = []
        train_losses_model4 = []
        train_losses_model5 = []
        train_losses_model6 = []
        train_losses_model7 = []
        train_losses_model8 = []
        train_losses_model9 = []
        train_losses_model10 = []
        
        
        lrs_model1 = []
        lrs_model2 = []
        lrs_model3 = []
        lrs_model4 = []
        lrs_model5 = []
        lrs_model6 = []
        lrs_model7 = []
        lrs_model8 = []
        lrs_model9 = []
        lrs_model10 = []
  
        for batch in train_dl:

            loss_model1, loss_model2, loss_model3, loss_model4, loss_model5, loss_model6, loss_model7, loss_model8, loss_model9, loss_model10  = model.training_step(batch)

            train_losses_model1.append(loss_model1)
            loss_model1.backward()

            train_losses_model2.append(loss_model2)
            loss_model2.backward()

            train_losses_model3.append(loss_model3)
            loss_model3.backward()

            train_losses_model4.append(loss_model4)
            loss_model4.backward()

            train_losses_model5.append(loss_model5)
            loss_model5.backward()

            train_losses_model6.append(loss_model6)
            loss_model6.backward()

            train_losses_model7.append(loss_model7)
            loss_model7.backward()

            train_losses_model8.append(loss_model8)
            loss_model8.backward()
            
            train_losses_model9.append(loss_model9)
            loss_model9.backward()

            train_losses_model10.append(loss_model10)
            loss_model10.backward()
            
            # Gradient clipping
            # if grad_clip: 
            #     nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizerModel1.step()
            optimizerModel1.zero_grad()

            optimizerModel2.step()
            optimizerModel2.zero_grad()

            optimizerModel3.step()
            optimizerModel3.zero_grad()

            optimizerModel4.step()
            optimizerModel4.zero_grad()

            optimizerModel5.step()
            optimizerModel5.zero_grad()

            optimizerModel6.step()
            optimizerModel6.zero_grad()

            optimizerModel7.step()
            optimizerModel7.zero_grad()

            optimizerModel8.step()
            optimizerModel8.zero_grad()

            optimizerModel9.step()
            optimizerModel9.zero_grad()

            optimizerModel10.step()
            optimizerModel10.zero_grad()


            # Record & update learning rate
            lrs_model1.append(get_lr(optimizerModel1))
            schedModel1.step()

            lrs_model2.append(get_lr(optimizerModel2))
            schedModel2.step()

            lrs_model3.append(get_lr(optimizerModel3))
            schedModel3.step()

            lrs_model4.append(get_lr(optimizerModel4))
            schedModel4.step()

            lrs_model5.append(get_lr(optimizerModel5))
            schedModel5.step()

            lrs_model6.append(get_lr(optimizerModel6))
            schedModel6.step()

            lrs_model7.append(get_lr(optimizerModel7))
            schedModel7.step()

            lrs_model8.append(get_lr(optimizerModel8))
            schedModel8.step()

            lrs_model9.append(get_lr(optimizerModel9))
            schedModel9.step()

            lrs_model10.append(get_lr(optimizerModel10))
            schedModel10.step()
        
        # Validation phase
        result = evaluate(model, valid_dl)
        result_model1 = result[0]
        result_model2 = result[1]
        result_model3 = result[2]
        result_model4 = result[3]
        result_model5 = result[4]
        result_model6 = result[5]
        result_model7 = result[6]
        result_model8 = result[7]
        result_model9 = result[8]
        result_model10 = result[9]
        

        result_model1['train_loss_model1'] = torch.stack(train_losses_model1).mean().item()
        result_model1['lrs_model1'] = lrs_model1

        result_model2['train_loss_model2'] = torch.stack(train_losses_model2).mean().item()
        result_model2['lrs_model2'] = lrs_model2

        result_model3['train_loss_model3'] = torch.stack(train_losses_model3).mean().item()
        result_model3['lrs_model3'] = lrs_model3

        result_model4['train_loss_model4'] = torch.stack(train_losses_model4).mean().item()
        result_model4['lrs_model4'] = lrs_model4

        result_model5['train_loss_model5'] = torch.stack(train_losses_model5).mean().item()
        result_model5['lrs_model5'] = lrs_model5

        result_model6['train_loss_model6'] = torch.stack(train_losses_model6).mean().item()
        result_model6['lrs_model6'] = lrs_model6

        result_model7['train_loss_model7'] = torch.stack(train_losses_model7).mean().item()
        result_model7['lrs_model7'] = lrs_model7

        result_model8['train_loss_model8'] = torch.stack(train_losses_model8).mean().item()
        result_model8['lrs_model8'] = lrs_model8

        result_model9['train_loss_model9'] = torch.stack(train_losses_model9).mean().item()
        result_model9['lrs_model9'] = lrs_model9

        result_model10['train_loss_model10'] = torch.stack(train_losses_model10).mean().item()
        result_model10['lrs_model10'] = lrs_model10




        result = [result_model1, result_model2, result_model3, result_model4, result_model5, result_model6, result_model7, result_model8, result_model9, result_model10]
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