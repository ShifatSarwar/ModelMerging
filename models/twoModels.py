import os
import torch
import math
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
        return loss_model1, loss_model2
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions

        loss_model1 = F.cross_entropy(out[0], labels) # Calculate loss
        loss_model2 = F.cross_entropy(out[1], labels) # Calculate loss
        
        acc_model1 = accuracy(out[0], labels)           # Calculate accuracy
        acc_model2 = accuracy(out[1], labels)           # Calculate accuracy
        
        return [{'val_loss_model1': loss_model1.detach(), 'val_acc_model1': acc_model1},
                   {'val_loss_model2': loss_model2.detach(), 'val_acc_model2': acc_model2}]
                   

    def validation_epoch_end(self, outputs):
        
        batch_losses_model1 = [x[0]['val_loss_model1'] for x in outputs]
        epoch_loss_model1 = torch.stack(batch_losses_model1).mean()   # Combine losses
        batch_accs_model1 = [x[0]['val_acc_model1'] for x in outputs]
        epoch_acc_model1 = torch.stack(batch_accs_model1).mean()      # Combine accuracies

        batch_losses_model2 = [x[1]['val_loss_model2'] for x in outputs]
        epoch_loss_model2 = torch.stack(batch_losses_model2).mean()   # Combine losses
        batch_accs_model2 = [x[1]['val_acc_model2'] for x in outputs]
        epoch_acc_model2 = torch.stack(batch_accs_model2).mean()      # Combine accuracies

        
        return [{'val_loss_model1': epoch_loss_model1.item(), 'val_acc_model1': epoch_acc_model1.item()},
                  {'val_loss_model2': epoch_loss_model2.item(), 'val_acc_model2': epoch_acc_model2.item()}]
       
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


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(ImageClassificationBase):
    def __init__(self, block=Bottleneck, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        nblocks = [6,12,24,16]
        model1_num_planes = 2*growth_rate
        self.model1_conv1 = nn.Conv2d(3, model1_num_planes, kernel_size=3, padding=1, bias=False)

        self.model1_dense1 = self._make_dense_layers(block, model1_num_planes, nblocks[0])
        model1_num_planes += nblocks[0]*growth_rate
        model1_out_planes = int(math.floor(model1_num_planes*reduction))
        self.model1_trans1 = Transition(model1_num_planes, model1_out_planes)
        model1_num_planes = model1_out_planes

        self.model1_dense2 = self._make_dense_layers(block, model1_num_planes, nblocks[1])
        model1_num_planes += nblocks[1]*growth_rate
        model1_out_planes = int(math.floor(model1_num_planes*reduction))
        self.model1_trans2 = Transition(model1_num_planes, model1_out_planes)
        model1_num_planes = model1_out_planes

        self.model1_dense3 = self._make_dense_layers(block, model1_num_planes, nblocks[2])
        model1_num_planes += nblocks[2]*growth_rate
        model1_out_planes = int(math.floor(model1_num_planes*reduction))
        self.model1_trans3 = Transition(model1_num_planes, model1_out_planes)
        model1_num_planes = model1_out_planes

        self.model1_dense4 = self._make_dense_layers(block, model1_num_planes, nblocks[3])
        model1_num_planes += nblocks[3]*growth_rate

        self.model1_bn = nn.BatchNorm2d(model1_num_planes)
        self.model1_linear = nn.Linear(model1_num_planes, num_classes)

        nblocks = [6,12,32,32]
        model2_num_planes = 2*growth_rate
        self.model2_conv1 = nn.Conv2d(3, model2_num_planes, kernel_size=3, padding=1, bias=False)

        self.model2_dense1 = self._make_dense_layers(block, model2_num_planes, nblocks[0])
        model2_num_planes += nblocks[0]*growth_rate
        model2_out_planes = int(math.floor(model2_num_planes*reduction))
        self.model2_trans1 = Transition(model2_num_planes, model2_out_planes)
        model2_num_planes = model2_out_planes

        self.model2_dense2 = self._make_dense_layers(block, model2_num_planes, nblocks[1])
        model2_num_planes += nblocks[1]*growth_rate
        model2_out_planes = int(math.floor(model2_num_planes*reduction))
        self.model2_trans2 = Transition(model2_num_planes, model2_out_planes)
        model2_num_planes = model2_out_planes

        self.model2_dense3 = self._make_dense_layers(block, model2_num_planes, nblocks[2])
        model2_num_planes += nblocks[2]*growth_rate
        model2_out_planes = int(math.floor(model2_num_planes*reduction))
        self.model2_trans3 = Transition(model2_num_planes, model2_out_planes)
        model2_num_planes = model2_out_planes

        self.model2_dense4 = self._make_dense_layers(block, model2_num_planes, nblocks[3])
        model2_num_planes += nblocks[3]*growth_rate

        self.model2_bn = nn.BatchNorm2d(model2_num_planes)
        self.model2_linear = nn.Linear(model2_num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out_model1 = self.model1_conv1(x)
        out_model1 = self.model1_trans1(self.model1_dense1(out_model1))
        out_model1 = self.model1_trans2(self.model1_dense2(out_model1))
        out_model1 = self.model1_trans3(self.model1_dense3(out_model1))
        out_model1 = self.model1_dense4(out_model1)
        out_model1 = F.avg_pool2d(F.relu(self.model1_bn(out_model1)), 4)
        out_model1 = out_model1.view(out_model1.size(0), -1)
        out_model1 = self.model1_linear(out_model1)

        out_model2 = self.model2_conv1(x)
        out_model2 = self.model2_trans1(self.model2_dense1(out_model2))
        out_model2 = self.model2_trans2(self.model2_dense2(out_model2))
        out_model2 = self.model2_trans3(self.model2_dense3(out_model2))
        out_model2 = self.model2_dense4(out_model2)
        out_model2 = F.avg_pool2d(F.relu(self.model2_bn(out_model2)), 4)
        out_model2 = out_model2.view(out_model2.size(0), -1)
        out_model2 = self.model2_linear(out_model2)

        return out_model1, out_model2


model = to_device(DenseNet(), device)

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
    
    for name, params in model.named_parameters():
        if 'model1' in name:
            model1_params.append(params)
        else:
            model2_params.append(params)
    
        
    optimizerModel1 = opt_func(model1_params, max_lr, weight_decay=weight_decay)
    optimizerModel2 = opt_func(model2_params, max_lr, weight_decay=weight_decay)

    schedModel1 = torch.optim.lr_scheduler.OneCycleLR(optimizerModel1, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    schedModel2 = torch.optim.lr_scheduler.OneCycleLR(optimizerModel2, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses_model1 = []
        train_losses_model2 = []
        lrs_model1 = []
        lrs_model2 = []
    
        for batch in train_dl:

            # loss_resnet18, loss_resnet50, loss_model1, loss_model2, loss_vgg16, loss_vgg19, loss_vgg13, loss_vgg11  = model.training_step(batch)
            loss_model1, loss_model2 = model.training_step(batch)

            train_losses_model1.append(loss_model1)
            loss_model1.backward()

            train_losses_model2.append(loss_model2)
            loss_model2.backward()

            optimizerModel1.step()
            optimizerModel1.zero_grad()

            optimizerModel2.step()
            optimizerModel2.zero_grad()

            lrs_model1.append(get_lr(optimizerModel1))
            schedModel1.step()

            lrs_model2.append(get_lr(optimizerModel2))
            schedModel2.step()
        
        # Validation phase
        result = evaluate(model, valid_dl)
        result_model1 = result[0]
        result_model2 = result[1]
        
        result_model1['train_loss_model1'] = torch.stack(train_losses_model1).mean().item()
        result_model1['lrs_model1'] = lrs_model1

        result_model2['train_loss_model2'] = torch.stack(train_losses_model2).mean().item()
        result_model2['lrs_model2'] = lrs_model2


        # result = [result_resnet18, result_resnet50, result_model1, result_model2, result_vgg16, result_vgg19, result_vgg13, result_vgg11]
        result = [result_model1, result_model2]
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