import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torchvision import datasets, transforms, models
import time

import os
from PIL import Image
from torch.utils.data import Dataset,DataLoader

voc_root = '/home/hyf/datasets/LIP/'

def read_images(root=voc_root, train=True):
    txt_fname = root  + ('train_id.txt' if train else 'val_id.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, ('train_images/' if train else 'val_images/')+i+'.jpg') for i in images]
    label = [os.path.join(root, ('train_segmentations/' if train else 'val_segmentations/')+i+'.png') for i in images]
    return data, label

def rand_crop(data, label, height, width):
    
    #data is PIL.Image object
    #label is PIL.Image object
    
    data, rect = transforms.RandomCrop((height, width))(data)
    label = transforms.FixedCrop(*rect)(label)
    return data, label

classes = ['background','Hat','Hair','Glove','Sunglasses','UpperClothes',
           'Dress','Coat','Socks','Pants','Jumpsuits','Scarf','Skirt',
           'Face','Left-arm','Right-arm','Left-leg','Right-leg',
           'Left-shoe','Right-shoe']

# RGB color for each class
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,128]]

len(classes), len(colormap)

def png2color(im):
    data = np.array(im, dtype='int32')
    return np.array([colormap[i] for cols in data.tolist() for i in cols], dtype='uint8').reshape(data.shape[0],data.shape[1],3) # 根据索引得到color图

def png2label(im):
    L = np.asarray(np.array(im), np.int64)
    return  L.copy()

def img_transforms(im, label):
    #im, label = rand_crop(im, label, *crop_size)
    im_tfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    im = im_tfs(im)
    label = png2label(label)
    label = torch.from_numpy(label)
    return im, label

class LIPSegDataset(Dataset):
    
    # LIP dataset
    
    def __init__(self, train, transforms):
        self.crop_size = [100,100]
        self.transforms = transforms
        data_list, label_list = read_images(train=train)
        self.data_list = data_list
        self.data_list = self._filter(self.data_list)
        self.label_list = label_list
        self.label_list = self._filter(self.label_list)
        _filter
        print('Read ' + str(len(self.data_list)) + ' images')
        
    def _filter(self, images): # 过滤掉图片大小小于 crop 大小的图片
        return [im for im in images if (Image.open(im).size[1] >= self.crop_size[0] and 
                                        Image.open(im).size[0] >= self.crop_size[1])]
        
    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('L')
        img, label = self.transforms(img, label)
        return img, label
    
    def __len__(self):
        return len(self.data_list)

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss
	
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    # Returns accuracy score evaluation result.
      # overall accuracy
      # mean accuracy
      # mean IU
      # fwavacc

    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc