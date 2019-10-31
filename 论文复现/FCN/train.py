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

import torch.optim as optim
from datetime import datetime
from torch.optim import lr_scheduler

from fcn import *
from tool import *

gamma      = 0.5
epochs     = 50
lr         = 1e-4
momentum   = 0
w_decay    = 1e-5
step_size  = 50
use_gpu = True

lip_train = LIPSegDataset(True, img_transforms)
lip_val = LIPSegDataset(False, img_transforms)

train_data = DataLoader(lip_train, 1, shuffle=True, num_workers=8)
val_data = DataLoader(lip_val, 1,  num_workers=8)

net = MyFCN(20)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net,device_ids=[0])
net.to(device)

criterion = cross_entropy2d
optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

def train():
    for epoch in range(epochs):
        scheduler.step()

        ts = time.time()
        for iter, batch in enumerate(train_data):
            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(batch[0].cuda())
                labels = Variable(batch[1].cuda())
            else:
                inputs, labels = Variable(batch[0]), Variable(batch[1])

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.data))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        #torch.save(net.state_dictdict(), voc_root)
        filepath = os.path.join(voc_root, 'checkpoint_model_epoch_{}.pth'.format(epoch)) #最终参数模型
        torch.save(net.state_dictdict(), filepath)
        #val(epoch)


def val(epoch):
    fcn_model.eval()
    total_ious = []
    pixel_accs = []
    for iter, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch[0].cuda())
        else:
            inputs = Variable(batch[1])

        output = fcn_model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

        target = batch['l'].cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    IU_scores[epoch] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch] = pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)
	
if __name__ ==  "__main__":
	train()