import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def convrelu(in_channels, out_channels, kernel, padding, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
    )

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MyFCN_diy_all(nn.Module):
    def __init__(self, ResidualBlock=ResidualBlock, num_classes=30):
        super().__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  1, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 1, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 1, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 1, stride=2)
        #self.fc = nn.Linear(512, 10)
        
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        #out = out.view(out.size(0), -1)
        #out = self.fc(out)
        return out
	
	
def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).cuda()

class MyFCN(nn.Module):
    def __init__(self, n_class):
        self.n_class = n_class
        super().__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_layers = list(self.base_model.children())
        
        self.layer00 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=100, bias=False)
        self.layer0 = nn.Sequential(*self.base_layers[1:4]) # size=(N, 64, x.H/4, x.W/4)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        
        self.layer1 = nn.Sequential(*self.base_layers[4]) # size=(N, 256, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(256, 256, 1, 0)
        
        self.layer2 = self.base_layers[5]  # size=(N, 512, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(512, 512, 1, 0)
        
        self.layer3 = self.base_layers[6]  # size=(N, 1024, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(1024, 1024, 1, 0)
        
        self.layer4 = self.base_layers[7]  # size=(N, 2048, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(2048, 2048, 1, 0)
        
        self.conv0 = convrelu(2048, 4096, 7, 0)
        #self.conv0 = convrelu(2048, 4096, 1, 0)
        self.conv1 = convrelu(4096, 4096, 1, 0)
        
        self.scores1 = nn.Conv2d(4096, n_class, kernel_size=1, stride=1, padding=0, bias=False)
        self.scores2 = nn.Conv2d(1024, n_class, kernel_size=1, stride=1, padding=0, bias=False)
        self.scores3 = nn.Conv2d(512, n_class, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.upsample_8x = nn.ConvTranspose2d(n_class, n_class, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(n_class, n_class, 16) # 使用双线性 kernel
        
        self.upsample_4x = nn.ConvTranspose2d(n_class, n_class, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(n_class, n_class, 4) # 使用双线性 kernel
        
        self.upsample_2x = nn.ConvTranspose2d(n_class, n_class, 4, 2, 1, bias=False)   
        self.upsample_2x.weight.data = bilinear_kernel(n_class, n_class, 4) # 使用双线性 kernel
        
    def forward(self, input):
        # the base net
        layer0 = self.layer00(input)
        layer0 = self.layer0(layer0)
        layer0 = self.layer0_1x1(layer0) 
        
        layer1 = self.layer1(layer0)
        layer1 = self.layer1_1x1(layer1)
        
        layer2 = self.layer2(layer1)
        layer2 = self.layer2_1x1(layer2)
        
        layer3 = self.layer3(layer2)
        layer3 = self.layer3_1x1(layer3)
        
        layer4 = self.layer4(layer3)
        layer4 = self.layer4_1x1(layer4)

        # fully conv
        fc5 = self.conv0(layer4)
        fc5 = F.dropout(fc5,p=0.5)
        fc5 = self.conv1(fc5)
        fc5 = F.dropout(fc5,p=0.5)
        fc5 = self.scores1(fc5)
        fc5 = self.upsample_2x(fc5)
        
        fc6 = self.scores2(layer3)
        fc6 = fc6[:,:,abs(fc6.size()[2]-fc5.size()[2])//2:fc5.size()[2]+abs(fc6.size()[2]-fc5.size()[2])//2,abs(fc6.size()[3]-fc5.size()[3])//2:fc5.size()[3]+abs(fc6.size()[3]-fc5.size()[3])//2]
        fc6 = fc6 + fc5
        fc6 = self.upsample_4x(fc6)

        fc7 = self.scores3(layer2)
        fc7 = fc7[:,:,abs(fc7.size()[2]-fc6.size()[2])//2:fc6.size()[2]+abs(fc7.size()[2]-fc6.size()[2])//2,abs(fc7.size()[3]-fc6.size()[3])//2:fc6.size()[3]+abs(fc7.size()[3]-fc6.size()[3])//2]
        fc7 = fc7 + fc6
        fc7 = self.upsample_8x(fc7)
        
        out = fc7[:,:,abs(fc7.size()[2]-input.size()[2])//2:input.size()[2]+abs(fc7.size()[2]-input.size()[2])//2,abs(fc7.size()[3]-input.size()[3])//2:input.size()[3]+abs(fc7.size()[3]-input.size()[3])//2]
        return out