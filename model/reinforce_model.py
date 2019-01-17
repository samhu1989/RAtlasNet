#
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim;
from torch.nn.parameter import Parameter
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F;
import resnet;
import math;
import collections;

def resize(*input):
    y = input[0];
    size = input[1:];
    if isinstance(size[0],torch.Size):
        size = size[0];
    if isinstance(y,torch.FloatTensor) or isinstance(y,torch.cuda.FloatTensor):
        y = y.resize_(size);
    else:
        y = y.resize(*size);
    return y;

#atlasnet
class STN3d(nn.Module):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = self.mp1(x)
        #print(x.size())
        x,_ = torch.max(x, 2)
        #print(x.size())
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True, trans = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.trans = trans
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
        
    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500,odim=3,bn=True):
        self.bottleneck_size = bottleneck_size;
        super(PointGenCon, self).__init__();
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size,self.bottleneck_size,1);
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size,self.bottleneck_size//2,1);
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2,self.bottleneck_size//4,1);
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4,odim,1);
        self.bn = bn;
        self.th = nn.Tanh();
        if self.bn:
            self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size);
            self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2);
            self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4);
        
    def forward(self, x):
        # print(x.size());
        if self.bn:
            x = F.relu(self.bn1(self.conv1(x)));
            x = F.relu(self.bn2(self.conv2(x)));
            x = F.relu(self.bn3(self.conv3(x)));
            x = self.th(self.conv4(x))
        else:
            x = F.relu(self.conv1(x));
            x = F.relu(self.conv2(x));
            x = F.relu(self.conv3(x));
            x = self.th(self.conv4(x))
        return x;
        
class MLP(nn.Module):
    def __init__(self,size=[2,1024,1],act=nn.ReLU()):
        super(MLP,self).__init__();
        d = collections.OrderedDict();
        for i in range(len(size)-1):
            d['linear_%d'%i] = nn.Linear(size[i],size[i+1]);
            if i ==  len(size) - 2:
                if isinstance(act,nn.Module):
                    d['relu_%d'%i] = act;
            else:
                d['relu_%d'%i] = nn.ELU();
        self.net = nn.Sequential(d);
        
    def forward(self,x):
        return self.net(x);
            
class RModel(nn.Module):
    def __init__(self,xen=lambda x:x,yen=lambda y:y,de=MLP([2,1024,1],act=None),val=MLP([2,1024,1],act=None),nmax=10):
        super(RModel,self).__init__();
        self.xen = xen;
        self.yen = yen;
        self.de = de;
        self.val = val;
        self.nmax = nmax;
        self._init_layers();
        
    def forward(self,*input):
        X = input[0];
        xsig = self.xen(X);
        if len(input) > 1:
            Y = input[1];
        else:
            Y = torch.zeros(1,1);
            if X.is_cuda:
                Y = Y.cuda();
        ylst = [];
        vlst = [];
        ylst.append(Y);
        ysig = self.yen(Y);
        sig = torch.cat((xsig,ysig),1).contiguous();
        V = self.val(sig);
        vlst.append(V);
        for i in range(self.nmax):
            ysig = self.yen(ylst[-1]);
            sig = torch.cat((xsig,ysig),1).contiguous();
            Y = ylst[-1] + self.de(sig);
            ysig = self.yen(Y);
            sig = torch.cat((xsig,ysig),1).contiguous();
            V = self.val(sig);
            if i > 0 and (V >= vlst[-1]):
                ylst.append(Y);
                vlst.append(V);
                break;
            else:
                ylst.append(Y);
                vlst.append(V);
        return ylst,vlst;
    
    def _init_layers(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels;
                m.weight.data.normal_(0,math.sqrt(2./n));
            elif isinstance(m,nn.Conv1d):
                m.weight.data.normal_(0.0,0.02);
            elif isinstance(m,nn.BatchNorm1d):
                m.weight.data.normal_(1.0,0.02);
                m.bias.data.fill_(0)
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1);
                m.bias.data.zero_();
            elif isinstance(m,nn.Linear):
                m.weight.data.normal_(0.0,0.5);
                m.bias.data.fill_(0)