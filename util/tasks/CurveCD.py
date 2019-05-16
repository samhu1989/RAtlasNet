from __future__ import print_function
import sys
import os
sys.path.append(os.path.dirname(__file__))
import argparse
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
import numpy as np;
import matplotlib.pyplot as plt;
import pdb;
import torch.nn.functional as F;
import resnet;
import math;
sys.path.append("./ext/");
import dist_chamfer as ext;
cd =  ext.chamferDist();
knn = ext.knn;

class CurveCDLoss(nn.Module):
    def __init__(self,k=8):
        super(CurveCDLoss,self).__init__();
        self.k_ = k;
        
    def forward(self,xyz1,xyz2,w=0.1):
        c1 = self.curve(xyz1);
        s1 = torch.cat([xyz1,w*c1],dim=2);
        c2 = self.curve(xyz2);
        s2 = torch.cat([xyz2,w*c2],dim=2);
        d1,d2 = cd(s1.contiguous(),s2.contiguous());
        loss = (torch.mean(d1)) + (torch.mean(d2));
        return loss;
        
    def curve(self,xyz):
        idx = knn(xyz,self.k_);
        idx = idx.type(torch.long);
        nn = xyz[idx[:,:,:,0],idx[:,:,:,1],:];
        dc = nn.contiguous() - xyz.view(xyz.size(0),xyz.size(1),1,xyz.size(2));
        v = dc.view(-1,dc.size(2),dc.size(3));
        var = torch.bmm(v.transpose(1,2),v) / float(self.k_);
        var = var.contiguous().view(xyz.size(0),-1,9);
        return var;
        

