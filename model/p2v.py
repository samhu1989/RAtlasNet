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
import numpy as np
import torch.nn.functional as F;
import math;

#defferentiable voxelization module
class P2V(nn.Module):
    def __init__(self):
        super(P2V, self).__init__();
        self.s1 = torch.nn.Parameter(torch.ones(1));
        self.s2 = torch.nn.Parameter(torch.ones(1));
    def forward(self,xyz,grid):
        x = xyz.view(xyz.size(0),xyz.size(1),1,1,1,xyz.size(2));
        g = grid.view(1,grid.size(0),grid.size(1),grid.size(2),grid.size(3),1);
        g = g.type(x.type());
        v = torch.abs(self.s2)*torch.exp(torch.neg(torch.abs(self.s1)*torch.sum((x - g)**2,dim=1)));
        v = torch.softmax(v,dim=4);
        p,_ = torch.topk(v,2,4,True,True);
        v = p[...,0] - p[...,1];
        return v;
