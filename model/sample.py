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
from torch.autograd import Function;

class Sampler(nn.Module):
    def __init__(self):
        self.mu = 0.5;
        self.std = 0.5;
        super(Sampler, self).__init__();
    
    def forward(self,prob,N):
        b = prob.size(0);
        p = prob.size(1);
        s = torch.sum(prob,dim=[2,3]);
        s = s.view(b,p,1);
        z = mu+sigma*torch.randn(b,p,2,40*N);
        qzv = self.qz(z);
        pzv = self.pz(z,prob);
        ratio = pzv / qzv; 
        k,_ = torch.max( ratio,dim=-1 );
        k = k.view(b,p,1);
        #select samples
        z,pzr = self.select(z,pzv/s,torch.rand_like(qzv)*k*qzv,N);
        return z,pzr;
        
    def qz(self,z):
        qzv = float(1/(np.sqrt(2*np.pi*self.std**2))*torch.exp(-0.5*(z-self.mu)**2/self.std**2);
        qzv = torch.cumprod(qzv,dim=-2);
        return qzv[:,:,-1,:].contiguous();
        
    def pz(self,z,prob):
        return;
        
    def select(self,z,pzv,qzv,N):
        return;