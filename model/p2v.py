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

#defferentiable voxelization module
class P2V(nn.Module):
    def __init__(self):
        super(P2V, self).__init__();
        
    def forward(self, x):
        
        return x;
