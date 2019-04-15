from __future__ import print_function
import sys
import os
sys.path.append(os.path.dirname(__file__))
import numpy as np;
import argparse
import random
from .p2v import P2V;
import torch;
from torch.utils.checkpoint import checkpoint;
import torch.nn as nn

class P2VToy(nn.Module):
    def __init__(self,*args,**kwargs):
        super(P2VToy, self).__init__();
        self.grid_dim = kwargs['grid_dim'];
        self.grid = torch.from_numpy(np.mgrid[-1:1:complex(0,self.grid_dim),-1:1:complex(0,self.grid_dim),-1:1:complex(0,self.grid_dim)].astype(np.float32));
        self.p2v = P2V();
        
    def forward(self,x):
        return checkpoint(self.p2v,x,self.grid);