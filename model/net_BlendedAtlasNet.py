import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.parameter import Parameter
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import resnet
import math
from torch.utils.checkpoint import checkpoint;
from functools import partial;
#NOTE:checkpoint is widely used to reduce GPU memory consumption  
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
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size,self.bottleneck_size,1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size,self.bottleneck_size//2,1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2,self.bottleneck_size//4,1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4,odim,1)
        self.bn = bn
        self.th = nn.Tanh()
        if self.bn:
            self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
            self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
            self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)
        
    def forward(self, x):
        def func(i,bn,x):
            if bn:
                if i == 0:
                    x = F.relu(self.bn1(self.conv1(x)))
                elif i == 1:
                    x = F.relu(self.bn2(self.conv2(x)))
                else:
                    x = F.relu(self.bn3(self.conv3(x)))
                    x = self.th(self.conv4(x))
            else:
                if i == 0:
                    x = F.relu(self.conv1(x))
                elif i == 1:
                    x = F.relu(self.conv2(x))
                else:
                    x = F.relu(self.conv3(x))
                    x = self.th(self.conv4(x))
            return x;
        x = checkpoint(partial(func,0,self.bn),x);
        x = checkpoint(partial(func,1,self.bn),x);
        x = checkpoint(partial(func,2,self.bn),x);
        return x
        
class WGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500,odim=10,bn=True):
        self.bottleneck_size = bottleneck_size
        super(WGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size,self.bottleneck_size,1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size,self.bottleneck_size//2,1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2,self.bottleneck_size//4,1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4,odim,1)
        self.bn = bn
        self.act = nn.Softmax(dim=1)
        if self.bn:
            self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
            self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
            self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)
        
    def forward(self, x):
        def func(i,bn,x):
            if bn:
                if i == 0:
                    x = F.relu(self.bn1(self.conv1(x)))
                elif i == 1:
                    x = F.relu(self.bn2(self.conv2(x)))
                else:
                    x = F.relu(self.bn3(self.conv3(x)))
                    x = self.act(self.conv4(x))
            else:
                if i == 0:
                    x = F.relu(self.conv1(x))
                elif i == 1:
                    x = F.relu(self.conv2(x))
                else:
                    x = F.relu(self.conv3(x))
                    x = self.act(self.conv4(x))
            return x;
        x = checkpoint(partial(func,0,self.bn),x);
        x = checkpoint(partial(func,1,self.bn),x);
        x = checkpoint(partial(func,2,self.bn),x);
        return x;
    
class OptEncoder(nn.Module):
    def __init__(self,bottleneck_size=1024):
        self.bottleneck_size = bottleneck_size
        super(OptEncoder,self).__init__()
        self.f = Parameter(torch.zeros(1,self.bottleneck_size))
        self.f.data.normal_(0,math.sqrt(2./float(self.bottleneck_size)))
        
    def forward(self,x):
        return self.f;
        
class BlendedAtlasNet(nn.Module):
    def __init__(self,*args,**kwargs):
        super(BlendedAtlasNet, self).__init__()
        self.pts_num = kwargs['pts_num']
        self.bottleneck_size = 1024
        self.grid_num = kwargs['grid_num']
        self.grid_dim = kwargs['grid_dim']
        self.mode = kwargs['mode']
        self.bn = True
        if self.mode == 'SVR':
            self.encoder = resnet.resnet18(pretrained=False,num_classes=1024)
        elif self.mode == 'AE':
            self.encoder = nn.Sequential(
                PointNetfeat(self.pts_num, global_feat=True, trans = False),
                nn.Linear(1024, self.bottleneck_size),
                nn.BatchNorm1d(self.bottleneck_size),
                nn.ReLU()
                )
        elif self.mode == 'OPT':
            self.encoder = OptEncoder(self.bottleneck_size)
            self.bn = False
        else:
            assert False,'unkown mode of BlendedAtlasNet'
        self.wdecoder = WGenCon(bottleneck_size=self.grid_dim+self.bottleneck_size,odim=self.grid_num,bn=self.bn);
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size=self.grid_dim+self.bottleneck_size,bn=self.bn) for i in range(0,self.grid_num)]);
        self.inv_decoder = PointGenCon(bottleneck_size=3+self.bottleneck_size,bn=self.bn)
        self._init_layers()

    def forward(self,*input):
        x = input[0]
        if x.dim() == 4:
            x = x[:,:3,:,:].contiguous()
        grid = None;
        f = None;
        if self.mode != 'OPT':
            def defunc(x):
                return self.encoder(x);
            f = checkpoint(defunc,x);
        else:
            f = self.encoder(x);
        if len(input) > 1:
            grid = input[1];
        else:
            grid = self.rand_grid(f);
        expf = f.unsqueeze(2).expand(f.size(0),f.size(1),grid.size(2)).contiguous()
        w = torch.cat((grid,expf),1).contiguous()
        w = self.wdecoder(w)
        w = w.view(w.size(0),w.size(1),1,w.size(2))
        yo = None;
        def blend(i,*input):
            decoder = self.decoder[i];
            grid = input[0];
            expf = input[1];
            w = input[2];
            y = torch.cat((grid,expf),1).contiguous();
            y = decoder(y);
            y = y*w;
            return y;
        for i in range(0,self.grid_num):
            y = partial(blend,i)(grid,expf,w[:,i,:,:]);
            if yo is None:
                yo = y;
            else:
                yo += y;
        invo = torch.cat((yo,expf),1).contiguous()
        invo = self.inv_decoder(invo)
        #transpose from (bn,3,#points) to (bn,3,#points)
        yo = yo.transpose(2,1).contiguous()
        invo = invo.transpose(2,1).contiguous()
        grid = grid.transpose(2,1).contiguous()
        out = {}
        out['y'] = yo
        out['inv_x'] = invo
        out['grid_x'] = grid
        out['blend_w'] = w
        return out
    
    def rand_grid(self,x):
        rand_grid = torch.FloatTensor(x.size(0),self.grid_dim,self.pts_num)
        if self.grid_dim == 3:
            rand_grid.normal_(0.0,1.0)
            rand_grid += 1e-9
            rand_grid = rand_grid / torch.norm(rand_grid,p=2.0,dim=1,keepdim=True)
        else:
            rand_grid.uniform_(0.0,1.0)
        if isinstance(x,Variable):
            rand_grid = Variable(rand_grid)
        if x.is_cuda:
            rand_grid = rand_grid.cuda()
        return rand_grid
    
    def _init_layers(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,nn.Conv1d):
                m.weight.data.normal_(0.0,0.02)
            elif isinstance(m,nn.BatchNorm1d):
                m.weight.data.normal_(1.0,0.02)
                m.bias.data.fill_(0)
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()