#
import os;
from .task import Task;
from ..ply import *;
from ..Lutils import *;
from ..utils import *;
from ..datasets import *;
import torch;
from torch.autograd import Variable;
from torch.utils.data import DataLoader;
import torch.nn as nn
import math;
import json;
import numpy as np;
from functools import partial;

sys.path.append("./ext/");
import dist_chamfer as ext;
distChamfer =  ext.chamferDist();
knn = ext.knn;

class RealTask(Task):
    def __init__(self):
        super(RealTask,self).__init__();
        self.tskname = os.path.basename(__file__).split('.')[0];
        
    def run(self,*args,**kwargs):
        self.start();
        self.step();
        return;
        
    def start(self):
        if self.cnt > 0:
            return;
        self.SVR = (self.opt['mode']=='SVR');
        self.train_data = ShapeNet(SVR=self.SVR,normal = False,class_choice = None,train=True);
        self.train_load = DataLoader(self.train_data,batch_size=self.opt['batchSize'],shuffle=True, num_workers=int(self.opt['workers']));
        self.valid_data = ShapeNet(SVR=self.SVR,normal = False,class_choice = None,train=False);
        self.valid_load = DataLoader(self.valid_data,batch_size=self.opt['batchSize'],shuffle=False, num_workers=int(self.opt['workers']));
        self.load_pretrain();
        #
        self.train_cd = AverageValueMeter();
        self.train_inv = AverageValueMeter();
        self.valid_cd = AverageValueMeter();
        self.valid_inv = AverageValueMeter();
        self.optim = optim.Adam(self.net.parameters(),lr=self.opt['lr'],weight_decay=self.opt['weight_decay']);
        for group in self.optim.param_groups:
            group.setdefault('initial_lr', group['lr']);
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim,40,eta_min=0,last_epoch=self.opt['last_epoch']);
        #
        self.train_loss_acc0 = 1e-9;
        self.train_loss_accs = 0;
        self.eval();
        
    def sample(self,prob,N):
        def sample_prob(pos,prob):
            g = np.linspace(0.0,1.0,N);
            step = 1.0 / (N-1);
            xn = pos[0] / step;
            yn = pos[1] / step;
            x1 = g[int(xn)];
            x1w = pos[0] - x1;
            x2 = g[int(xn)+1];
            x2w = x2 - pos[0];
            y1 = g[int(yn)];
            y1w = pos[1] - y1;
            y2 = g[int(yn)+1];
            y2w = y2 - pos[1];
            for i in range(prob.shape[0]):
                p1 = prob[i,int(xn),int(yn)];
                p2 = prob[i,int(xn)+1,int(yn)];
                p3 = prob[i,int(xn),int(yn)+1];
                p4 = prob[i,int(xn)+1,int(yn)+1];
                
            return np.array(,dtype=np.float32)
        samples = np.random.uniform(0.0,1.0,size=[2,2*N]);
        sampled = [];
        for i in range(prob.shape[0]):
            grid_lst = [[]]*int(prob.shape[1]);
            func = partial(sample_prob,prob=prob[i,...]);
            sprob = np.apply_along_axis(func,0,samples);
            ind = np.argsort(sprob[0,...]);
            for j in range(N):
                grid_lst[int(sprob[1,ind[j]])].append([samples[0,ind[j]],samples[1,ind[j]]]);
            for j in range(prob.shape[1]):
                grid_lst[j] = np.array(grid_lst[j],dtype=np.float32);
                grid_lst[j] = torch.from_numpy(grid_lst[j]);
                grid_lst[j] = grid_lst[j].cuda();
            sampled.append(grid_lst);
        return sampled;
        
    def grid(self,N):
        g = np.linspace(0.0,1.0,N);
        x,y = np.meshgrid(g,g);
        x = x.reshape(-1,1);
        y = y.reshape(-1,1);
        grid = np.concatenate((x,y),axis=1);
        grid = grid.transpose(1,0)
        return np.repeat(grid.reshape(1,2,-1),self.opt['batchSize'],axis=0);
        
    def prob(self,pts,N=32):
        with torch.no_grad():
            points = Variable(pts);
            points = points.transpose(2,1).contiguous();
            points = points.cuda();
            grid = self.grid(N);
            grid = torch.from_numpy(grid);
            grid = grid.type(points.type());
            out = self.net(points,grid);
            _,dist2 = distChamfer(points.transpose(2,1).contiguous(),out['y']);
            d , _ = knn(points.transpose(2,1),1,rdist=True);
            d = torch.mean(d,dim=1,keepdim=True);
            d = d.view(d.size(0),1);
            prob = torch.exp(torch.neg( dist2 / (d/9) )).view(points.size(0),self.net.grid_num,N,N);
        return prob;
        
    def eval(self):
        self.valid_cd.reset();
        self.valid_inv.reset();
        for item in self.valid_data.cat:
            self.valid_data.perCatValueMeter[item].reset();
        self.net.eval();
        for i, data in enumerate(self.valid_load, 0):
            img, points, cat, a, name = data;
            prob = self.prob(points);
            sampled = self.sample(prob.cpu().numpy(),self.net.pts_num);
            break;

    def load_pretrain(self):
        if self.opt['model']!='':
            partial_restore(self.net,self.opt['model']);
            print("Previous weights loaded");
        
    def step(self):
        if self.cnt == 0:
            return;
        self.eval();

    def createOptim(self):
        self.optim = optim.Adam(self.net.parameters(),lr = self.opt['lr'],weight_decay=self.opt['weight_decay']);