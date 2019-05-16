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
flag = 0.0;

def clusterMean(dataset):
    return np.sum(np.array(dataset)) / len(dataset)

def ecludDist(x, y):
    return np.sum(np.square(np.array(x) - np.array(y)))

def kMeans(dataset, dist, center, k): 
    global flag 
    all_kinds = [] 
    for _ in range(k): 
        temp = [] 
        all_kinds.append(temp) 
    for i in dataset: 
        temp = [] 
        for j in center: 
            temp.append(dist(i, j)) 
        all_kinds[temp.index(min(temp))].append(i)
    flag += 1;
    center_ = np.array([clusterMean(i) for i in all_kinds]) 
    if (center_ == center).all() or flag > 20:
        return center
    else: 
        center = center_ 
        kMeans(dataset, dist, center, k)

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
        
    def prob(self,pts,N=32):
        with torch.no_grad():
            points = Variable(pts);
            points = points.cuda();
            idx = knn(xyz,self.k_);
            idx = idx.type(torch.long);
            nn = xyz[idx[:,:,:,0],idx[:,:,:,1],:];
            dc = nn.contiguous() - xyz.view(xyz.size(0),xyz.size(1),1,xyz.size(2));
            v = dc.view(-1,dc.size(2),dc.size(3));
            var = torch.bmm(v.transpose(1,2),v) / float(self.k_);
            var = var.contiguous().view(xyz.size(0),-1,9);
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