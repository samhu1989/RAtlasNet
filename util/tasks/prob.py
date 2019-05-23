#
import os;
from .task import Task;
from ..ply import *;
from ..Lutils import *;
from ..utils import *;
from ..datasets import *;
from ..sample import reject_sample;
import torch;
from torch.autograd import Variable;
from torch.utils.data import DataLoader;
import torch.nn as nn
import math;
import json;
import numpy as np;
from functools import partial;
import matplotlib;
matplotlib.use('Agg');
import matplotlib.pyplot as plt;

sys.path.append("./ext/");
import dist_chamfer as ext;
distChamfer =  ext.chamferDist();
knn = ext.knn;
from PIL import Image;
from ..cmap import *;

def view_color(y,c=None):
    if c is None:
        c = colorcoord(y);
    return pd.concat([pd.DataFrame(y),pd.DataFrame(c)],axis=1,ignore_index=True);

def view_ae(dirname,net,pts,index,cat,opt):
    grid = None;
    fidx = None;
    with torch.no_grad():
        points = Variable(pts);
        points = points.transpose(2,1).contiguous();
        points = points.cuda();
        if opt['grid_dim'] == 3:
            grid,Li,Lw,fidx = sphere_grid(points.size()[0],opt['pts_num'],'cot');
        elif opt['grid_dim'] == 2:
            grid,Li,Lw,fidx = patch_grid(points.size()[0],opt['pts_num'],opt['grid_num']);
        grid = Variable(grid);
        grid = grid.cuda();
        out  = net(points,grid);
    y = out['y'];
    ply_path = dirname+os.sep+'ply';
    if not os.path.exists(ply_path):
        os.mkdir(ply_path);
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    face = np.zeros(shape=[fidx.shape[0]],dtype=T);
    for i in range(fidx.shape[0]):
        face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
    y = y.cpu().data.numpy();
    gt = points.transpose(2,1).contiguous();
    gt = gt.cpu().data.numpy();
    c = colorindex(y[0,...],opt['grid_num']);
    for i in range(y.shape[0]):
        write_ply(ply_path+os.sep+'%02d_%02d_%s_gt.ply'%(index,i,cat[0]),points = pd.DataFrame(gt[i,...]));
        write_ply(ply_path+os.sep+'%02d_%02d_%s.ply'%(index,i,cat[0]),points = view_color(y[i,...],c),faces=pd.DataFrame(face),color=True);


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
            if cat[0] != 'chair':
                continue;
            prob = self.prob(points);
            view_ae(self.tskdir,self.net,points,i,cat,self.opt);
            probnp = prob.cpu().numpy();
            fig,axes = plt.subplots(5,5,figsize=(20.48,20.48));
            for j in range(prob.size(1)):
                p = probnp[0,j,...]; 
                img = Image.fromarray((255*p).astype('uint8'));
                x = reject_sample(p,self.opt['pts_num']//self.opt['grid_num']);
                x = 31*x;
                ax = axes[j//5,j%5];
                ax.imshow(img);
                ax.plot(x[0,...],x[1,...],'ro');
            plt.savefig(self.tskdir+os.sep+'ply'+os.sep+'p_%d'%i+'.png');
            plt.close(fig);
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