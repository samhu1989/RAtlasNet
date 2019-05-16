import os;
from .task import Task;
from ..ply import *;
from ..Lutils import *;
import torch;
from torch.autograd import Variable as V;
import torch.nn as nn
import math;
from ..vox2th import voxfile2th;
from ..vox2th import th2voxfile;
import numpy as np;

class RealTask(Task):
    def __init__(self):
        super(RealTask,self).__init__();
        self.tskname = os.path.basename(__file__).split('.')[0];
        self.gt = None;
        self.pts = torch.from_numpy(np.random.uniform(-1.0,1.0,size=[1,3,150]).astype(np.float32));
        self.pts = self.pts.cuda();
        self.pts = V(self.pts,requires_grad=True);
        
        
    def run(self,*args,**kwargs):
        self.start();
        self.step();
        return;
        
    def start(self):
        if self.cnt > 0:
            return;
        self.gt = voxfile2th(self.opt['data']);
        self.gt = self.gt.view(1,self.gt.size(0),self.gt.size(1),self.gt.size(2));
        self.gt = self.gt.type(self.pts.type());
        self.gt = V(self.gt,requires_grad=False);
        param = [self.pts];
        param.extend(self.net.parameters())
        self.optim = optim.Adam(param,lr = self.opt['lr']);

    def step(self):
        pts = torch.tanh(self.pts);
        vox = self.net(pts);
        loss = torch.neg(self.gt*torch.log(vox+1e-9)+(1.0-self.gt)*torch.log((1.0-vox)+1e-9));
        loss = torch.mean(loss);
        self.optim.zero_grad();
        loss.backward();
        self.optim.step();
        pts = pts.transpose(2,1).contiguous();
        pts = pts.data.cpu().numpy();
        v = vox.data.cpu();
        if self.cnt%200==0:
            write_ply(self.plydir+os.sep+'pts_%02d.ply'%(self.cnt),points = pd.DataFrame(pts[0,...]));
            print(self.cnt)
            print(v.min(),v.max());
            print(loss.data.cpu().numpy());
            #print(v);
            #v = (v - v.min()) / (v.max() - v.min()) ;
            v[v>0.5] = 1.0;
            v[v<=0.5] = 0.0;
            th2voxfile(v[0,...],self.plydir+os.sep+'v_%02d.vox'%(self.cnt));
        

        
        
        