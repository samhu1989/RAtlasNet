#
from __future__ import print_function
from __future__ import division
import argparse;
import os;
import random;
import numpy as np;
import torch;
import torch.nn as nn;
import torch.nn.parallel;
import torch.backends.cudnn as cudnn;
import torch.optim as optim;
import torch.utils.data;
import torchvision.datasets as dset;
import torchvision.transforms as transforms;
import torchvision.utils as vutils;
from torch.autograd import Variable;
import sys;
sys.path.append('./model/')
from dataset import *;
from invmodel import *;
from reinforce_model import *;
from utils import *;
from Lutils import *;
from ply import *;
from cmap import colorcoord;
import torch.nn.functional as F
import json
import time, datetime
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;
from matplotlib import cm;

def func(x):
    if np.abs(x) > 1.0:
        return 0.5*x**2 - 0.5;
    else:
        return np.sin(10*np.pi*x)/(10*np.pi*x);
        
x = np.linspace(-2.0,2.0,1000,endpoint=True).astype(np.float32);
x = x.reshape((-1,1));
_x = np.linspace(-2.0,2.0,200,endpoint=True)+np.random.normal(0.0,scale=0.001,size=200);
#
_x = _x.astype(np.float32);
_x = _x.reshape((-1,1));
#
y = np.apply_along_axis(func,1,x);
_y = np.apply_along_axis(func,1,_x);
_y += np.random.normal(0.0,scale=0.015,size=200).reshape((-1,1));
fig = plt.figure(figsize=(20.48,9.6));
plt.subplot(241);
plt.plot(x,y,'k-',_x,_y,'rx');
plt.title('GT and training set')
nepoch = 2000;
#MLP
print('MLP');
mlp = MLP([1,4096,1],act=None);
mlp.cuda();
optimizer = optim.Adam(mlp.parameters(),lr = 0.001);
for epoch in range(nepoch):
    mlp.train();
    print('epoch:',epoch);
    arr = np.arange(_x.shape[0]);
    np.random.shuffle(arr);
    for j in range(_x.shape[0]):
        optimizer.zero_grad();
        xi = np.zeros([1,1],dtype=np.float32);
        xi[0,0] = _x[arr[j],0];
        xi = Variable(torch.from_numpy(xi))
        xi = xi.cuda();
        yout = mlp(xi);
        ygt = np.zeros([1,1],dtype=np.float32);
        ygt[0,0] = _y[arr[j],0];
        ygt = Variable(torch.from_numpy(ygt));
        ygt = ygt.cuda();
        loss = torch.mean((yout - ygt)**2);
        loss.backward();
        optimizer.step();

yo = y.copy();
for i in range(x.shape[0]):
    mlp.eval();
    xi = np.zeros([1,1],dtype=np.float32);
    xi[0,0] = x[i,0];
    xi = Variable(torch.from_numpy(xi))
    xi = xi.cuda();
    yout = mlp(xi);
    yo[i,0] = yout.data[0,0];

plt.subplot(242);
plt.plot(x,y,'k-',x,yo,'b-');
plt.title('GT and trained MLP');

#reinforce_model
print('Reinforced Model');
rmodel = RModel();
rmodel.cuda();

optv = optim.Adam(rmodel.val.parameters(),lr = 0.001);
for epoch in range(nepoch):
    rmodel.train();
    print('epoch:',epoch);
    arr = np.arange(_x.shape[0]);
    np.random.shuffle(arr);
    mval = np.zeros([200],dtype=np.float32);
    for j in range(_x.shape[0]):
        optv.zero_grad();
        xi = np.zeros([128,2],dtype=np.float32);
        xi[:,0] = _x[arr[j],0];
        xi[:,1] = _y[arr[j],0];
        err = np.random.normal(0.0,scale=1.0,size=[128]).astype(np.float32);
        xi[:,1] += err;
        err = err**2;
        xi = Variable(torch.from_numpy(xi))
        xi = xi.cuda();
        err = Variable(torch.from_numpy(err))
        err = err.cuda();
        v = rmodel.val(xi);
        vloss = torch.mean((v - err)**2);
        mval[j] = vloss.data.cpu().numpy();
        vloss.backward();
        optv.step();
    print('mval:',np.mean(mval));
        
sX = np.arange(-2.0, 2.0, 0.005);
sY = np.arange(-0.25,1.7, 0.005);
sX,sY = np.meshgrid(sX,sY);
sZ = sX.copy();
for i in range(sX.shape[0]):
    rmodel.eval();
    xi = np.zeros([sX.shape[1],2],dtype=np.float32);
    xi[:,0] = sX[i,:];
    xi[:,1] = sY[i,:];
    xi = Variable(torch.from_numpy(xi))
    xi = xi.cuda();
    yout = rmodel.val(xi);
    sZ[i,:] = yout.cpu().data[:,0];
ax = fig.add_subplot(243, projection='3d');
surf = ax.plot_surface(sX,sY,sZ,cmap=cm.coolwarm,linewidth=0,antialiased=False);
    
opty = optim.Adam(rmodel.de.parameters(),lr = 0.001);
for epoch in range(nepoch):
    rmodel.train();
    print('epoch:',epoch);
    arr = np.arange(_x.shape[0]);
    np.random.shuffle(arr);
    for j in range(_x.shape[0]):
        opty.zero_grad();
        xi = np.zeros([1,1],dtype=np.float32);
        xi[0,0] = _x[arr[j],0];
        xi = Variable(torch.from_numpy(xi))
        xi = xi.cuda();
        ylst,vlst = rmodel(xi);
        ygt = np.zeros([1,1],dtype=np.float32);
        ygt[0,0] = _y[arr[j],0];
        ygt = Variable(torch.from_numpy(ygt));
        ygt = ygt.cuda();
        yloss = 0.0;
        for t in range(len(ylst)):
            v = (ylst[t] - ygt)**2;
            vgt = v.clone();
            if t < len(ylst) - 1:
                yloss = v + yloss*0.1;
        yloss = torch.mean(yloss);
        yloss.backward();
        opty.step();
        
yo = y.copy();
vec = [];
for i in range(x.shape[0]):
    rmodel.eval();
    xi = np.zeros([1,1],dtype=np.float32);
    xi[0,0] = x[i,0];
    xi = Variable(torch.from_numpy(xi))
    xi = xi.cuda();
    yout,_ = rmodel(xi);
    yo[i,0] = yout[-2].data[0,0];
    vec.append(len(yout)-1);

plt.subplot(244);
plt.plot(x,y,'k-',x,yo,'b-');
plt.title('GT and trained Reinforced Model');

plt.subplot(245);
plt.title('GT and #Step of Reinforced Model');
plt.hist(vec, bins=11, color="#13eac9",range=(0,11))

sX = np.arange(-2.0, 2.0, 0.005);
sY = np.arange(-0.25,1.7, 0.005);
sX,sY = np.meshgrid(sX,sY);
sZ = sX.copy();
for i in range(sX.shape[0]):
    for j in range(sX.shape[1]):
        sZ[i,j] = (sY[i,j] - func(sX[i,j]))**2;
ax = fig.add_subplot(246, projection='3d');
surf = ax.plot_surface(sX,sY,sZ, cmap=cm.coolwarm,linewidth=0, antialiased=False);

plt.savefig('./view.png');
