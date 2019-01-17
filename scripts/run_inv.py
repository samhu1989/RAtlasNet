from __future__ import print_function
from __future__ import division
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable;
import sys
sys.path.append('./model/')
from dataset import *
from invmodel import *
from utils import *
from Lutils import *
from ply import *
from cmap import colorcoord;
import torch.nn.functional as F
import sys
import os
import json
import time, datetime

sys.path.append("./ext/");
import dist_chamfer as ext;
distChamfer =  ext.chamferDist();

def eval_ae(net,cd_meter,inv_meter,pts):
    with torch.no_grad():
        points = Variable(pts);
        points = points.transpose(2,1).contiguous()
        points = points.cuda()
        pts_res,inv_err  = net(points)
        dist1, dist2 = distChamfer(points.transpose(2,1).contiguous(),pts_res);
        cd = (torch.mean(dist1)) + (torch.mean(dist2))
        cd_meter.update(cd.data);
        inv_meter.update(inv_err.data);
    return cd.data,inv_err.data;

def train_ae(net,optimizer,cd_meter,inv_meter,pts,opt):
    optimizer.zero_grad();
    points = Variable(pts);
    points = points.transpose(2,1).contiguous();
    points = points.cuda();
    pts_res,inv_err = network(points);
    inv_meter.update(inv_err.data[0]);
    dist1, dist2 = distChamfer(points.transpose(2,1).contiguous(),pts_res);
    cd = (torch.mean(dist1)) + (torch.mean(dist2));
    cd_meter.update(cd.data);
    loss = cd + opt.w*inv_err;
    #loss = inv_err;
    loss.backward();
    optimizer.step();
    return loss,cd,inv_err;
    
def eval_svr(net,cd_meter,inv_meter,pts,img):
    with torch.no_grad():
        img = Variable(img);
        img = img.cuda();
        points = Variable(pts);
        points = points.cuda();
        pts_res,inv_err = net(img);
        dist1, dist2 = distChamfer(points,pts_res);
        cd = (torch.mean(dist1)) + (torch.mean(dist2));
        cd_meter.update(cd.data);
        inv_meter.update(inv_err.data);
    return cd.data,inv_err.data;
    
def train_svr(net,optimizer,cd_meter,inv_meter,pts,img,opt):
    optimizer.zero_grad();
    img = Variable(img);
    img = img.cuda();
    points = Variable(pts);
    points = points.cuda();
    pts_res,inv_err = network(img);
    inv_meter.update(inv_err.data);
    dist1, dist2 = distChamfer(points,pts_res);
    cd = (torch.mean(dist1)) + (torch.mean(dist2));
    cd_meter.update(cd.data);
    loss = cd + opt.w*inv_err;
    loss.backward();
    optimizer.step();
    return loss,cd,inv_err;
    
def write_log(logname,val_cd,val_inv,dataset_test,train_cd=None,train_inv=None,epoch=None):
    log_dict = {};
    log_dict['val_cd'] = np.asscalar(val_cd.avg.cpu().numpy());
    log_dict['val_inv'] = np.asscalar(val_inv.avg.cpu().numpy());
    for item in dataset_test.cat:
        print(item,np.asscalar(dataset_test.perCatValueMeter[item].avg.cpu().numpy()))
        log_dict.update({item:np.asscalar(dataset_test.perCatValueMeter[item].avg.cpu().numpy())})
    if train_cd is not None:
        log_dict['train_cd'] = np.asscalar(train_cd.avg.cpu().numpy());
    if train_inv is not None:
        log_dict['train_inv'] = np.asscalar(train_inv.avg.cpu().numpy());
    if epoch is not None:
        log_dict['epoch'] = epoch;
    with open(logname, 'a') as f: #open and append
        f.write('json_stats: '+json.dumps(log_dict)+'\n');
    return;
        
bestnum = 10;
best_cd = np.zeros(bestnum);
best_all = np.zeros(bestnum);
        
def save_model(logname,dirname,net,opt,vcd,vall):
    global best_cd;
    global best_all;
    if opt.part == 'full':
        sdict = net.state_dict();
    elif opt.part == 'en':
        sdict = net.encoder.state_dict();
    elif opt.part == 'de':
        sdict = net.all_decoder.state_dict();
    cdname = dirname+os.sep+opt.mode+'_'+opt.part+'_cd';
    allname = dirname+os.sep+opt.mode+'_'+opt.part+'_all';
    if vcd < best_cd[-1]:
        best_cd[-1] = vcd;
        best_cd = np.sort(best_cd);
        bidx = np.searchsorted(best_cd,vcd);
        for idx in range(bestnum-2,bidx-1,-1):
            if os.path.exists(cdname+'_%d'%idx+'.pth'):
                if os.path.exists(cdname+'_%d'%(idx+1)+'.pth'):
                    os.remove(cdname+'_%d'%(idx+1)+'.pth');
                print('rename '+cdname+'_%d'%(idx)+'.pth'+' '+cdname+'_%d'%(idx+1)+'.pth');
                os.rename(cdname+'_%d'%(idx)+'.pth',cdname+'_%d'%(idx+1)+'.pth');
        print('saving model');
        torch.save(sdict,cdname+'_%d'%(bidx)+'.pth');
        with open(logname,'a') as f:
            f.write('best_cd:'+np.array2string(best_cd,precision=6,separator=',')+'\n');
    if vall < best_all[-1]:
        best_all[-1] = vall;
        best_all = np.sort(best_all);
        bidx = np.searchsorted(best_all,vall);
        for idx in range(bestnum-2,bidx-1,-1):
            if os.path.exists(allname+'_%d'%idx+'.pth'):
                if os.path.exists(allname+'_%d'%(idx+1)+'.pth'):
                    os.remove(allname+'_%d'%(idx+1)+'.pth');
                print('rename '+allname+'_%d'%(idx)+'.pth'+' '+allname+'_%d'%(idx+1)+'.pth');
                os.rename(allname+'_%d'%(idx)+'.pth',allname+'_%d'%(idx+1)+'.pth');
        print('saving model');
        torch.save(sdict,allname+'_%d'%(bidx)+'.pth');
        with open(logname,'a') as f:
            f.write('best_all:'+np.array2string(best_all,precision=6,separator=',')+'\n');

def load_model(path,net):
    bname = os.path.basename(path);
    if '_full_' in bname:
        net.load_state_dict(torch.load(path));
    elif '_en_' in bname:
        net.encoder.load_state_dict(torch.load(path));
    elif '_de_' in bname:
        net.all_decoder.load_state_dict(torch.load(path));
    else:
        net.load_state_dict(torch.load(path));
        
def view_color(y,c=None):
    if c is None:
        c = colorcoord(y);
    return pd.concat([pd.DataFrame(y),pd.DataFrame(c)],axis=1,ignore_index=True);
        
def view_ae(dirname,net,pts,index,cat,opt):
    points = Variable(pts, volatile=True);
    points = points.transpose(2,1).contiguous();
    points = points.cuda();
    grid = None;
    fidx = None;
    if opt.grid_dim == 3:
        grid,Li,Lw,fidx = sphere_grid(points.size()[0],opt.pts_num,'cot');
    elif opt.grid_dim == 2:
        grid,Li,Lw,fidx = patch_grid(points.size()[0],opt.pts_num,opt.grid_num);
    grid = Variable(grid,volatile=True);
    grid = grid.cuda();
    y,inv_err  = net(points,grid);
    y_inv = net.inv_y;
    ply_path = dirname+os.sep+'ply';
    if not os.path.exists(ply_path):
        os.mkdir(ply_path);
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    face = np.zeros(shape=[fidx.shape[0]],dtype=T);
    for i in range(fidx.shape[0]):
        face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
    y = y.cpu().data.numpy();
    inv_y = net.inv_y.cpu().data.numpy();
    grid = grid.transpose(2,1).contiguous().cpu().data.numpy();
    c = colorcoord(grid[0,...])
    write_ply(ply_path+os.sep+'%02d_%s_grid.ply'%(index,cat[0]),points = view_color(grid[0,...],c),faces=pd.DataFrame(face),color=True);
    for i in range(y.shape[0]):
        write_ply(ply_path+os.sep+'%02d_%02d_%s.ply'%(index,i,cat[0]),points = view_color(y[i,...],c),faces=pd.DataFrame(face),color=True);
        write_ply(ply_path+os.sep+'%02d_%02d_%s_inv.ply'%(index,i,cat[0]),points = view_color(inv_y[i,...],c),faces=pd.DataFrame(face),color=True);
        
def view_svr(dirname,net,img,index,cat,opt):
    img = Variable(img,volatile=True);
    img = img.cuda();
    grid = None;
    fidx = None;
    if opt.grid_dim == 3:
        grid,Li,Lw,fidx = sphere_grid(points.size()[0],opt.pts_num,'cot');
    elif opt.grid_dim == 2:
        grid,Li,Lw,fidx = patch_grid(points.size()[0],opt.pts_num,opt.grid_num);
    grid = Variable(grid,volatile=True);
    grid = grid.cuda();
    y,inv_err  = net(img,grid);
    ply_path = dirname+os.sep+'ply';
    if not os.path.exists(ply_path):
        os.mkdir(ply_path);
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    face = np.zeros(shape=[fidx.shape[0]],dtype=T);
    for i in range(fidx.shape[0]):
        face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
    y = y.cpu().data.numpy();
    inv_y = net.inv_y.cpu().data.numpy();
    grid = grid.transpose(2,1).contiguous().cpu().data.numpy();
    c = colorcoord(grid[0,...])
    write_ply(ply_path+os.sep+'%02d_%s_grid.ply'%(index,cat[0]),points = view_color(grid[0,...],c),faces=pd.DataFrame(face),color=True);
    for i in range(y.shape[0]):
        write_ply(ply_path+os.sep+'%02d_%02d_%s.ply'%(index,i,cat[0]),points = view_color(y[i,...],c),faces=pd.DataFrame(face),color=True);
        write_ply(ply_path+os.sep+'%02d_%02d_%s_inv.ply'%(index,i,cat[0]),points = view_color(inv_y[i,...],c),faces=pd.DataFrame(face),color=True);
    
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize','-bs',type=int,default=32,help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--nepoch', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--model','-m',type=str,default = '',help='model path');
parser.add_argument('--part','-p',type=str,default = 'full',help='save part');
parser.add_argument('--train_part','-tp',type=str,default = 'full',help='train part');
parser.add_argument('--mode','-md', type=str, default = 'AE',  help='mode');
parser.add_argument('--pts_num', type=int, default = 2500,  help='number of points')
parser.add_argument('--grid_num', type=int, default = 1,  help='number of grid')
parser.add_argument('--grid_dim', type=int, default = 3,  help='dim of grid')
parser.add_argument('--dir', type=str, default = '', help='dir')
parser.add_argument('--lr',type=float,default=0.001,help='set learning rate');
parser.add_argument('--weight_decay',type=float,default=1e-9,help='set weight decay');
parser.add_argument('--gpu','-gpu',type=str,default='0',help='set gpu');
parser.add_argument('--ply',type=int,default=0,help='write out ply');
parser.add_argument('--w','-w',type=float,default=1.0,help='train on all data');

opt = parser.parse_args();
if not 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '';
if opt.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu;
print('visible gpu '+os.environ['CUDA_VISIBLE_DEVICES']);
now = datetime.datetime.now();
if opt.dir == '':
    save_path = 'InvAtlasNet'+now.isoformat();
else:
    save_path = opt.dir;
dir_name =  os.path.join('log',save_path);
if not os.path.exists(dir_name):
    os.mkdir(dir_name);
logname = os.path.join(dir_name,'log.txt');

opt.manualSeed = random.randint(1,10000) # fix seed
print("Random Seed:", opt.manualSeed);
random.seed(opt.manualSeed);
torch.manual_seed(opt.manualSeed);

#Create train/test dataloader on new views and test dataset on new models
SVR = False;
if opt.mode == 'SVR':
    SVR = True;

dataset = ShapeNet( SVR=SVR, normal = False, class_choice = None, train=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

dataset_test = ShapeNet( SVR=SVR, normal = False, class_choice = None, train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers))

print('training set',len(dataset.datapath));
print('testing set',len(dataset_test.datapath));

cudnn.benchmark = True;
len_dataset = len(dataset);

#create network
network = InvAtlasNet(num_points=opt.pts_num,grid_num=opt.grid_num,grid_dim=opt.grid_dim,mode=opt.mode);
network.cuda();
#=====================================================
with open(logname,'a') as f:
    for v in sys.argv:
        f.write(str(v)+' ');
    f.write('\n');
    f.write(str(network)+'\n');
#=====================================================
if opt.model != '':
    try:
        load_model(opt.model,network);
    except:
        print("Try partial");
        partial_restore(network,opt.model);
    print(" Previous weight loaded ");
#======================================================
val_cd = AverageValueMeter();
val_cd.reset();
val_inv = AverageValueMeter();
val_inv.reset();
#=====================================================
network.eval();
for i, data in enumerate(dataloader_test, 0):
    img, points, cat, _ , _ = data;
    if SVR:
        cd,inv = eval_svr(network,val_cd,val_inv,points,img);
    else:
        cd,inv = eval_ae(network,val_cd,val_inv,points);
    val_cd.update(cd);
    val_inv.update(inv);
    dataset_test.perCatValueMeter[cat[0]].update(cd);
    if bool(opt.ply):
        if SVR:
            view_svr(dir_name,network,img,i,cat,opt);
        else:
            view_ae(dir_name,network,points,i,cat,opt);
    print('[%d: %d/%d] val loss:  %f '%(0,i,len(dataset_test)/opt.batchSize,cd));
write_log(logname,val_cd,val_inv,dataset_test);
best_cd[...] = np.asscalar(val_cd.avg.cpu().numpy()); + 0.01;
best_all[...] = np.asscalar(val_cd.avg.cpu().numpy()) + np.asscalar(val_inv.avg.cpu().numpy()); + 0.01;
#=====================================================
optimizer = None;
if opt.train_part == 'en':
    optimizer = optim.Adam(network.encoder.parameters(),lr = opt.lr,weight_decay=opt.weight_decay);
elif opt.train_part == 'de':
    optimizer = optim.Adam(network.decoder.parameters(),lr = opt.lr,weight_decay=opt.weight_decay);
elif opt.train_part == 'full':
    optimizer = optim.Adam(network.parameters(),lr = opt.lr,weight_decay=opt.weight_decay);
elif opt.train_part == 'inv':
    optimizer = optim.Adam(network.inv_decoder.parameters(),lr = opt.lr,weight_decay=opt.weight_decay);
elif opt.train_part == 'deinv':
    param = [];
    param.extend(network.decoder.parameters());
    param.extend(network.inv_decoder.parameters());
    optimizer = optim.Adam(param,lr = opt.lr,weight_decay=opt.weight_decay);
elif opt.train_part == 'redeinv':
    network.decoder.apply(weights_init);
    network.inv_decoder.apply(weights_init);
    param = [];
    param.extend(network.decoder.parameters());
    param.extend(network.inv_decoder.parameters());
    optimizer = optim.Adam(param,lr = opt.lr,weight_decay=opt.weight_decay);
#=====================================================
trainloss_acc0 = 1e-9;
trainloss_accs = 0;
train_cd = AverageValueMeter();
train_inv = AverageValueMeter();
#=====================================================
nb = len(dataset)/opt.batchSize
for epoch in range(opt.nepoch):
    #TRAIN MODE
    train_cd.reset();
    train_inv.reset();
    network.train()
    for i, data in enumerate(dataloader, 0):
        img, points, cat, _ , _= data;
        if SVR:
            loss,cd,inv_err = train_svr(network,optimizer,train_cd,train_inv,points,img,opt);
        else:
            loss,cd,inv_err = train_ae(network,optimizer,train_cd,train_inv,points,opt);
        trainloss_accs = trainloss_accs * 0.99 + loss.data[0];
        trainloss_acc0 = trainloss_acc0 * 0.99 + 1;
        print('[%d: %d/%d] train loss:%f,%f,%f/%f' %(epoch,i,nb,cd.data[0],inv_err.data[0],loss.data[0],trainloss_accs/trainloss_acc0));
    #VALIDATION
    val_cd.reset();
    val_inv.reset();
    for item in dataset_test.cat:
        dataset_test.perCatValueMeter[item].reset();
#======================================================
    network.eval()
    for i, data in enumerate(dataloader_test, 0):
        img, points, cat, _, _ = data
        if SVR:
            cd,inv = eval_svr(network,val_cd,val_inv,points,img);
        else:
            cd,inv = eval_ae(network,val_cd,val_inv,points);
        val_cd.update(cd);
        val_inv.update(inv);
        dataset_test.perCatValueMeter[cat[0]].update(cd);
        print('[%d: %d/%d] val loss:%f ' %(epoch,i,len(dataset_test)/opt.batchSize,cd));
    write_log(logname,val_cd,val_inv,dataset_test,train_cd,train_inv,epoch);
#=======================================================
    save_model(logname,dir_name,network,opt,val_cd.avg,val_cd.avg+val_inv.avg);