from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F;
from distutils.version import LooseVersion;
import scipy;
from scipy.sparse import dok_matrix;
from scipy.sparse import csr_matrix;
from scipy.spatial import ConvexHull;
from scipy.spatial import Delaunay
import sys;
from .ply import *;

def partial_restore(net,path,keymap={}):
    olddict = torch.load(path);
    #print(olddict.keys());
    mdict = net.state_dict();
    #print(olddict.keys());
    #print(mdict.keys());
    newdict = {};
    for k,v in mdict.items():
        if ( k in olddict ) and ( v.size() == olddict[k].size() ):
            newdict[k] = olddict[k];
        elif k in keymap and keymap[k] in olddict:
            newdict[k] = olddict[keymap[k]];
        else:
            print(k,'in model is not assigned');
    mdict.update(newdict);
    net.load_state_dict(mdict);
    
def resize(*input):
    y = input[0];
    size = input[1:];
    if isinstance(size[0],torch.Size):
        size = size[0];
    if isinstance(y,torch.FloatTensor) or isinstance(y,torch.cuda.FloatTensor):
        y = y.resize_(size);
    else:
        y = y.resize(*size);
    return y;

def randsphere1(m=None):
    pts = np.random.normal(size=[m,3]).astype(np.float32);
    pts += 1e-8;
    pts_norm = np.sqrt(np.sum(np.square(pts),axis=1,keepdims=True));
    pts /= pts_norm;
    return pts.transpose(1,0);

def randsphere2(m=None):
    pts = np.zeros([3,m],np.float32);
    n = np.linspace(1,m,m);
    n += np.random.normal();
    tmp = 0.5*(np.sqrt(5)-1)*n;
    theta = 2.0*np.pi*(tmp - np.floor(tmp));
    pts[0,:] = np.cos(theta);
    pts[1,:] = np.sin(theta);
    pts[2,:] = 2.0*(n - n.min()) / (n.max()-n.min()) - 1.0;
    scale = np.sqrt(1 - np.square(pts[2,:]));
    pts[0,:] *= scale;
    pts[1,:] *= scale;
    return pts;

randsphere = randsphere2

def triangulateSphere(pts):
    hull_list = [];
    for i in range(pts.shape[0]):
        pt = pts[i,...];
        hull = ConvexHull(pt.transpose(1,0));
        for j in range(hull.simplices.shape[0]):
            simplex = hull.simplices[j,:];
            triangle = pt[:,simplex];
            m = triangle[:,0];
            p0p1 = triangle[:,1] -  triangle[:,0];
            p1p2 = triangle[:,2] -  triangle[:,1];
            k = np.cross(p0p1,p1p2);
            if np.dot(m,k) < 0:
                tmp = hull.simplices[j,1];
                hull.simplices[j,1] = hull.simplices[j,2];
                hull.simplices[j,2] = tmp;
        hull_list.append(hull);
    return hull_list;

def triangulate(pts):
    hull_list = [];
    for i in range(pts.shape[0]):
        pt = pts[i,...];
        hull = Delaunay(pt.transpose(1,0));
        for j in range(hull.simplices.shape[0]):
            simplex = hull.simplices[j,:];
            triangle = pt[:,simplex];
            m = np.array([0,0,0],dtype=np.float32);
            m[0:2] = triangle[:,0];
            p0p1 = np.array([0,0,0],dtype=np.float32);
            p1p2 = np.array([0,0,0],dtype=np.float32);
            p0p1[0:2] = triangle[:,1] -  triangle[:,0];
            p1p2[0:2] = triangle[:,2] -  triangle[:,1];
            k = np.cross(p0p1,p1p2);
            if np.dot(m,k) < 0:
                tmp = hull.simplices[j,1];
                hull.simplices[j,1] = hull.simplices[j,2];
                hull.simplices[j,2] = tmp;
        hull_list.append(hull);
    return hull_list;

def sphere_grid(b,m,Ltype):
    if Ltype == 'rgf' or Ltype == 'grgf' or Ltype == 'link':
        return sphere_grid_link(b,m);
    else:
        return sphere_grid_laplace(b,m,Ltype);
        
def ply_grid(b,fpath):
    ply_data = read_ply(fpath);
    pts = np.array(ply_data['points']);
    pts = pts.transpose((1,0));
    pts = pts.reshape((1,3,-1));
    fidx = np.array(ply_data['mesh']);
    grid = torch.FloatTensor(pts);
    Li,Lw = link(b,pts,fidx);
    grid = grid.repeat(b,1,1).contiguous();
    return grid,Li,Lw,fidx;

def repeat_face(fidx,n,num):
    simp = fidx.simplices;
    newsimp = np.zeros([simp.shape[0]*n,3],dtype=simp.dtype)
    for i in range(n):
        newsimp[i*simp.shape[0]:(i+1)*simp.shape[0],:] = simp + i*num;
    fidx.simplices = newsimp;
    
def patch_grid(b,m,n):
    pts = np.random.uniform(0,1,[1,2,m//n]).astype(np.float32);
    fidx = triangulate(pts);
    repeat_face(fidx[0],n,m//n);
    grid = torch.FloatTensor(pts);
    grid = grid.repeat(b,1,1).contiguous();
    if torch.cuda.is_available():
        grid = grid.cuda();
    return grid,None,None,fidx[0].simplices.copy();
    
def sphere_grid_laplace(b,m,Ltype):
    sphere = randsphere(m);
    hulllst = triangulateSphere(sphere.reshape([1,3,-1]));
    grid = torch.FloatTensor(sphere);
    if Ltype=='cot':
        Li,Lw = laplace_cot(b,sphere,hulllst[0].simplices.copy());
    elif Ltype == 'graph':
        Li,Lw = laplace(b,sphere,hulllst[0].simplices.copy());
    elif Ltype == 'rgd':
        Li,Lw = graph_neighbor(b,sphere,hulllst[0].simplices.copy());
    elif Ltype == 'cn':
        Li,Lw = cot_neighbor(b,sphere,hulllst[0].simplices.copy());
    else:
        print('unimplemented type of laplacian operator');
        sys.exit(-1);
    grid = grid.repeat(b,1,1).contiguous();
    if torch.cuda.is_available():
        Li = Li.cuda();
        Lw = Lw.cuda();
        grid = grid.cuda();
    return grid,Li,Lw,hulllst[0].simplices.copy();

def sphere_grid_link(b,m):
    sphere = randsphere(m);
    hulllst = triangulateSphere(sphere.reshape([1,3,-1]));
    grid = torch.FloatTensor(sphere);
    Li,Lw = link(b,sphere,hulllst[0].simplices.copy());
    grid = grid.repeat(b,1,1).contiguous();
    return grid,Li,Lw,hulllst[0].simplices.copy();

def scale_sphere_to_cube(sphere):
    th = 1.1e-1;
    for i in range(sphere.shape[1]):
        v = sphere[:,i].copy();
        idx = np.argsort(-np.abs(v));
        if  (np.abs(np.abs(sphere[idx[0],i]) - np.abs(sphere[idx[1],i]))<th) and (np.abs(np.abs(sphere[idx[0],i]) - np.abs(sphere[idx[2],i]))< th):
            sphere[:,i] = np.sign(sphere[:,i]);
        elif np.abs(np.abs(sphere[idx[0],i]) - np.abs(sphere[idx[1],i])) < th:
            sphere[idx[0:2],i] = np.sign(sphere[idx[0:2],i]);
        else:
            sphere[idx[0],i] = np.sign(sphere[idx[0],i]);
            
def cube_grid_link(b,m):
    sphere = randsphere(m);
    hulllst = triangulateSphere(sphere.reshape([1,3,-1]));
    scale_sphere_to_cube(sphere);
    grid = torch.FloatTensor(sphere);
    Li,Lw = link(b,sphere,hulllst[0].simplices.copy());
    grid = grid.repeat(b,1,1).contiguous();
    if torch.cuda.is_available():
        Li = Li.cuda();
        Lw = Lw.cuda();
        grid = grid.cuda();
    return grid,Li,Lw,hulllst[0].simplices.copy();

def link(b,sphere,fidx):
    n = sphere.shape[-1];
    D = dok_matrix((n,n),dtype=np.float32);
    for i in range(fidx.shape[0]):
        v0 = fidx[i,0];
        v1 = fidx[i,1];
        v2 = fidx[i,2];
        D[v0,v1] = 1.0;
        D[v1,v0] = 1.0;
        D[v0,v2] = 1.0;
        D[v2,v0] = 1.0;
        D[v1,v2] = 1.0;
        D[v2,v1] = 1.0;
    D = D.tocsr();
    #D += scipy.sparse.eye(n);
    maxnum = 0; 
    for i in range(n):
        num = D.indptr[i+1] - D.indptr[i];
        if maxnum < num:
            maxnum = num;
    Liv = np.zeros([n,maxnum],dtype=np.int);
    Lwv = np.zeros([b,n,maxnum],dtype=np.float32);
    for i in range(n):
        num = D.indptr[i+1] - D.indptr[i];
        Liv[i,0:num] = D.indices[D.indptr[i]:D.indptr[i+1]];
        for bi in range(b):
            Lwv[bi,i,0:num] = D.data[D.indptr[i]:D.indptr[i+1]];
    Li = torch.LongTensor(Liv.flatten().tolist());
    Lw = torch.FloatTensor(Lwv);
    return Li,Lw;

def laplace(b,sphere,fidx):
    n = sphere.shape[-1];
    D = dok_matrix((n,n),dtype=np.float32);
    for i in range(fidx.shape[0]):
        v0 = fidx[i,0];
        v1 = fidx[i,1];
        v2 = fidx[i,2];
        D[v0,v1] = 1.0;
        D[v1,v0] = 1.0;
        D[v0,v2] = 1.0;
        D[v2,v0] = 1.0;
        D[v1,v2] = 1.0;
        D[v2,v1] = 1.0;
    D = D.tocsr() ;
    rsum = D.sum(1);
    for i in range(n):
        D.data[D.indptr[i]:D.indptr[i+1]] /= rsum[i,0];
    D -= scipy.sparse.eye(n);
    maxnum = 0; 
    for i in range(n):
        num = D.indptr[i+1] - D.indptr[i];
        if maxnum < num:
            maxnum = num;
    Liv = np.zeros([n,maxnum],dtype=np.int);
    Lwv = np.zeros([b,n,maxnum],dtype=np.float32);
    for i in range(n):
        num = D.indptr[i+1] - D.indptr[i];
        Liv[i,0:num] = D.indices[D.indptr[i]:D.indptr[i+1]];
        for bi in range(b):
            Lwv[bi,i,0:num] = D.data[D.indptr[i]:D.indptr[i+1]];
    Li = torch.LongTensor(Liv.flatten().tolist());
    Lw = torch.FloatTensor(Lwv);
    return Li,Lw;

def graph_neighbor(b,sphere,fidx):
    n = sphere.shape[-1];
    D = dok_matrix((n,n),dtype=np.float32);
    for i in range(fidx.shape[0]):
        v0 = fidx[i,0];
        v1 = fidx[i,1];
        v2 = fidx[i,2];
        D[v0,v1] = 1.0;
        D[v1,v0] = 1.0;
        D[v0,v2] = 1.0;
        D[v2,v0] = 1.0;
        D[v1,v2] = 1.0;
        D[v2,v1] = 1.0;
    D = D.tocsr() ;
    maxnum = 0; 
    for i in range(n):
        num = D.indptr[i+1] - D.indptr[i];
        if maxnum < num:
            maxnum = num;
    Liv = np.zeros([n,maxnum],dtype=np.int);
    Lwv = np.zeros([b,n,maxnum],dtype=np.float32);
    for i in range(n):
        num = D.indptr[i+1] - D.indptr[i];
        Liv[i,0:num] = D.indices[D.indptr[i]:D.indptr[i+1]];
        for bi in range(b):
            Lwv[bi,i,0:num] = D.data[D.indptr[i]:D.indptr[i+1]];
    Li = torch.LongTensor(Liv.flatten().tolist());
    Lw = torch.FloatTensor(Lwv);
    return Li,Lw;

def cot_neighbor(b,sphere,fidx):
    n = sphere.shape[-1];
    D = dok_matrix((n,n),dtype=np.float32);
    for i in range(fidx.shape[0]):
        vi0 = fidx[i,0];
        vi1 = fidx[i,1];
        vi2 = fidx[i,2];
        v0 = sphere[:,vi0];
        v1 = sphere[:,vi1];
        v2 = sphere[:,vi2];
        c01 = getcot(v0-v2,v1-v2);
        c02 = getcot(v0-v1,v2-v1);
        c12 = getcot(v1-v0,v2-v0);
        D[vi0,vi1] += c01;
        D[vi1,vi0] += c01;
        D[vi0,vi2] += c02;
        D[vi2,vi0] += c02;
        D[vi1,vi2] += c12;
        D[vi2,vi1] += c12;
    D = D.tocsr() ;
    D *= 0.5;
    maxnum = 0; 
    for i in range(n):
        num = D.indptr[i+1] - D.indptr[i];
        if maxnum < num:
            maxnum = num;
    Liv = np.zeros([n,maxnum],dtype=np.int);
    Lwv = np.zeros([b,n,maxnum],dtype=np.float32);
    for i in range(n):
        num = D.indptr[i+1] - D.indptr[i];
        Liv[i,0:num] = D.indices[D.indptr[i]:D.indptr[i+1]];
        for bi in range(b):
            Lwv[bi,i,0:num] = D.data[D.indptr[i]:D.indptr[i+1]];
    Li = torch.LongTensor(Liv.flatten().tolist());
    Lw = torch.FloatTensor(Lwv);
    return Li,Lw;

def getcot(a,b):
    return np.dot(a,b) / np.linalg.norm(np.cross(a,b));

def laplace_cot(b,sphere,fidx):
    n = sphere.shape[-1];
    D = dok_matrix((n,n),dtype=np.float32);
    for i in range(fidx.shape[0]):
        vi0 = fidx[i,0];
        vi1 = fidx[i,1];
        vi2 = fidx[i,2];
        v0 = sphere[:,vi0];
        v1 = sphere[:,vi1];
        v2 = sphere[:,vi2];
        c01 = getcot(v0-v2,v1-v2);
        c02 = getcot(v0-v1,v2-v1);
        c12 = getcot(v1-v0,v2-v0);
        D[vi0,vi1] += c01;
        D[vi1,vi0] += c01;
        D[vi0,vi2] += c02;
        D[vi2,vi0] += c02;
        D[vi1,vi2] += c12;
        D[vi2,vi1] += c12;
    D = D.tocsr() ;
    D *= 0.5;
    rsum = D.sum(1);
    for i in range(n):
        D.data[D.indptr[i]:D.indptr[i+1]] /= rsum[i,0];
    D -= scipy.sparse.eye(n);
    maxnum = 0; 
    for i in range(n):
        num = D.indptr[i+1] - D.indptr[i];
        if maxnum < num:
            maxnum = num;
    Liv = np.zeros([n,maxnum],dtype=np.int);
    Lwv = np.zeros([b,n,maxnum],dtype=np.float32);
    for i in range(n):
        num = D.indptr[i+1] - D.indptr[i];
        Liv[i,0:num] = D.indices[D.indptr[i]:D.indptr[i+1]];
        for bi in range(b):
            Lwv[bi,i,0:num] = D.data[D.indptr[i]:D.indptr[i+1]];
    Li = torch.LongTensor(Liv.flatten().tolist());
    Lw = torch.FloatTensor(Lwv);
    return Li,Lw;

def get_lossnet(name,net,data,n,dist,Ltype,output=False):
    img, points , _ , _ , _ = data;
    loss_net = None;
    if name.startswith('AE_'):
        points = Variable(points, volatile=True);
        points = points.transpose(2,1).contiguous();
        points = points.cuda();
        pointsReconstructed  = net(points);
        dist1, dist2 = dist(points.transpose(2,1).contiguous(), pointsReconstructed);
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2));
    elif name.startswith('SVR_'):
        img = Variable(img, volatile=True);
        img = img.cuda();
        points = Variable(points, volatile=True);
        points = points.cuda();
        grid,Li,Lw,fidx = sphere_grid(img.size()[0],n,Ltype);
        grid = Variable(grid, volatile=True);
        grid = grid.transpose(2,1).contiguous();
        grid = grid.cuda();
        Li = Variable(Li, volatile=True);
        Li = Li.cuda();
        Lw = Variable(Lw, volatile=True);
        Lw = Lw.cuda();
        pointsReconstructed = net(img,grid,Li,Lw);
        dist1, dist2 = dist(points, pointsReconstructed);
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2));
    if output:
        r = (loss_net,img,pointsReconstructed,fidx);
    else:
        r = loss_net;
    return r;

def write_img_and_ply(path,index,img,y,fidx,cat):
    ply_path = path+os.sep+'ply';
    if not os.path.exists(ply_path):
        os.mkdir(ply_path);
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    face = np.zeros(shape=[fidx.shape[0]],dtype=T);
    for i in range(fidx.shape[0]):
        face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
    for i in range(y.shape[0]):
        write_ply(ply_path+os.sep+'%02d_%02d_%s.ply'%(index,i,cat),points = pd.DataFrame(y[i,...]),faces=pd.DataFrame(face));
    im_path = path+os.sep+'im';
    if not os.path.exists(im_path):
        os.mkdir(im_path);
    fig = plt.figure(figsize=(2.44,2.44));
    img = img.transpose(0,2,3,1);
    for i in range(img.shape[0]):
        plt.imshow(img[i,...]);
        fig.savefig(im_path+os.sep+'%02d_%02d_%s.png'%(index,i,cat));
    plt.close(fig);
    
def get_lossnet_train(name,net,data,n,dist,Ltype):
    img, points , _ , _ , _ = data;
    loss_net = None;
    if name.startswith('AE_'):
        points = Variable(points);
        points = points.transpose(2,1).contiguous();
        points = points.cuda();
        pointsReconstructed  = net(points);
        dist1, dist2 = dist(points.transpose(2,1).contiguous(), pointsReconstructed);
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2));
    elif name.startswith('SVR_'):
        img = Variable(img);
        img = img.cuda();
        points = Variable(points);
        points = points.cuda();
        grid,Li,Lw,_ = sphere_grid(img.size()[0],n,Ltype);
        grid = Variable(grid);
        grid = grid.transpose(2,1).contiguous();
        grid = grid.cuda();
        Li = Variable(Li);
        Li = Li.cuda();
        Lw = Variable(Lw);
        Lw = Lw.cuda();
        pointsReconstructed = net(img,grid,Li,Lw);
        if dist.m == 'plain':
            dist1, dist2 = dist(pointsReconstructed,points);
        else:
            dist1, dist2 = dist(pointsReconstructed,points,Li,Lw);
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2));
    return loss_net;
    
def interp(x,Li,Lw):
    if isinstance(x,Variable) and not isinstance(Li,Variable):
        Li = Variable(Li);
        Lw = Variable(Lw);
    if x.is_cuda:
        Li = Li.cuda();
        Lw = Lw.cuda();
    Lx = x;
    Lx = Lx.index_select(dim=2,index=Li).contiguous();
    Lw = Lw / Lw.sum(dim=-1,keepdim=True);
    Lw = resize(Lw,Lw.size()[0],1,Lw.size()[1],Lw.size()[-1]);
    Lx = resize(Lx,Lx.size()[0],Lx.size()[1],Lx.size()[2]//Lw.size()[-1],Lw.size()[-1]);
    Lx = resize((Lw*Lx).sum(dim=-1).contiguous(),x.size());
    return Lx;