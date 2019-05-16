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

sys.path.append("./ext/");
import dist_chamfer as ext;
distChamfer =  ext.chamferDist();

def eval_ae(net,pts):
    with torch.no_grad():
        points = Variable(pts);
        points = points.transpose(2,1).contiguous();
        points = points.cuda();
        out = net(points);
        dist1, dist2 = distChamfer(points.transpose(2,1).contiguous(),out['y']);
        cd = (torch.mean(dist1)) + (torch.mean(dist2))
        inv_err = torch.mean(torch.sum((out['inv_x'] - out['grid_x'])**2,dim=2));
    return cd.data.cpu().numpy(),inv_err.data.cpu().numpy();

def train_ae(net,optim,cd_meter,inv_meter,pts,opt):
    optim.zero_grad();
    points = Variable(pts,requires_grad=True);
    points = points.transpose(2,1).contiguous();
    points = points.cuda();
    out = net(points);
    dist1, dist2 = distChamfer(points.transpose(2,1).contiguous(),out['y']);
    cd = (torch.mean(dist1)) + (torch.mean(dist2))
    inv_err = torch.mean(torch.sum((out['inv_x'] - out['grid_x'])**2,dim=2));
    cd_meter.update(cd.data.cpu().numpy());
    inv_meter.update(inv_err.data.cpu().numpy())
    loss = cd + opt['w']*inv_err;
    loss.backward();
    optim.step();
    return loss,cd,inv_err;
    
def eval_svr(net,pts,img):
    with torch.no_grad():
        img = Variable(img);
        img = img.cuda();
        points = Variable(pts);
        points = points.cuda();
        out = net(img);
        dist1, dist2 = distChamfer(points,out['y']);
        cd = (torch.mean(dist1)) + (torch.mean(dist2));
        inv_err = torch.mean(torch.sum((out['inv_x'] - out['grid_x'])**2,dim=2));
    return cd.data.cpu().numpy(),inv_err.data.cpu().numpy();
    
def train_svr(net,optim,cd_meter,inv_meter,pts,img,opt):
    optim.zero_grad();
    img = Variable(img,requires_grad=True);
    img = img.cuda();
    points = Variable(pts);
    points = points.cuda();
    out = net(img);
    dist1, dist2 = distChamfer(points,out['y']);
    cd = (torch.mean(dist1)) + (torch.mean(dist2));
    inv_err = torch.mean(torch.sum((out['inv_x'] - out['grid_x'])**2,dim=2));
    cd_meter.update(cd.data.cpu().numpy());
    inv_meter.update(inv_err.data.cpu().numpy());
    loss = cd + opt['w']*inv_err;
    loss.backward();
    optim.step();
    return loss,cd,inv_err;
    
def write_log(logfile,val_cd,val_inv,dataset_test,train_cd=None,train_inv=None,epoch=None):
    log_dict = {};
    log_dict['val_cd'] = val_cd.avg;
    log_dict['val_inv'] = val_inv.avg;
    for item in dataset_test.cat:
        print(item,dataset_test.perCatValueMeter[item].avg)
        log_dict.update({item:dataset_test.perCatValueMeter[item].avg})
    if train_cd is not None:
        log_dict['train_cd'] = train_cd.avg;
    if train_inv is not None:
        log_dict['train_inv'] = train_inv.avg;
    if epoch is not None:
        log_dict['epoch'] = epoch;
    logfile.write('json_stats: '+json.dumps(log_dict)+'\n');
    return;
        
bestnum = 3;
best_cd = np.zeros(bestnum);
best_all = np.zeros(bestnum);
        
def save_model(logtxt,dirname,net,opt,vcd,vall):
    global best_cd;
    global best_all;
    cdname = dirname+os.sep+opt['mode']+'gn'+str(opt['grid_num'])+'_cd';
    allname = dirname+os.sep+opt['mode']+'gn'+str(opt['grid_num'])+'_all';
    name = dirname+os.sep+opt['mode']+'gn'+str(opt['grid_num'])+'_current';
    sdict = net.state_dict();
    torch.save(sdict,name+'.pth');
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
        print('saving model at '+cdname+'_%d'%(bidx)+'.pth');
        torch.save(sdict,cdname+'_%d'%(bidx)+'.pth');
        logtxt.write('saving model at '+cdname+'_%d'%(bidx)+'.pth\n');
        logtxt.write('best_cd:'+np.array2string(best_cd,precision=6,separator=',')+'\n');
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
        print('saving model at '+allname+'_%d'%(bidx)+'.pth\n');
        torch.save(sdict,allname+'_%d'%(bidx)+'.pth\n');
        logtxt.write('saving model at '+allname+'_%d'%(bidx)+'.pth\n');
        logtxt.write('best_all:'+np.array2string(best_all,precision=6,separator=',')+'\n');
        
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
        self.optim = optim.SGD(self.net.parameters(),lr=self.opt['lr'],weight_decay=self.opt['weight_decay']);
        for group in self.optim.param_groups:
            group.setdefault('initial_lr', group['lr']);
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim,40,eta_min=0,last_epoch=self.opt['last_epoch']);
        #
        self.train_loss_acc0 = 1e-9;
        self.train_loss_accs = 0;
        self.eval();
        write_log(self.logtxt,self.valid_cd,self.valid_inv,self.valid_data,None,None,self.cnt);
        best_all.fill(self.opt['w']*self.valid_inv.avg+self.valid_cd.avg);
        best_cd.fill(self.valid_cd.avg)
        
    def eval(self):
        self.valid_cd.reset();
        self.valid_inv.reset();
        for item in self.valid_data.cat:
            self.valid_data.perCatValueMeter[item].reset();
        self.net.eval();
        for i, data in enumerate(self.valid_load, 0):
            img, points, cat, _, _ = data;
            if self.SVR:
                cd,inv = eval_svr(self.net,points,img);
            else:
                cd,inv = eval_ae(self.net,points);
            self.valid_cd.update(cd);
            self.valid_inv.update(inv);
            self.valid_data.perCatValueMeter[cat[0]].update(cd);
            print('[%d: %d/%d] val loss:%f ' %(self.cnt,i,len(self.valid_data)/self.opt['batchSize'],cd));
        
            
    def train(self):
        self.lr_scheduler.step();
        self.net.train()
        for i, data in enumerate(self.train_load, 0):
            img, points, cat, _ , _= data;
            if self.SVR:
                loss,cd,inv_err = train_svr(self.net,self.optim,self.train_cd,self.train_inv,points,img,self.opt);
            else:
                loss,cd,inv_err = train_ae(self.net,self.optim,self.train_cd,self.train_inv,points,self.opt);
            self.train_loss_accs = self.train_loss_accs * 0.99 + loss.data.cpu().numpy();
            self.train_loss_acc0 = self.train_loss_acc0 * 0.99 + 1;
            print('[%d: %d/%d] train loss:%f,%f,%f/%f' %(self.cnt+self.opt['last_epoch'],i,len(self.train_data)//self.opt['batchSize'],cd.data.cpu().numpy(),inv_err.data.cpu().numpy(),loss.data.cpu().numpy(),self.train_loss_accs/self.train_loss_acc0));

    def load_pretrain(self):
        if self.opt['model']!='':
            partial_restore(self.net,self.opt['model']);
            print("Previous weights loaded");
        
    def step(self):
        if self.cnt == 0:
            return;
        self.train();
        self.eval();
        write_log(self.logtxt,self.valid_cd,self.valid_inv,self.valid_data,self.train_cd,self.train_inv,self.cnt+self.opt['last_epoch']);
        save_model(self.logtxt,self.tskdir,self.net,self.opt,self.valid_cd.avg,self.valid_cd.avg+self.opt['w']*self.valid_inv.avg);