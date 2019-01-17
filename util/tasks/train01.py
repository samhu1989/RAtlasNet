#
import os;
from .task import Task;
from ..ply import *;
from ..Lutils import *;
from ..datasets import *;
import torch;
from torch.autograd import Variable
import torch.nn as nn
import math;

sys.path.append("./ext/");
import dist_chamfer as ext;
distChamfer =  ext.chamferDist();

class RealTask(Task):
    def __init__(self):
        super(RealTask,self).__init__();
        self.tskname = os.path.basename(__file__).split('.')[0]
        
    def run(self,*args,**kwargs):
        self.start();
        self.step();
        return;
        
    def start(self):
        if self.cnt > 0:
            return;
        if self.opt['mode'] != 'OPT':
            assert False, self.taskname + ' must run under AE mode but got mode:'+self.opt['mode'];
        self.loadData();
        self.createOptim();
        
    def step(self):
        randw = torch.FloatTensor(self.Lw.size());
        randw.uniform_(0.0,1.0);
        randw = self.Lw*randw;
        randgrid = interp(self.xgrid,self.Li,randw);
        out = self.net(randgrid,randgrid);
        y = out['y'];
        invy = out['inv_x'];
        rgrid = out['grid_x'];
        dist1, dist2 = distChamfer(self.gt,y);
        cd = (torch.mean(dist1)) + (torch.mean(dist2));
        inv_err = torch.mean(torch.sum((invy - rgrid)**2,dim=2));
        loss = cd + self.opt['w']*inv_err;
        self.optim.zero_grad();
        loss.backward();
        self.optim.step();
        if self.cnt == 0 or (self.cnt+1) == math.pow(2,math.floor(math.log2(self.cnt+1))) :
            self.logtxt.write("niter:"+str(self.cnt)+"cd:"+str(cd.data)+"inv"+str(inv_err.data)+'\n');
            if self.opt['ply']:
                self.writeply();
        
    def writeply(self):
        out = self.net(self.xgrid,self.xgrid);
        points = out['y'].data.cpu().numpy()[0,...];
        ply_path = self.plydir+os.sep+os.path.basename(self.opt['data']).split('.')[0]+'_w%f_gn%d'%(self.opt['w'],self.opt['grid_num']);
        if not os.path.exists(ply_path):
            os.mkdir(ply_path);
        write_ply(ply_path+os.sep+'iter%03d.ply'%(self.cnt),points = pd.DataFrame(points),faces=pd.DataFrame(self.plyface));
        
    def loadData(self):
        # GT
        data = read_ply(self.opt['data']);
        pgt = data['points'];
        gtpoints = np.zeros([1,pgt.shape[0],pgt.shape[1]],dtype=np.float32);
        gtpoints[0,...] = pgt;
        self.gt = Variable(torch.from_numpy(gtpoints));
        if torch.cuda.is_available():
            self.gt = self.gt.cuda();
        #Grid
        grid,Li,Lw,fidx = sphere_grid(1,self.opt['pts_num'],'link');
        T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
        if self.opt['ply']:
            self.plyface = np.zeros(shape=[fidx.shape[0]],dtype=T);
            for j in range(fidx.shape[0]):
                self.plyface[j] = (3,fidx[j,0],fidx[j,1],fidx[j,2]);
        #
        self.xgrid = Variable(grid);
        if torch.cuda.is_available():
            self.xgrid = self.xgrid.cuda();
        
        self.Lw = Lw;
        self.Li = Li;
    
    def createOptim(self):
        self.optim = optim.Adam(self.net.parameters(),lr = self.opt['lr'],weight_decay=self.opt['weight_decay']);