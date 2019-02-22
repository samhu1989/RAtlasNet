#
import os;
from .task import Task;
from ..ply import *;
from ..Lutils import *;
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
        self.tskname = os.path.basename(__file__).split('.')[0];
        
    def run(self,*args,**kwargs):
        grid,Li,Lw,fidx = ply_grid(1,self.opt['grid_ply']);
        T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
        points = grid.numpy()[0,...];
        points = points.transpose((1,0));
        if self.opt['ply']:
            self.plyface = np.zeros(shape=[fidx.shape[0]],dtype=T);
            for j in range(fidx.shape[0]):
                self.plyface[j] = (3,fidx[j,0],fidx[j,1],fidx[j,2]);
            write_ply('./debug.ply',points = pd.DataFrame(points),faces=pd.DataFrame(self.plyface));
        return;
        
        

        
        
        

        
