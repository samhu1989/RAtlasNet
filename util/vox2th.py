import torch;
from .pyvox.parser import VoxParser;
from .pyvox.writer import VoxWriter;
from .pyvox.models import Vox;
import sys;
import numpy as np;

def voxfile2th(path):
    m = VoxParser(path).parse();
    d = m.to_dense();
    d = d / np.max(d)
    d = torch.from_numpy(d);
    return d;
    
def th2voxfile(vox,path):
    d = vox.cpu().numpy();
    vox = Vox.from_dense(128*d.astype("int32"));
    VoxWriter(path,vox).write();
    
    