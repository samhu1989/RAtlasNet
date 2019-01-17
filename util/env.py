import os;
import torch.backends.cudnn as cudnn
import random;
import torch
def set_env(opt):
    if not 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '';
    if opt.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu;
    print('visible gpu '+os.environ['CUDA_VISIBLE_DEVICES']);
    if opt.manualSeed == 0:
    	opt.manualSeed = random.randint(1,10000)
    print("Random Seed:", opt.manualSeed);
    random.seed(opt.manualSeed);
    torch.manual_seed(opt.manualSeed);    
    cudnn.benchmark = True;
	