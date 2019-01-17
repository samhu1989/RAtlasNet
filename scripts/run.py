from __future__ import print_function
from __future__ import division
import sys;
sys.path.append('./');
from util import *;
from model import *;
#
if __name__ == "__main__":
    #get options from command line inputs
    opt = option.get_option();
    #set gpu
    env.set_env(opt);
    #create network
    net = nets.get_net(opt);
    #create tasks
    tasks = tasks.get_tasks(opt);
    for epoch in range(opt.nepoch):
        for task in tasks:
            task(net,**opt.__dict__);