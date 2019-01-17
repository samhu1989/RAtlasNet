from __future__ import absolute_import;
from __future__ import print_function;
import os;
import sys;
import torch;
#dynamic import the network modules to local namespace
from importlib import import_module;
for pn,dns,fns in os.walk(os.path.dirname(__file__)):
    mn = pn.split(os.sep)[-1];
    for fn in fns:
        if fn.startswith('net_') and fn.endswith('.py'):
            submn = fn.split('.')[0];
            m = import_module('%s.%s'%(mn,submn));
            for name in dir(m):
                if not name in locals().keys():
                    locals()[name] = m.__getattribute__(name);
                    
def get_net(opt):
    args = [];
    net = None;
    try:
        print('Getting '+opt.net);
        net = eval(opt.net)(*args,**opt.__dict__);
    except Exception as e:
        print('running into exceptions');
        print(e);
    else:
        print('Got '+opt.net);
    if torch.cuda.is_available():
        net = net.cuda();
        print('using cuda');
    return net;