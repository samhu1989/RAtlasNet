from __future__ import division;
from __future__ import print_function;
import os;
import sys;
#dynamic import the network modules to local namespace
from importlib import import_module;
def get_tasks(opt):
    tasks = [];
    for taskname in opt.exec:
        pn = os.path.dirname(__file__);
        mn = pn.split(os.sep)[-1]
        mpn = pn.split(os.sep)[-2]
        submn = taskname;
        m = import_module(mpn+'.'+mn+'.'+submn);
        tasks.append(m.RealTask());
    return tasks;