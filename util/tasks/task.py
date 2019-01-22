import os;
import json;
class Task(object):
    def __init__(self):
        super(Task,self).__init__();
        self.cnt = 0;
        self.opt = None;
        self.net = None;
        return;
    
    def run(self,*args,**kargs):
        print("Empty Task");
        pass
        
    def __call__(self,*args,**kwargs):
        self.init_tsk(*args,**kwargs);
        self.run(*args,**kwargs);
        self.done_tsk();
        
    def init_tsk(self,*args,**kwargs):
        self.net = args[0];
        self.opt = kwargs;
        if self.cnt > 0:
            return;
        if 'dir' in self.opt and self.opt['dir'] !='':
            self.logdir = self.opt['dir'];
        else:
            self.logdir = '.'+os.sep+'log'+os.sep+self.net.__class__.__name__;
        if not os.path.exists(self.logdir):
            os.mkdir(self.logdir);
        self.tskdir = self.logdir + os.sep + self.tskname;
        if not os.path.exists(self.tskdir):
            os.mkdir(self.tskdir);
        if self.opt['ply']:
            self.plydir = self.tskdir+os.sep+'ply';
            if not os.path.exists(self.plydir):
                os.mkdir(self.plydir);
        self.logtxt = open(self.tskdir+os.sep+'log.txt','a');
        self.logtxt.write(str(self.net)+'\n');
        self.logtxt.write('options:'+json.dumps(self.opt)+'\n');
        return;
            
    def done_tsk(self):
        self.cnt += 1;
	self.logtxt.flush();
        return;
        
    def __del__(self):
        self.logtxt.close();