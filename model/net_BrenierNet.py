#
from .net_InvAtlatNet import *;
class Brenier(nn.Module):
    def __init__(self):
        super(Brenier,self).__init__();
        
    def forward(self,*input):
        #b,3,n
        X = input[0];
        #b,3,m
        Y = input[1];
        #b,m
        H = input[2];
        Y = Y.transpose(2,1).contiguous();
        V = torch.bmm(Y,X)+H.view(H.size(0),H.size(1),1)
        V,_ = torch.max(V,dim=1);
        return V;
        
class BlockBrenier(nn.Module):
    def __init__(self):
        super(BlockBrenier,self).__init__();
        self.base = Brenier();
        
    def forward(self,*input):
        X = input[0];
        Ys = input[1];
        Hs = input[2];
        maxval = self.base(*[X,Ys[0],Hs[0]]);
        for i in range(1,len(Ys)):
            maxval = torch.max(maxval,self.base(*[X,Ys[i],Hs[i]]));
        return maxval;
#numeric gradient        
class NumGrad(nn.Module):
    def __init__(self,eps=1e-4):
        self.eps = eps;
        super(NumGrad,self).__init__();
        
    def forward(self,func,X,*other):
        grad_lst = [];
        input = [];
        input.append(X);
        input.extend(other);
        e = func(*input);
        for i in range(X.size(1)):
            eps = torch.zeros(X.size(0),X.size(1),X.size(2));
            eps[:,i,:] = self.eps;
            input[0] = X + eps.type(X.type());
            grad = ( func(*input) - e ) / self.eps ;
            grad_lst.append(grad.view(grad.size(0),1,grad.size(1)));
        return torch.cat(grad_lst,1).contiguous();
        
class Ydecoder(nn.Module):
    def __init__(self,size=1024,ynum=100,grid_dim=3,bn=False):
        self.bn = bn;
        self.grid_dim = grid_dim;
        super(Ydecoder,self).__init__();
        self.fc1 = nn.Linear(size,size//2);
        self.fc2 = nn.Linear(size//2,self.grid_dim*ynum);
        self.th = nn.Tanh();
        if self.bn:
            self.bn1 = torch.nn.BatchNorm1d(size//2);
    
    def forward(self,x):
        x = F.relu(self.fc1(x));
        x = self.th(self.fc2(x));
        return x.view(x.size(0),self.grid_dim,-1);
        
class Hdecoder(nn.Module):
    def __init__(self,size=1024,ynum=100,bn=False):
        self.bn = bn;
        super(Hdecoder,self).__init__();
        self.fc1 = nn.Linear(size,size//2);
        self.fc2 = nn.Linear(size//2,ynum);
        self.th = nn.Tanhshrink();
        if self.bn:
            self.bn1 = torch.nn.BatchNorm1d(size//2);
    
    def forward(self,x):
        x = F.relu(self.bn1(self.fc1(x)));
        return self.th(self.fc2(x));        
        
class BrenierNet(nn.Module):
    def __init__(self,*args,**kwargs):
        super(BrenierNet, self).__init__()
        self.pts_num = kwargs['pts_num']
        self.bottleneck_size = 1024
        self.grid_num = kwargs['grid_num']
        self.grid_dim = kwargs['grid_dim']
        self.mode = kwargs['mode']
        self.bn = True
        if self.mode == 'SVR':
            self.encoder = resnet.resnet18(pretrained=False,num_classes=1024);
        elif self.mode == 'AE':
            self.encoder = nn.Sequential(
                PointNetfeat(self.pts_num, global_feat=True, trans = False),
                nn.Linear(1024, self.bottleneck_size),
                nn.BatchNorm1d(self.bottleneck_size),
                nn.ReLU()
                )
        elif self.mode == 'OPT':
            self.encoder = OptEncoder(self.bottleneck_size)
            self.bn = False
        else:
            assert False,'unkown mode of BlendedAtlasNet'
        self.Yde = nn.ModuleList([Ydecoder(size=self.bottleneck_size,ynum=self.pts_num*4//self.grid_num,bn=self.bn) for i in range(0,self.grid_num)]);
        self.Hde = nn.ModuleList([Hdecoder(size=self.bottleneck_size,ynum=self.pts_num*4//self.grid_num,bn=self.bn) for i in range(0,self.grid_num)]);
        self.gradfn = NumGrad();
        self.brenier = BlockBrenier();
        self._init_layers()

    def forward(self,*input):
        x = input[0]
        if x.dim() == 4:
            x = x[:,:3,:,:].contiguous()
        f = self.encoder(x)
        outs = []
        grid = None
        if len(input) > 1:
            grid = input[1]
        else:
            grid = self.rand_grid(f)
        Ys = [];
        Hs = [];
        for i in range(0,self.grid_num):
            Ys.append(self.Yde[i](f));
            Hs.append(self.Hde[i](f));
        yo = self.gradfn(self.brenier,grid,Ys,Hs);
        #transpose from (bn,3,#points) to (bn,#points,3)
        yo = yo.transpose(2,1).contiguous()
        grid = grid.transpose(2,1).contiguous()
        out = {}
        out['y'] = yo
        out['inv_x'] = 0.0
        out['grid_x'] = grid
        return out
    
    def rand_grid(self,x):
        rand_grid = torch.FloatTensor(x.size(0),self.grid_dim,self.pts_num);
        if self.grid_dim == 3:
            rand_grid.normal_(0.0,1.0)
            rand_grid += 1e-9
            rand_grid = rand_grid / torch.norm(rand_grid,p=2.0,dim=1,keepdim=True);
        else:
            rand_grid.uniform_(0.0,1.0)
        if isinstance(x,Variable):
            rand_grid = Variable(rand_grid)
        if x.is_cuda:
            rand_grid = rand_grid.cuda()
        return rand_grid
    
    def _init_layers(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels;
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,nn.Conv1d):
                m.weight.data.normal_(0.0,0.02)
            elif isinstance(m,nn.BatchNorm1d):
                m.weight.data.normal_(1.0,0.02)
                m.bias.data.fill_(0)
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                m.weight.data.normal_(0.0,0.02)
                m.bias.data.normal_(0.0,0.02)