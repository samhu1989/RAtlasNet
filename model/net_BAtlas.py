from .net_BlendedAtlasNet import *;

class PointMLP(nn.Module):
    def __init__(self, bottleneck_size=2500,odim=3,bn=True):
        self.bottleneck_size = bottleneck_size;
        super(PointMLP, self).__init__();
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size,self.bottleneck_size//2,1);
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size//2,self.bottleneck_size//4,1);
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//4,self.bottleneck_size//8,1);
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//8,odim,1);
        self.bn = bn;
        self.act = nn.Softmax(dim=1);
        if self.bn:
            self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size//2);
            self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//4);
            self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//8);
        
    def forward(self, x):
        if self.bn:
            x = F.relu(self.bn1(self.conv1(x)));
            x = F.relu(self.bn2(self.conv2(x)));
            x = F.relu(self.bn3(self.conv3(x)));
            x = self.act(self.conv4(x))
        else:
            x = F.relu(self.conv1(x));
            x = F.relu(self.conv2(x));
            x = F.relu(self.conv3(x));
            x = self.act(self.conv4(x))
        return x;

class BAtlas(nn.Module):
    def __init__(self,*args,**kwargs):
        super(BAtlas, self).__init__();
        self.pts_num = kwargs['pts_num'];
        self.bottleneck_size = 1024;
        self.grid_num = kwargs['grid_num'];
        self.grid_dim = kwargs['grid_dim'];
        self.mode = kwargs['mode'];
        self.bn = True;
        if self.mode == 'SVR':
            self.encoder = resnet.resnet18(pretrained=False,num_classes=1024);
        elif self.mode == 'AE':
            self.encoder = nn.Sequential(
                PointNetfeat(self.pts_num, global_feat=True, trans = False),
                nn.Linear(1024, self.bottleneck_size),
                nn.BatchNorm1d(self.bottleneck_size),
                nn.ReLU()
                );
        elif self.mode == 'OPT':
            self.encoder = OptEncoder(self.bottleneck_size);
            self.bn = False
        else:
            assert False,'unkown mode of BlendedAtlasNet'
        self.wdecoder = PointMLP(bottleneck_size=self.grid_dim+self.bottleneck_size,odim=self.grid_num,bn=self.bn);
        self.decoder = nn.ModuleList([PointMLP(bottleneck_size=self.grid_dim+self.bottleneck_size,odim=3,bn=self.bn) for i in range(0,self.grid_num)]);
        self.inv_decoder = PointGenCon(bottleneck_size=3+self.bottleneck_size,odim=3,bn=self.bn);
        self._init_layers();

    def forward(self,*input):
        x = input[0];
        if x.dim() == 4:
            x = x[:,:3,:,:].contiguous();
        f = self.encoder(x);
        outs = [];
        grid = None;
        if len(input) > 1:
            grid = input[1];
        else:
            grid = self.rand_grid(f);
        inv_e = 0.0;
        expf = f.unsqueeze(2).expand(f.size(0),f.size(1),grid.size(2)).contiguous();
        w = torch.cat((grid,expf),1).contiguous();
        w = self.wdecoder(w);
        w = w.view(w.size(0),w.size(1),1,w.size(2));
        yo = None;
        for i in range(0,self.grid_num):
            y = torch.cat((grid,expf),1).contiguous();
            y = self.decoder[i](y);
            y = y*w[:,i,:,:];
            if yo is None:
                yo = y;
            else:
                yo += y;
        invo = torch.cat((yo,expf),1).contiguous();
        invo = self.inv_decoder(invo);
        #transpose from (bn,3,#points) to (bn,3,#points)
        yo = yo.transpose(2,1).contiguous();
        invo = invo.transpose(2,1).contiguous();
        grid = grid.transpose(2,1).contiguous();
        out = {};
        out['y'] = yo;
        out['inv_x'] = invo;
        out['grid_x'] = grid;
        out['blend_w'] = w;
        return out;
    
    def rand_grid(self,x):
        rand_grid = torch.FloatTensor(x.size(0),self.grid_dim,self.pts_num);
        if self.grid_dim == 3:
            rand_grid.normal_(0.0,1.0);
            rand_grid += 1e-9;
            rand_grid = rand_grid / torch.norm(rand_grid,p=2.0,dim=1,keepdim=True);
        else:
            rand_grid.uniform_(0.0,1.0);
        if isinstance(x,Variable):
            rand_grid = Variable(rand_grid);
        if x.is_cuda:
            rand_grid = rand_grid.cuda();
        return rand_grid;
    
    def _init_layers(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels;
                m.weight.data.normal_(0,math.sqrt(2./n));
            elif isinstance(m,nn.Conv1d):
                m.weight.data.normal_(0.0,0.02);
            elif isinstance(m,nn.BatchNorm1d):
                m.weight.data.normal_(1.0,0.02);
                m.bias.data.fill_(0)
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1);
                m.bias.data.zero_();