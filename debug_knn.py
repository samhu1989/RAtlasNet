import sys;
import torch;
sys.path.append("./ext/");
import dist_chamfer as ext;
b = 2
N = 5;
k = 3;
xyz = torch.Tensor(b,N,3).uniform_(-1,1);
print(xyz);
xyz = xyz.cuda();
idx = ext.knn(xyz,k,debug=True);
xyz1 = xyz.view(xyz.size(0),1,xyz.size(1),xyz.size(2));
xyz2 = xyz.view(xyz.size(0),xyz.size(1),1,xyz.size(2));
dist = torch.sum((xyz1-xyz2)**2,3);
dst,idx = torch.topk(dist,k+1,dim=2,largest=False,sorted=True);
print(dst.cpu().numpy());
print(idx.cpu().numpy());