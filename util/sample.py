import numpy as np;
from functools import partial;

def interppz(pos,prob):
    N = prob.shape[1];
    g = np.linspace(0.0,1.0,N);
    step = 1.0 / (N-1);
    xn = pos[0] / step;
    yn = pos[1] / step;
    x1 = g[int(xn)];
    x1w = pos[0] - x1;
    x2 = g[int(xn)+1];
    x2w = x2 - pos[0];
    y1 = g[int(yn)];
    y1w = pos[1] - y1;
    y2 = g[int(yn)+1];
    y2w = y2 - pos[1];
    p1 = prob[int(yn),int(xn)];
    p2 = prob[int(yn)+1,int(xn)];
    p3 = prob[int(yn),int(xn)+1];
    p4 = prob[int(yn)+1,int(xn)+1];
    p12 = ( p1*x2w + p2*x1w ) / (x1w+x2w);
    p34 = ( p3*x2w + p4*x1w ) / (x1w+x2w);
    return ( p12*y2w + p34*y1w ) / (y1w+y2w);

def pz(z,prob):
    func = partial(interppz,prob=prob);
    return np.apply_along_axis(func,0,z);
    
def getz(N,mu,sigma):
    z = np.random.normal(mu,sigma,size=[2,N]).astype(dtype=np.float32);
    idx = (z[0,:]>=0.0) & (z[0,:]<=1.0) & (z[1,:]>=0.0) & (z[1,:]<=1.0);
    z = z[:,idx];
    return z;
    
def qz(z,mu,sigma):
    qz0 = 1/(np.sqrt(2*np.pi)*sigma**2)*np.exp(-0.5*(z[0,:]-mu)**2/sigma**2);
    qz1 = 1/(np.sqrt(2*np.pi)*sigma**2)*np.exp(-0.5*(z[1,:]-mu)**2/sigma**2);
    qz = qz0*qz1;
    return qz;

def reject_sample(prob,N):
    prob = prob / np.sum(prob);
    sigma = 0.5;
    mu = 0.5;
    z = getz(100*N,mu,sigma);
    qzv = qz(z,mu,sigma);
    pzv = pz(z,prob);
    ratio = pzv / qzv; 
    k = np.max( ratio );
    sample = z[:,pzv>=np.random.uniform(0,k*qzv,size=qzv.shape)];
    while sample.shape[1] < N:
        print('sample:',sample.shape[1]);
        z = getz(10*N,mu,sigma);
        qzv = qz(z,mu,sigma);
        pzv = pz(z,prob);
        ratio = pzv / qzv; 
        k = np.max( ratio );
        ns = z[:,pzv>=np.random.uniform(0,k*qzv,size=qzv.shape)];
        sample = np.column_stack((sample,ns));
    return sample[:,:N];