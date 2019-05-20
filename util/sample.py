from numpy import np;

def pz(z,prob):
    return p;
    
def getz(N):
    z = np.random.normal(mu,sigma,size=[2,N],dtype=np.float32);
    idx = z[0,:]>=0.0 and z[0,:]<=1.0 and z[1,:]>=0.0 and z[1,:]<=1.0;
    z = z[:,idx];
    return z;
    
def qz(z,sigma,mu)
    qz0 = 1/(np.sqrt(2*np.pi)*sigma**2)*np.exp(-0.5*(z[0,:]-mu)**2/sigma**2);
    qz1 = 1/(np.sqrt(2*np.pi)*sigma**2)*np.exp(-0.5*(z[1,:]-mu)**2/sigma**2);
    qz = qz0*qz1;
    return qz;

def reject_sample(prob,N):
    b = prob.shape[0];
    npatch = prob.shape[1];
    sigma = 0.5;
    mu = 0.5;
    ret = np.zeros([b,npatch,2,N],dtype=np.float32);
    z = getz(50*N);
    qzv = qz(z,sigma,mu);
    for bi in range(b):
        for pi in range(npatch):
            pzv = pz(z,prob[bi,pi,...]);
            ratio = qzv / pzv; 
            k = 1.0 / np.min( ratio );
            sample = z[:,np.random.uniform(0.1,1.0,size=ratio.shape,dtype=np.float32)>k*ratio];
            while sample.shape[1] < N:
                z = getz(50*N);
                qzv = qz(z,sigma,mu);
                pzv = pz(z,prob[bi,pi,...]);
                ratio = qzv / pzv; 
                k = 1.0 / np.min( ratio );
                ns = z[:,np.random.uniform(0.1,1.0,size=ratio.shape,dtype=np.float32)>k*ratio];
                sample = np.column_stack((sample,ns));
            ret[bi,pi,...] = sample[:,:N];
    return ret;