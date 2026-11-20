
"""POST HOC – not preregistered.
SPLINCE: independent implementation of Holstege et al. theorem1, Eq4.
No external implementation was fetched or copied. Row data apply column P.T.
Frozen covariance convention Cov(X)+1e-6 I is retained for whitening.
"""
import numpy as np

def covariance(x,z):
 x=np.asarray(x,np.float64);z=np.asarray(z,np.float64)
 return np.matmul((x-x.mean(0)).T,(z-z.mean()))/len(z)

def canonical_columns(q):
 q=np.asarray(q,np.float64).copy()
 for j in range(q.shape[1]):
  pivot=int(np.argmax(abs(q[:,j])))
  if q[pivot,j]<0:q[:,j]*=-1
 return q

def fit_splince(x,z,w):
 x=np.asarray(x,np.float64);mu=x.mean(0);xc=x-mu;cov=np.matmul(xc.T,xc)/len(x)+1e-6*np.eye(x.shape[1]);e,v=np.linalg.eigh((cov+cov.T)/2)
 if e.min()<=0:raise RuntimeError('SPLINCE_COVARIANCE_NOT_POSITIVE')
 wh=np.matmul(v/np.sqrt(e),v.T);whplus=np.matmul(v*np.sqrt(e),v.T);cz=covariance(x,z);cy=covariance(x,w);a=np.matmul(wh,cz);b=np.matmul(wh,cy)
 if np.linalg.norm(a)==0 or np.linalg.norm(b)==0 or np.linalg.matrix_rank(np.column_stack((a,b)))!=2:raise RuntimeError('SPLINCE_THEOREM1_ASSUMPTION_FAILED')
 qa,_=np.linalg.qr(a[:,None],mode='complete');U=canonical_columns(qa[:,1:]);qab,_=np.linalg.qr(np.column_stack((a,b)),mode='complete');Uminus=canonical_columns(qab[:,2:]);V,_=np.linalg.qr(np.column_stack((b,Uminus)),mode='reduced');V=canonical_columns(V)
 P=np.matmul(np.matmul(whplus,V),np.linalg.solve(np.matmul(U.T,V),np.matmul(U.T,wh)))
 return dict(mean=mu,P=P,projection_rank=int(np.linalg.matrix_rank(P)),removed_rank=int(x.shape[1]-np.linalg.matrix_rank(P)),U_V_condition=float(np.linalg.cond(np.matmul(U.T,V))),dtype=str(P.dtype))

def transform_splince(x,fit):return np.matmul((np.asarray(x,np.float64)-fit['mean']),fit['P'].T)+fit['mean']

def fit_weber(x,z):
 x=np.asarray(x,np.float64);z=np.asarray(z,np.float64);p=z.mean();variance=float(np.mean((z-p)**2))
 if variance<=0:raise RuntimeError('WEBER_CONSTANT_TARGET')
 return dict(beta=covariance(x,z)/variance,p=float(p),requires_true_Z_at_application=True,dtype='float64')

def transform_weber(x,z,fit):return np.asarray(x,np.float64)-np.outer(np.asarray(z,np.float64)-fit['p'],fit['beta'])
