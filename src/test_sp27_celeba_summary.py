import numpy as np
from finalize_sp27_celeba import leave_pair_out,slope,p_one,chunk_valid

def main():
 x=np.array([0.,1.,2.,3.,4.]);y=np.array([.1,.6,.7,1.9,1.6]);ids=np.arange(5);cv=leave_pair_out(x,y,ids)
 for i in range(5):
  mask=ids!=i;fit=np.linalg.lstsq(np.c_[np.ones(mask.sum()),x[mask]],y[mask],rcond=None)[0];assert abs(cv[i]-(fit[0]+fit[1]*x[i]))<1e-12
 # Duplicated sampled pairs must ALL be removed from each training fold.
 take=np.array([0,0,1,2,2,3,4]);xx=x[take];yy=y[take];ii=ids[take];cv=leave_pair_out(xx,yy,ii)
 for k in range(len(take)):
  mask=ii!=ii[k];fit=np.linalg.lstsq(np.c_[np.ones(mask.sum()),xx[mask]],yy[mask],rcond=None)[0];assert abs(cv[k]-(fit[0]+fit[1]*xx[k]))<1e-12
 assert abs(slope(x,2*x+3)-2)<1e-12 and np.isnan(slope(np.ones(3),np.arange(3)))
 assert p_one(np.full(1000,.8),.8,.6,True)==1/1001
 print('PASS: leave-one-pair-out excludes all duplicates; slope; centered-bootstrap p-value floor')
if __name__=='__main__':main()
