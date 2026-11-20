import numpy as np
from run_sp27_celeba_main import main_payload,split_manifest_fields

def main():
 manifest=dict(seed=123,split_config='A80B20',**split_manifest_fields(dict(seed=20261103,split_config='A80B20',n_A_images=80)))
 assert manifest['seed']==123 and manifest['split_seed']==20261103 and manifest['n_A_images']==80
 scores=dict(sample_id=np.asarray(['a','b','c','d','e']),identity_id=np.arange(5),partition=np.asarray(['A_train','A_val','B_train','B_val','test']),score=np.array([.1,.2,.3,.4,.5]));ix=np.array([1,3,4]);out=main_payload(scores,ix)
 assert all(v.shape==(3,) for v in out.values()) and np.array_equal(out['score'],scores['score'][ix])
 scores['identity_id']=np.array(['a','b','c','d','e'],dtype=object)
 try:main_payload(scores,ix)
 except RuntimeError:pass
 else:raise AssertionError('object metadata accepted')
 print('PASS: only validation/test rows exported; numeric scores unchanged; object metadata rejected')
if __name__=='__main__':main()
