"""POST HOC – not preregistered. Scope/prediction guards; no model fitting."""
from pathlib import Path
import hashlib,json,numpy as np
from sp27_v22_scope import fit_indices,PRED_SHA
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'results/case_studies/candid_ptx/sp27_posthoc_v2';OUT=ROOT/'results/case_studies/candid_ptx/sp27_posthoc_v2_2'
def main():
 parts={k:np.array(v) for k,v in dict(A_train=[0,1,2,3],A_val=[4],B_train=[5,6],B_val=[7],test=[8,9]).items()};assert np.array_equal(fit_indices(parts,10),[0,1,2,3,4,8,9])
 bad=dict(parts,A_val=np.array([4,5]))
 try:fit_indices(bad,10)
 except RuntimeError:pass
 else:raise AssertionError('B-leakage guard failed')
 original=(OLD/'PREDICTIONS_V2.md').read_bytes();new=(OUT/'PREDICTIONS_V2_2.md').read_bytes();assert new==original.replace(b'S-rel',b'S-rel2');assert hashlib.sha256(new).hexdigest()==PRED_SHA
 assert json.loads((OLD/'V2_SLIM_DELIVERED.json').read_text())['verified']
 (OUT/'SCOPE_UNIT_TESTS.json').write_text(json.dumps(dict(analysis_title='POST HOC – not preregistered',passed=True,tests=['exact excluded B membership','invalid membership raises','predictions replacement only','prediction SHA locked','prior slim delivered']),indent=2)+'\n');print('PASS V2.2 membership and prediction guards')
if __name__=='__main__':main()
