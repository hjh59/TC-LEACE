"""Exercise figure/CSV exports against their actual table schemas using toy data."""
import tempfile
from pathlib import Path
import pandas as pd
import finalize_sp27_celeba as F

def main():
 with tempfile.TemporaryDirectory() as d:
  root=Path(d);F.OUT=root/'out';F.BASE=root/'base';F.OUT.mkdir();F.BASE.mkdir();mainrows=[];elig=[];summary=[];pair=[]
  for p in (0,1):
   for f in range(5):
    for z in (0,1):elig.append(dict(partition='A_train',pair_id=p,fold=f,Z_value=z,n_images=100))
  pd.DataFrame(elig).to_csv(F.BASE/'ELIGIBILITY_TABLE.csv',index=False)
  for model in F.MODELS:
   for p in (0,1):
    for f in range(5):mainrows.append(dict(model=model,pair_id=p,fold=f,method='leace',A_insample_linear_AUROC=.5))
    for k,v in [('abs_beta',.1+p*.2),('unconditional_minus_conditional_suite',.01+p*.02)]:pair.append(dict(model=model,pair_id=p,method='leace',metric=k,estimate=v))
   metrics=['leace__unconditional_linear','leace__unconditional_MLP','leace__slope','leace__conditional_suite','leace__impute_true','leace__linear_h3_guard_MAE']+[f'leace__{p}_{k}' for p in ('linear','MLP') for k in ('within','cross','total')]+['cc2_minus_leace__linear_cross','cc2_minus_leace__conditional_suite']
   for group in ('all','low','middle','high'):
    for k in metrics:summary.append(dict(model=model,beta_band=group,metric=k,estimate=.1,ci_low=.05,ci_high=.15))
  pd.DataFrame(mainrows).to_csv(F.OUT/'METHOD_FOLD_RESULTS.csv',index=False);F.make_figures(pd.DataFrame(summary),pd.DataFrame(pair),pd.DataFrame())
  expected=['F1_THREE_LAYERS.png','F2_IMPUTATION_SCATTER.png','F3_DAMAGE.png','F4_CC2_MINUS_LEACE.png','TABLE2_STRUCTURAL.csv']
  assert all((F.OUT/name).stat().st_size>0 for name in expected)
 print('PASS: F1–F4 and structural table generated from declared schemas (synthetic data only)')
if __name__=='__main__':main()
