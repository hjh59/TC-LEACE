"""User-fixed label-count eligibility rules for SP27; no model fitting."""
import numpy as np

def eligibility(cells,dataset):
    required=('A_train','A_val','B_train','B_val','test')
    if dataset not in ('celeba','candid'):raise ValueError('unknown dataset')
    for name in required:
        if name not in cells or np.asarray(cells[name]).shape!=(2,2):raise ValueError('four W-by-Z cells required per partition')
    reasons={name:[] for name in required};a=np.asarray(cells['A_train']);av=np.asarray(cells['A_val']);bt=np.asarray(cells['B_train']);bv=np.asarray(cells['B_val']);te=np.asarray(cells['test'])
    if a.min()<50:reasons['A_train'].append('within_W_Z_positive_or_negative_lt50')
    if dataset=='celeba' and bt.sum(axis=0).min()<30:reasons['B_train'].append('Z_positive_or_negative_lt30')
    if dataset=='celeba' and av.sum(axis=1).min()<20:reasons['A_val'].append('W_positive_or_negative_lt20')
    if dataset=='celeba' and bv.sum(axis=0).min()<20:reasons['B_val'].append('Z_positive_or_negative_lt20')
    if dataset=='celeba':
        if te.min()<100:reasons['test'].append('test_four_cell_lt100')
        conditional=bool(te.min()>=30);t5=bool(te.min()>=100)
    else:
        conditional=bool(te.min()>=20);t5=bool(te.min()>=20)
        if not conditional:reasons['test'].append('test_conditional_or_T5_cell_lt20')
    training=not any(reasons[name] for name in required if name!='test')
    return dict(partition_reasons=reasons,training_eligible=training,test_eligible=not reasons['test'],conditional_eligible=conditional,T5_eligible=t5,stop_required=bool(dataset=='candid' and a.min()<50),fold_eligible=bool(training and not reasons['test']))
