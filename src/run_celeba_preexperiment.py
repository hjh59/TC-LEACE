from pathlib import Path
import os
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


DATA=Path(os.environ.get('CELEBA_DATA_ROOT','${DATA_ROOT}/Dataset/CelebA'))


MANIFEST=DATA/'analysis_manifest/identity_split_seed20261015.tsv'


ANNOT=DATA/'Anno/list_attr_celeba.txt'


BASE=ROOT/'results/case_studies/celeba'


def pre_labels(manifest, split='pre_experiment'):
 """Read attributes only for the requested identity partition.

 The default remains the pre-experiment partition. Confirmatory callers must
 pass the explicit partition only after the T13 hash gate has accepted it.
 """
 if split not in ('pre_experiment','confirmatory'):raise ValueError(f'unknown CelebA split: {split}')
 pre=manifest.split.eq(split).to_numpy();rows=manifest.loc[pre]
 index={name:i for i,name in enumerate(rows.image_filename)}
 with ANNOT.open() as f:
  expected=int(f.readline().strip());names=f.readline().split()
  if expected!=len(manifest) or len(names)!=40:raise RuntimeError('CelebA annotation header mismatch')
  y=np.full((len(rows),40),-1,dtype=np.int8);found=0
  for line in f:
   parts=line.split()
   i=index.get(parts[0])
   if i is None:continue
   if len(parts)!=41:raise RuntimeError('malformed attribute row')
   y[i]=(np.asarray(parts[1:],dtype=np.int8)==1).astype(np.int8);found+=1
 if found!=len(rows):raise RuntimeError(f'pre-experiment annotation rows found={found}, expected={len(rows)}')
 return rows.reset_index(drop=True),names,y

