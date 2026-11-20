"""Label-blind CelebA representation extraction, pinned HF models only.

Reads Parquet image bytes and image paths only. It does not select or load
attribute columns. The pre-frozen identity manifest determines output order.
"""
from __future__ import annotations
import argparse,hashlib,io,json,time,subprocess
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoImageProcessor,AutoModel

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('${DATA_ROOT}/Dataset/CelebA')
PARQUET=Path('${DATA_ROOT}/Dataset/CelebA_HF/img_align+identity+attr')
SPLIT=DATA/'analysis_manifest/identity_split_seed20261015.tsv'
CACHE=ROOT/'results/case_studies/celeba/model_cache'
OUT=ROOT/'results/case_studies/celeba/representations'
SEED=20261015; BATCH=64
MODELS={
 'clip':{'repo':'openai/clip-vit-base-patch32','revision':'3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268','dir':'clip-vit-base-patch32','dim':768,'feature':'vision_model.pooler_output before visual projection'},
 'dinov2':{'repo':'facebook/dinov2-base','revision':'f9e44c814b77203eaa57a6bdbbd535f21ede1415','dir':None,'dim':768,'feature':'final CLS token at official 518x518 input'},
 'siglip':{'repo':'google/siglip-base-patch16-224','revision':'7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed','dir':'siglip-base-patch16-224','dim':768,'feature':'vision_model.pooler_output before visual projection'},
 'mae':{'repo':'facebook/vit-mae-base','revision':'25b184bea5538bf5c4c852c79d221195fdd2778d','dir':'vit-mae-base','dim':768,'feature':'final CLS token; mask_ratio=0 to encode all patches'} }
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,required=True);ap.add_argument('--gpu',type=int,default=0);args=ap.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 torch.cuda.set_device(args.gpu);spec=MODELS[args.model]
 import pandas as pd
 order=pd.read_csv(SPLIT,sep='\t',usecols=['image_filename'])
 if len(order)!=202599 or order.image_filename.nunique()!=202599:raise RuntimeError('identity split manifest mismatch')
 keys=[x.rsplit('.',1)[0] for x in order.image_filename];index={k:i for i,k in enumerate(keys)}
 model_dir=(ROOT/'results/stage2/model_cache/dinov2-base') if args.model=='dinov2' else CACHE/spec['dir']
 processor=AutoImageProcessor.from_pretrained(model_dir,local_files_only=True)
 if args.model=='dinov2':processor.size={'height':518,'width':518};processor.crop_size={'height':518,'width':518}
 model=AutoModel.from_pretrained(model_dir,local_files_only=True).eval().cuda()
 if args.model=='mae': model.config.mask_ratio=0.0
 def embed(images):
  pixels=processor(images=images,return_tensors='pt').pixel_values.cuda(non_blocking=True)
  with torch.inference_mode():
   if args.model in ('clip','siglip'):out=model.vision_model(pixel_values=pixels).pooler_output
   else:out=model(pixel_values=pixels).last_hidden_state[:,0,:]
  return out.float().cpu().numpy()
 dest=OUT/args.model;dest.mkdir(parents=True,exist_ok=True)
 feat_path=dest/'features.npy';state_path=dest/'shards_done.json'
 if feat_path.exists() and not state_path.exists():raise RuntimeError('incomplete untracked feature file exists; remove it before starting')
 if feat_path.exists():
  x=np.lib.format.open_memmap(feat_path,mode='r+',dtype=np.float32,shape=(len(order),spec['dim']))
  shards_done=json.loads(state_path.read_text())['shards_completed'];done={q['file'] for q in shards_done}
 else:
  x=np.lib.format.open_memmap(feat_path,mode='w+',dtype=np.float32,shape=(len(order),spec['dim']));shards_done=[];done=set()
 seen=np.zeros(len(order),dtype=np.bool_);started=time.perf_counter()
 for p in sorted(PARQUET.glob('*.parquet')):
  pf=pq.ParquetFile(p);shard_n=0;xs=[];ids=[]
  if p.name in done:
   for rb in pf.iter_batches(columns=['image.path'],batch_size=8192):
    for ent in rb.column(0).field('path').to_pylist():
     key=ent.rsplit('/',1)[-1].rsplit('.',1)[0];i=index[key]
     if seen[i]:raise RuntimeError(f'duplicate completed image {key}')
     seen[i]=True
   continue
  for rb in pf.iter_batches(columns=['image.path','image.bytes'],batch_size=BATCH):
   image_col=rb.column(0);col=image_col.field('path').to_pylist();raw=image_col.field('bytes').to_pylist()
   for path,blob in zip(col,raw):
    key=path.rsplit('/',1)[-1].rsplit('.',1)[0]
    if key not in index:raise RuntimeError(f'unknown image path {path}')
    i=index[key]
    if seen[i]:raise RuntimeError(f'duplicate image key {key}')
    with Image.open(io.BytesIO(blob)) as im: xs.append(im.convert('RGB'))
    ids.append(i);seen[i]=True;shard_n+=1
    if len(xs)>=BATCH:
     v=embed(xs)
     if v.shape!=(len(ids),spec['dim']) or not np.isfinite(v).all():raise RuntimeError('invalid embedding batch')
     x[np.asarray(ids,dtype=np.int64)]=v;xs=[];ids=[]
  if xs:
   v=embed(xs)
   if v.shape!=(len(ids),spec['dim']) or not np.isfinite(v).all():raise RuntimeError('invalid embedding batch')
   x[np.asarray(ids,dtype=np.int64)]=v
  x.flush();shards_done.append({'file':p.name,'images':shard_n})
  state_path.write_text(json.dumps({'shards_completed':shards_done},indent=2)+'\n')
  print(f'{args.model}: {p.name} {shard_n} images',flush=True)
 if not seen.all():raise RuntimeError(f'missing {int((~seen).sum())} images')
 del x;x=np.load(feat_path,mmap_mode='r')
 rng=np.random.default_rng(SEED);sample=set(rng.choice(len(order),size=500,replace=False).tolist());sims=[]
 # Re-scan path and image payloads for a 500-image deterministic repeatability check.
 images=[];ids=[]
 for p in sorted(PARQUET.glob('*.parquet')):
  pf=pq.ParquetFile(p)
  for rb in pf.iter_batches(columns=['image.path','image.bytes'],batch_size=BATCH):
   image_col=rb.column(0);paths=image_col.field('path').to_pylist();blobs=image_col.field('bytes').to_pylist()
   for path,blob in zip(paths,blobs):
    key=path.rsplit('/',1)[-1].rsplit('.',1)[0];i=index[key]
    if i not in sample:continue
    with Image.open(io.BytesIO(blob)) as im:images.append(im.convert('RGB'))
    ids.append(i)
    if len(images)==BATCH or (len(ids)==len(sample)):
     v=embed(images)
     for i0,v0 in zip(ids,v):
      a=np.asarray(x[i0],dtype=np.float64);b=np.asarray(v0,dtype=np.float64)
      sims.append(float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))))
     images=[];ids=[]
    if len(sims)==500:break
   if len(sims)==500:break
  if len(sims)==500:break
 if images:
  v=embed(images)
  for i0,v0 in zip(ids,v):
   a=np.asarray(x[i0],dtype=np.float64);b=np.asarray(v0,dtype=np.float64)
   sims.append(float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))))
 if len(sims)!=500:raise RuntimeError(f'repeatability sample mismatch: {len(sims)}')
 weights={str(f.relative_to(model_dir)):sha(f) for f in model_dir.rglob('*') if f.is_file() and f.suffix in ('.bin','.safetensors')}
 meta={'model':spec['repo'],'revision':spec['revision'],'feature':spec['feature'],'input_size':getattr(processor,'size',None),'images':len(x),'dimension':spec['dim'],'dtype':'float32','nan':int(np.isnan(x).sum()),'inf':int(np.isinf(x).sum()),'repeatability_n':len(sims),'repeatability_cosine_median':float(np.median(sims)),'repeatability_pass':bool(np.median(sims)>=.999),'identity_split_manifest_sha256':sha(SPLIT),'checkpoint_sha256':weights,'git_head':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'seconds':time.perf_counter()-started,'scope':'all aligned images; image/path columns only; no attribute columns read'}
 (dest/'qc.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__':main()
