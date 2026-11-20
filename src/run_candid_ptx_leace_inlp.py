from pathlib import Path
import hashlib,json,os,time
import numpy as np
import pandas as pd
import torch
from PIL import Image

OUT=Path('./results/case_studies/candid_ptx')


MODEL=Path('${DATA_ROOT}/experiments/joint_predictability_gate/model_cache')


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def extract(worker=0,workers=1,batch=32,chunk_n=128):
    from transformers import AutoImageProcessor,AutoModel
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(OUT/'manifest.csv')
    device=torch.device(f'cuda:{worker%torch.cuda.device_count()}')
    torch.cuda.set_device(device);torch.set_num_threads(1)
    processor=AutoImageProcessor.from_pretrained(MODEL,local_files_only=True)
    model=AutoModel.from_pretrained(MODEL,local_files_only=True).eval().to(device)
    for p in model.parameters():p.requires_grad_(False)
    chunks=OUT/'chunks';chunks.mkdir(exist_ok=True)
    n=len(manifest);nc=(n+chunk_n-1)//chunk_n;started=time.time()
    for ci in range(nc):
        if ci%workers!=worker:continue
        path=chunks/f'chunk_{ci:04d}.npy';lo=ci*chunk_n;hi=min(n,lo+chunk_n)
        if path.exists():
            z=np.load(path,mmap_mode='r')
            if z.shape==(hi-lo,768) and z.dtype==np.float32 and np.isfinite(z).all():continue
            path.unlink()
        vec=[]
        with torch.inference_mode():
            for start in range(lo,hi,batch):
                ims=[]
                for q in manifest.image_path.iloc[start:min(start+batch,hi)]:
                    with Image.open(q) as im:ims.append(im.convert('RGB'))
                px=processor(images=ims,return_tensors='pt',size={'shortest_edge':224},crop_size={'height':224,'width':224}).pixel_values.to(device)
                out=model(pixel_values=px).last_hidden_state[:,0,:].float().cpu().numpy()
                vec.extend(out)
        arr=np.asarray(vec,dtype=np.float32)
        if arr.shape!=(hi-lo,768) or not np.isfinite(arr).all():raise RuntimeError(f'Bad chunk {ci}: {arr.shape}')
        tmp=path.with_suffix('.tmp.npy');np.save(tmp,arr);os.replace(tmp,path)
        print(f'GPU {device} finished {ci+1}/{nc}, rows {hi}/{n}',flush=True)
    torch.cuda.synchronize(device)
    print(f'worker={worker}/{workers} elapsed_min={(time.time()-started)/60:.1f}',flush=True)


def consolidate():
    m=pd.read_csv(OUT/'manifest.csv');n=len(m);chunk_n=128;nc=(n+chunk_n-1)//chunk_n
    x=np.lib.format.open_memmap(OUT/'embeddings.npy',mode='w+',dtype=np.float32,shape=(n,768))
    for ci in range(nc):
        q=np.load(OUT/'chunks'/f'chunk_{ci:04d}.npy',mmap_mode='r');lo=ci*chunk_n;x[lo:lo+len(q)]=q
    x.flush();del x
    x=np.load(OUT/'embeddings.npy',mmap_mode='r')
    qc={'n':n,'dim':768,'dtype':'float32','nan':int(np.isnan(x).sum()),'inf':int(np.isinf(x).sum()),
        'norm_median':float(np.median(np.linalg.norm(x,axis=1))),'model':'microsoft/rad-dino',
        'model_weights_sha256':sha256(MODEL/'model.safetensors'),'input_size':224,
        'features':'last_hidden_state CLS token','embedding_sha256':sha256(OUT/'embeddings.npy')}
    if qc['nan'] or qc['inf']:raise RuntimeError('Non-finite embeddings')
    (OUT/'embedding_qc.json').write_text(json.dumps(qc,indent=2)+'\n');print(json.dumps(qc,indent=2))

