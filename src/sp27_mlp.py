"""SP27 frozen MLP training implementation (copied without legacy runners)."""
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
def auc(y,s):return float(roc_auc_score(y,s)) if np.unique(y).size==2 and np.isfinite(s).all() else float('nan')
class DenseMLP(nn.Module):
    def __init__(self,d,hidden,dropout):
        super().__init__();layers=[];prev=d
        for h in hidden:layers.extend([nn.Linear(prev,h),nn.ReLU(),nn.Dropout(dropout)]);prev=h
        layers.append(nn.Linear(prev,1));self.net=nn.Sequential(*layers)
    def forward(self,x):return self.net(x).squeeze(-1)

def fit_mlp(x,y,xv,yv,hidden,drop,lr,wd,seed,epochs=200,patience=15,return_best_epoch=False):
    torch.manual_seed(int(seed));device=x.device
    model=DenseMLP(x.shape[1],hidden,drop).to(device)
    y=y.float();yv=yv.float();n1=y.sum().clamp_min(1);n0=(len(y)-y.sum()).clamp_min(1)
    weights=torch.where(y>.5,len(y)/(2*n1),len(y)/(2*n0))
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    best=-np.inf;best_loss=float('inf');best_state=None;best_epoch=0;stale=0;batch=2048
    for epoch in range(1,epochs+1):
        model.train()
        for ids in torch.randperm(len(y),device=device).split(batch):
            opt.zero_grad();logits=model(x[ids]);loss=(nn.functional.binary_cross_entropy_with_logits(logits,y[ids],reduction='none')*weights[ids]).mean();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            vp=torch.sigmoid(model(xv)).detach().cpu().numpy();vl=float(nn.functional.binary_cross_entropy_with_logits(model(xv),yv).cpu())
        va=auc(yv.cpu().numpy(),vp)
        if va>best+1e-5 or (abs(va-best)<=1e-5 and vl<best_loss):
            best=va;best_loss=vl;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};best_epoch=epoch;stale=0
        else:stale+=1
        if stale>=patience:break
    if best_state:model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        vs=torch.sigmoid(model(xv)).cpu().numpy()
    return model,best,vs,(best_epoch if return_best_epoch else epoch)

def mlp_predict(model,x):
    model.eval()
    with torch.no_grad():return torch.sigmoid(model(x)).detach().cpu().numpy()
