import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as A
from scipy.stats import rankdata
from folds import load
SH=['fs50','fs25','jitter10','jitter30','adc4g12b','adc2g10b','adc2g8b','noise_lo','noise_hi','rot15','rot45','rot180','mirror','nogyro']
def rn(d,c):
    p=d[c].values; f=d.fold.values; r=np.zeros(len(p))
    for k in range(5): m=f==k; r[m]=(rankdata(p[m])-0.5)/m.sum()
    return r
def run(setting,r,s,B=2000,norm=True):
    d=load(r,s); y=d.y.values
    S={c:(rn(d,c) if norm else d[c].values) for c in d.columns if '|' in c}
    def auc(idx,c): return A(y[idx],S[c][idx])
    def sm(idx,v): return np.mean([auc(idx,f'{sh}|{v}') for sh in SH])
    def stat(idx):
        h1=auc(idx,'clean|ptq')-auc(idx,'clean|fp32')
        h2=(auc(idx,'clean|ptq')-sm(idx,'ptq'))-(auc(idx,'clean|fp32')-sm(idx,'fp32'))
        h3=sm(idx,'qat_sr')-sm(idx,'ptq')
        return h1,h2,h3
    full=np.arange(len(y)); est=stat(full)
    rng=np.random.default_rng(0); pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; bs=[]
    for b in range(B):
        idx=np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))]; bs.append(stat(idx))
    bs=np.array(bs)
    lo,hi=np.percentile(bs,[2.5,97.5],axis=0)
    p1=np.mean(bs[:,0]<=-0.02)
    p2=2*min(np.mean(bs[:,1]<=0),np.mean(bs[:,1]>=0)); p3=2*min(np.mean(bs[:,2]<=0),np.mean(bs[:,2]>=0))
    ps=np.array([p1,p2,p3]); o=np.argsort(ps); ph=np.empty(3); run_max=0
    for i,j in enumerate(o): run_max=max(run_max,min(1,(3-i)*ps[j])); ph[j]=run_max
    print(setting,s,'norm' if norm else 'pooled',[f'{e:+.3f} ({l:+.3f},{h:+.3f})' for e,l,h in zip(est,lo,hi)],'p raw',round(p1,4),round(p2,4),round(p3,4),'holm',np.round(ph,4))
import sys
if __name__=='__main__':
    st=sys.argv[1]
    for s,r in enumerate(sys.argv[2:]):
        for norm in (False,True): run(st,r,s,B=2000,norm=norm)
