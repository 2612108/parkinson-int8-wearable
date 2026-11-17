import pandas as pd, numpy as np, json
from sklearn.metrics import roc_auc_score as A
from scipy.stats import rankdata
from folds import load, P, lab
age=dict(zip(P.id.astype(int),P.age)); sex=dict(zip(P.id.astype(int),P.gender))
out={}
def stats(d,col):
    y=d.y.values; p=d[col].values; f=d.fold.values
    pooled=A(y,p)
    rn=np.zeros(len(p))
    for k in range(5):
        m=f==k; rn[m]=(rankdata(p[m])-0.5)/m.sum()
    per=[A(y[f==k],p[f==k]) for k in range(5)]
    return pooled,A(y,rn),np.mean(per),np.std(per,ddof=1),min(per),max(per),rn
res=[]
runs=[('zero-phase','p56a',0),('zero-phase','p56s1',1),('zero-phase','p56s2',2),('causal','p56c0',0),('causal','p56c1',1),('causal','p56c2',2)]
R={}
for setting,r,s in runs:
    d=load(r,s)
    if 'fold' not in d.columns: raise SystemExit('nofold '+r)
    R[(setting,s)]=d
    for v in ['fp32','ptq','qat','qat_sr','ptq_a16','ptq_c2k','fp32_ft','ptq_ft','ptq_sr','fp32_sr']:
        c='clean|'+v
        if c not in d.columns: continue
        po,prn,mw,sw,mn,mx,_=stats(d,c)
        res.append(dict(setting=setting,seed=s,variant=v,pooled=round(po,3),pooled_ranknorm=round(prn,3),within_mean=round(mw,3),within_sd=round(sw,3),within_min=round(mn,3),within_max=round(mx,3)))
T=pd.DataFrame(res); pd.set_option('display.width',200); print(T.to_string(index=False))
T.to_csv('pooling_sensitivity.csv',index=False)
# seed variability of FP32
for st in ['zero-phase','causal']:
    t=T[(T.setting==st)&(T.variant=='fp32')]
    print(st,'FP32 across seeds: pooled',t.pooled.tolist(),'within',t.within_mean.tolist(),'SD within over seeds',round(t.within_mean.std(ddof=1),3))
# HC per fold
d=R[('zero-phase',0)]; print('HC per test fold', d[d.y==0].groupby('fold').size().tolist())
# within-sex AUROC with rank-normalized scores, pooled over seeds as mean
print('--- within sex (rank-normalized pooled), zero-phase')
for v in ['fp32','ptq','qat']:
    for sx in ['female','male']:
        vals=[]
        for s in range(3):
            d=R[('zero-phase',s)]; _,_,_,_,_,_,rn=stats(d,'clean|'+v)
            m=d.id.map(sex).values==sx; vals.append(A(d.y.values[m],rn[m]))
        print(v,sx,np.round(vals,3),'n=',int(m.sum()),'HC',int(((d.y==0).values&m).sum()))
# age+sex vs score+age+sex logistic (within each seed, CV by folds)
from sklearn.linear_model import LogisticRegression
print('--- logistic: score adds over age+sex (zero-phase, fp32/qat), out-of-fold')
for v in ['fp32','qat','ptq']:
  for s in range(3):
    d=R[('zero-phase',s)]; _,_,_,_,_,_,rn=stats(d,'clean|'+v)
    X0=np.c_[d.id.map(age).values, (d.id.map(sex)=='female').values.astype(float)]
    X1=np.c_[X0, rn]; y=d.y.values; f=d.fold.values
    p0=np.zeros(len(y)); p1=np.zeros(len(y))
    for k in range(5):
        tr=f!=k; te=f==k
        p0[te]=LogisticRegression(max_iter=1000).fit(X0[tr],y[tr]).predict_proba(X0[te])[:,1]
        p1[te]=LogisticRegression(max_iter=1000).fit(X1[tr],y[tr]).predict_proba(X1[te])[:,1]
    print(v,s,'age+sex',round(np.mean([A(y[f==k],p0[f==k]) for k in range(5)]),3),'+score',round(np.mean([A(y[f==k],p1[f==k]) for k in range(5)]),3))
