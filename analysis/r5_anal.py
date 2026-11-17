import pandas as pd,numpy as np,json,sys,re
from scipy import stats
D=[]
for i,f in enumerate(sys.argv[1:]):
    d=pd.read_csv(f); d['seed']=i; D.append(d)
d=pd.concat(D)
fp=d[d.variant=='fp32'].set_index(['seed','rep'])
d['loss']=d.auroc.values-fp.auroc.loc[list(zip(d.seed,d.rep))].values
d['wloss']=d.win_auc.values-fp.win_auc.loc[list(zip(d.seed,d.rep))].values
d['fam']=d.variant.str.replace(r'_\d+$','',regex=True)
print('FP32',fp.auroc.round(3).tolist())
for f in ['ptq','ptqd300','ptqc300','ptqcb300','ptqd2k','ptqc2k','ptqcb2k','ptq_a16','qat','ptq_cle']:
    x=d[d.fam==f]
    if not len(x): continue
    print(f"{f:9s} n={len(x):3d} loss {x.loss.mean():+.3f} >0.05 {np.mean(x.loss<-0.05)*100:4.1f}% worst {x.loss.min():+.3f} | winAUC loss {x.wloss.mean():+.3f} worst {x.wloss.min():+.3f} | rmse {x.logit_rmse.mean():.3f} agree {x.win_agree.mean():.3f}")
q=d[d.fam.str.match(r'ptq[dc]')].copy()
def dm(c): return q[c]-q.groupby(['seed','rep'])[c].transform('mean')
for c in ['logit_rmse','win_agree','wloss']:
    r=stats.spearmanr(q[c],q.loss); rw=stats.spearmanr(dm(c),dm('loss'))
    print(f"{c}: rho {r.correlation:+.2f}  within-model {rw.correlation:+.2f} p={rw.pvalue:.1e}")
# per-layer scales
A=pd.DataFrame([json.loads(s) for s in q.act_scales]); A.index=q.index
A.columns=[re.sub(r'^\d+:','',c) for c in A.columns] if A.columns.is_unique else A.columns
A=np.log(A)
mm=q.fam.str.startswith('ptqd')
res=[]
for c in A.columns:
    x=A[c]; xm=x-x.groupby([q.seed,q.rep]).transform('mean'); lm=dm('loss')
    sub=mm
    rw=stats.spearmanr(xm[sub],lm[sub]).correlation
    ratio=np.exp(A.loc[q.fam.str.startswith('ptqd'),c].mean()-A.loc[q.fam.str.startswith('ptqc'),c].mean())
    cv=np.exp(x[sub]).groupby([q.seed[sub],q.rep[sub]]).agg(lambda v:v.std()/v.mean()).mean()
    res.append((c,rw,ratio,cv))
R=pd.DataFrame(res,columns=['tensor','rho_within_minmax','minmax/clip','cv_minmax']).sort_values('rho_within_minmax')
pd.set_option('display.width',200); pd.set_option('display.max_colwidth',45)
print(R.to_string(index=False))
