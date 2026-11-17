import pandas as pd, numpy as np, sys, ast
from scipy import stats
fs=sys.argv[1:]
D=[]
for i,f in enumerate(fs):
    d=pd.read_csv(f); d['seed']=i; D.append(d)
d=pd.concat(D)
fp=d[d.variant=='fp32'].set_index(['seed','rep']).auroc
d['loss']=d.auroc.values-fp.loc[list(zip(d.seed,d.rep))].values
def fam(v):
    for p in ['ptqd300','ptqd2k','ptqc300','ptqc2k']:
        if v.startswith(p+'_'): return p
    return v
d['fam']=d.variant.map(fam)
print("FP32 AUROC range",fp.min().round(3),fp.max().round(3), "n models",len(fp))
for f in ['ptq','ptqd300','ptqd2k','ptqc300','ptqc2k','ptq_a16','ptq_cle','qat']:
    x=d[d.fam==f].loss
    print(f"{f:8s} n={len(x):3d} mean={x.mean():+.3f} sd={x.std():.3f} >0.05={np.mean(x<-0.05)*100:4.0f}% worst={x.min():+.3f} best={x.max():+.3f}")
# paired clip vs minmax same draw
for a,b in [('ptqd300','ptqc300'),('ptqd2k','ptqc2k')]:
    A=d[d.fam==a].copy(); B=d[d.fam==b].copy()
    A['k']=A.variant.str.split('_').str[1]; B['k']=B.variant.str.split('_').str[1]
    m=A.merge(B,on=['seed','rep','k'],suffixes=('_m','_c'))
    df=m.loss_c-m.loss_m
    # model-level mean to avoid pseudo-replication
    mm=m.groupby(['seed','rep']).apply(lambda g:(g.loss_c-g.loss_m).mean())
    print(f"{b}-{a}: draw-level mean {df.mean():+.3f}; model-level mean {mm.mean():+.3f} (models better {np.sum(mm>0)}/{len(mm)}), wilcoxon p={stats.wilcoxon(mm).pvalue:.3f}")
    print("   in_scale minmax median",m.in_scale_m.median().round(3),"clip",m.in_scale_c.median().round(3),"in_sat minmax",(m.in_sat_m.mean()*100).round(4),"% clip",(m.in_sat_c.mean()*100).round(3),"%")
# loss vs in_scale for minmax draws
for f in ['ptqd300','ptqd2k','ptqc300','ptqc2k']:
    x=d[d.fam==f]
    r=stats.spearmanr(x.in_scale,x.loss); r2=stats.spearmanr(x.in_sat,x.loss); r3=stats.spearmanr(x.out_scale,x.loss)
    # within-model correlation (demeaned)
    xm=x.copy(); 
    for c in ['in_scale','loss','out_scale']: xm[c]=xm[c]-xm.groupby(['seed','rep'])[c].transform('mean')
    rw=stats.spearmanr(xm.in_scale,xm.loss); ro=stats.spearmanr(xm.out_scale,xm.loss)
    print(f"{f}: rho(loss,in_scale)={r.correlation:+.2f} p={r.pvalue:.3f}; within-model {rw.correlation:+.2f} p={rw.pvalue:.3f}; rho(loss,in_sat)={r2.correlation:+.2f}; out_scale within {ro.correlation:+.2f} p={ro.pvalue:.3f}; in_scale range {x.in_scale.min():.3f}-{x.in_scale.max():.3f}, in_sat max {x.in_sat.max()*100:.3f}%")
