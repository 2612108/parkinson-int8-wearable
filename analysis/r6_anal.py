import pandas as pd, numpy as np, sys
from scipy import stats
base=sys.argv[1]; runs=sys.argv[2:]
D=[]
for r in runs:
    d=pd.read_csv(f'{base}/{r}/shift_results.csv'); d=d[d['shift']=='clean'].copy(); d['run']=r
    fp=d[d.variant=='fp32'].set_index('fold').auroc; d['loss']=d.auroc-d.fold.map(fp)
    d['fam']=d.variant.str.replace(r'_\d+$','',regex=True); D.append(d)
D=pd.concat(D); print('models',D.groupby('run').fold.nunique().sum())
names={'ptqd':'min/max','ptqc':'input clip 0.1/99.9','ptqcb':'input clip 1/99','ptqp':'per-tensor pct 99.99','ptqm':'per-tensor MSE','ptqk':'per-tensor KL'}
for s in ['300','2k']:
    for k,v in names.items():
        x=D[D.fam==k+s]
        if not len(x): continue
        mw=x.groupby(['run','fold']).loss.min()
        print(f"{s:4s} {v:22s} n={len(x)} mean={x.loss.mean():+.3f} >0.05={np.mean(x.loss<-0.05)*100:4.1f}% worst={x.loss.min():+.3f} BA={x.bal_acc.mean():.3f} agree={x.agree_fp32.mean():.3f}")
    a=D[D.fam=='ptqd'+s].groupby(['run','fold']).loss.mean()
    for k in ['ptqp','ptqm','ptqk','ptqcb']:
        b=D[D.fam==k+s].groupby(['run','fold']).loss.mean()
        if len(b): print('   ',k+s,'vs minmax per-model', round((b-a).mean(),3), 'better',int(((b-a)>0).sum()),'/',len(a), 'p',f"{stats.wilcoxon(b-a).pvalue:.1e}")
    b1=D[D.fam=='ptqcb'+s].groupby(['run','fold']).loss.mean()
    for k in ['ptqm','ptqk','ptqp']:
        b=D[D.fam==k+s].groupby(['run','fold']).loss.mean()
        if len(b): print('    clip1 vs',k, round((b1-b).mean(),3), 'clip1 better',int(((b1-b)>0).sum()),'/',len(b),'p',f"{stats.wilcoxon(b1-b).pvalue:.1e}")
for v in ['fp32','ptq','ptq_a16','qat']:
    x=D[D.variant==v]; print(v, f"{x.auroc.mean():.3f}", f"{x.loss.mean():+.3f}")
