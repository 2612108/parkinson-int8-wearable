import pandas as pd, numpy as np, json, sys
from scipy import stats
base=sys.argv[1]; runs=sys.argv[2:]
fam_re=r'_\d+$'
out=[]; sel=[]
for r in runs:
    d=pd.read_csv(f'{base}/{r}/shift_results.csv'); d=d[d['shift']=='clean'].copy(); d['run']=r
    fp=d[d.variant=='fp32'].set_index('fold').auroc
    d['loss']=d.auroc-d.fold.map(fp)
    d['fam']=d.variant.str.replace(fam_re,'',regex=True)
    out.append(d)
    va={}
    for l in open(f'{base}/{r}/calib.jsonl'):
        j=json.loads(l); va[j['fold']]=j.get('val_auc',{})
    for f,v in va.items():
        dd=d[d.fold==f].set_index('variant')
        for size in ['300','2k']:
            fams={'minmax':f'ptqd{size}','c0.01':f'ptqca{size}','c0.1':f'ptqc{size}','c1':f'ptqcb{size}'}
            # best draw by val among minmax
            mm=[n for n in v if n.startswith(fams['minmax']+'_')]
            if not mm: continue
            bd=max(mm,key=lambda n:v[n])
            # val-chosen clipping level: mean val auc over draws per level
            lv={k:np.mean([v[n] for n in v if n.startswith(fm+'_')]) for k,fm in fams.items() if k!='minmax'}
            kbest=max(lv,key=lv.get)
            # draw-wise: loss of the val-chosen level
            ch=dd[dd.fam==fams[kbest]].loss
            sel.append(dict(run=r,fold=f,size=size,bestdraw_loss=dd.loc[bd,'loss'],chosen=kbest,chosen_mean=ch.mean(),chosen_worst=ch.min(),
                            fp=fp[f]))
D=pd.concat(out); S=pd.DataFrame(sel)
print("models:",D.groupby('run').fold.nunique().sum())
for size in ['300','2k']:
    for lab,fm in [('minmax',f'ptqd{size}'),('clip0.01',f'ptqca{size}'),('clip0.1',f'ptqc{size}'),('clip1',f'ptqcb{size}')]:
        x=D[D.fam==fm].loss; m=D[D.fam==fm].groupby(['run','fold']).loss
        print(f"{size:4s} {lab:9s} n={len(x)} mean={x.mean():+.3f} >0.05={np.mean(x<-0.05)*100:4.1f}% worst={x.min():+.3f} worst-per-model max={-m.min().min():.3f} median-model-worst={-m.min().median():.3f}")
    s=S[S['size']==size]
    print(f"   val-best minmax draw: mean {s.bestdraw_loss.mean():+.3f} >0.05 {np.mean(s.bestdraw_loss<-0.05)*100:.0f}% worst {s.bestdraw_loss.min():+.3f}")
    print(f"   val-chosen clip level counts {s.chosen.value_counts().to_dict()} mean {s.chosen_mean.mean():+.3f} worst {s.chosen_worst.min():+.3f}")
for v in ['ptq','ptq_a16','qat','ptq_cle']:
    x=D[D.variant==v].loss; 
    if len(x): print(v, f"mean {x.mean():+.3f} worst {x.min():+.3f} n {len(x)}")
# paired per model clip0.1 vs minmax
for size in ['300','2k']:
    a=D[D.fam==f'ptqd{size}'].groupby(['run','fold']).loss.mean(); b=D[D.fam==f'ptqc{size}'].groupby(['run','fold']).loss.mean()
    dd=(b-a); print(size,'clip0.1-minmax per-model mean',round(dd.mean(),3),'better',int((dd>0).sum()),'/',len(dd),'wilcoxon p',round(stats.wilcoxon(dd).pvalue,4))
    # cluster by run (repeated runs share folds): run-level means
    print('   run-level diffs', dd.groupby(level=0).mean().round(3).to_dict())
for v in ['fp32','ptqd300','ptqc300','ptqd2k','ptqc2k']:
    s=D[D.fam==v]; print(v,'BA %.3f agree %.3f'%(s.bal_acc.mean(),s.agree_fp32.mean()))
print(D[D.variant=='fp32'].groupby('run').auroc.mean().round(3).to_dict())
