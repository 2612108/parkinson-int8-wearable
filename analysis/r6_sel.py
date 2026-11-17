import pandas as pd,numpy as np,json,sys
from scipy import stats
base='/mnt/user-data/uploads/PARKINSON/reports'; runs=sys.argv[1:]
S=[];D=[]
for r in runs:
    d=pd.read_csv(f'{base}/{r}/shift_results.csv'); d=d[d['shift']=='clean'].copy(); fp=d[d.variant=='fp32'].set_index('fold').auroc
    d['loss']=d.auroc-d.fold.map(fp); d['run']=r; D.append(d)
    for l in open(f'{base}/{r}/calib.jsonl'):
        j=json.loads(l); f=j['fold']; va=j.get('val_auc',{}); dd=d[d.fold==f].set_index('variant')
        for s in ['300','2k']:
            fams={'clip0.1':f'ptqc{s}','clip1':f'ptqcb{s}','pct':f'ptqp{s}','mse':f'ptqm{s}','kl':f'ptqk{s}'}
            mv={k:np.mean([va[n] for n in va if n.startswith(fm+'_')]) for k,fm in fams.items()}
            kb=max(mv,key=mv.get); ch=dd.loc[[n for n in dd.index if n.startswith(fams[kb]+'_')],'loss']
            mm=[n for n in va if n.startswith(f'ptqd{s}_')]; bd=max(mm,key=lambda n:va[n])
            S.append(dict(run=r,fold=f,size=s,chosen=kb,cm=ch.mean(),cw=ch.min(),bd=dd.loc[bd,'loss']))
S=pd.DataFrame(S)
for s in ['300','2k']:
    x=S[S['size']==s]; print(s,'val-chosen method',x.chosen.value_counts().to_dict(),'mean',round(x.cm.mean(),3),'worst',round(x.cw.min(),3),'| best minmax draw mean',round(x.bd.mean(),3),'>0.05',f"{np.mean(x.bd<-0.05)*100:.0f}%",'worst',round(x.bd.min(),3))
D=pd.concat(D); D['fam']=D.variant.str.replace(r'_\d+$','',regex=True)
for s in ['300','2k']:
    a=D[D.fam=='ptqd'+s].groupby(['run','fold']).loss.mean(); b=D[D.fam=='ptqc'+s].groupby(['run','fold']).loss.mean()
    print(s,'clip0.1 vs minmax better',int(((b-a)>0).sum()),'/25 p',f"{stats.wilcoxon(b-a).pvalue:.1e}", 'run-level', (b-a).groupby(level=0).mean().round(3).to_dict())
    zp=[r for r in runs if 'w' in r]; c=[r for r in runs if 'y' in r]
    for g,lab in [(zp,'zp'),(c,'causal')]:
        dd=(b-a)[(b-a).index.get_level_values(0).isin(g)]; print('   ',lab,int((dd>0).sum()),'/',len(dd),f"p={stats.wilcoxon(dd).pvalue:.3f}")
