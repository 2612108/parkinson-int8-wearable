"""Within-fold statistics over fold-seed units (corrected repeated k-fold t, Bouckaert & Frank 2004 / Nadeau & Bengio 2003),
TOST +-0.02, plus pooled and fold-rank-normalised pooled H1-H3 per seed. Usage: python r1_stats.py <setting> run0 run1 run2"""
import sys, json, numpy as np, pandas as pd
from scipy import stats
U='/mnt/user-data/uploads/PARKINSON/reports/'
SH=['fs50','fs25','jitter10','jitter30','adc4g12b','adc2g10b','adc2g8b','noise_lo','noise_hi','rot15','rot45','rot180','mirror','nogyro']
def units(runs):
    rows=[]
    for s,r in enumerate(runs):
        d=pd.read_csv(U+f'{r}/shift_results.csv'); d=d[d.input=='raw']
        d['seed']=s; rows.append(d)
    return pd.concat(rows)
def table(D):
    c=D[D['shift']=='clean'].pivot_table(index=['seed','fold'],columns='variant',values='auroc')
    m=D[D['shift'].isin(SH)].groupby(['seed','fold','variant']).auroc.mean().unstack()
    return c,m
def ct(x, ratio=0.25, margin=None):
    x=np.asarray(x); n=len(x); mu=x.mean(); se=np.sqrt((1/n+ratio)*x.var(ddof=1)); df=n-1
    tq=stats.t.ppf(0.975,df); out=dict(mean=round(mu,3),lo=round(mu-tq*se,3),hi=round(mu+tq*se,3),p=round(2*stats.t.sf(abs(mu/se),df),4),n=n,min=round(x.min(),3))
    if margin:
        p1=stats.t.sf((mu+margin)/se,df); p2=stats.t.cdf((mu-margin)/se,df); out['p_tost']=round(max(p1,p2),4)
    return out
if __name__=='__main__':
    setting=sys.argv[1]; runs=sys.argv[2:]
    D=units(runs); c,m=table(D)
    print('variants',list(c.columns)); print('clean means',c.mean().round(3).to_dict()); print('shift means',m.mean().round(3).to_dict())
    res={}
    def add(name,x,margin=None): res[name]=ct(x,margin=margin); print(name,res[name])
    for v in [x for x in c.columns if x!='fp32']:
        ref='fp32_ft' if v=='ptq_ft' else ('fp32_sr' if v=='ptq_sr' else 'fp32')
        add(f'clean {v}-{ref}', c[v]-c[ref], margin=0.02)
    add('shift qat-ptq', m['qat']-m['ptq']); add('H3 within: shift qat_sr-ptq', m['qat_sr']-m['ptq'])
    add('H2 within', (c['ptq']-m['ptq'])-(c['fp32']-m['fp32']))
    add('SR clean fp32_sr-fp32', c['fp32_sr']-c['fp32']); add('SR shift qat_sr-qat', m['qat_sr']-m['qat'])
    print('per-unit PTQ loss', (c['ptq']-c['fp32']).round(3).tolist())
    if 'ptq_a16' in c: print('per-unit a16 loss', (c['ptq_a16']-c['fp32']).round(3).tolist())
    if 'ptq_stemf' in c: print('per-unit stemf loss', (c['ptq_stemf']-c['fp32']).round(3).tolist())
    json.dump(res,open(f'/home/claude/rev1/within_{setting}.json','w'),indent=1)
