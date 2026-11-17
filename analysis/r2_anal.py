"""R2 analyses on p56r0-2: calibration-draw variance, CLE, causal-input mismatch, decision metrics, pooled Platt."""
import json, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score as A
from r1_stats import units, table, ct
U='/mnt/user-data/uploads/PARKINSON/reports/'
import sys
runs=sys.argv[1].split(',')
D=units(runs); C=D[D['shift']=='clean']
piv=C.pivot_table(index=['seed','fold'],columns='variant',values='auroc')
diff=piv.sub(piv.fp32,axis=0)
d300=diff[[c for c in diff if c.startswith('ptqd300')]]; d2k=diff[[c for c in diff if c.startswith('ptqd2k')]]
print('units',len(piv))
print('300-draw: mean %.3f; within-model SD (mean over units) %.3f; between-unit SD of unit means %.3f; worst draw %.3f; frac draws < -0.05: %.2f'%(d300.values.mean(),d300.std(axis=1,ddof=1).mean(),d300.mean(axis=1).std(ddof=1),d300.values.min(),(d300.values<-0.05).mean()))
print('2k-draw: mean %.3f; within SD %.3f; worst %.3f; frac < -0.05: %.2f'%(d2k.values.mean(),d2k.std(axis=1,ddof=1).mean(),d2k.values.min(),(d2k.values<-0.05).mean()))
for v in ['ptq','ptq_c2k','ptq_fold','ptq_cle','ptq_a16','qat','qat_sr','ptq_ft']:
    if v in diff:
        ref = diff['fp32_ft'] if v=='ptq_ft' else 0
        print(v, ct(diff[v]-ref, margin=0.02))
# causal input mismatch: zero-phase trained models fed causally filtered input
S=D.pivot_table(index=['seed','fold'],columns=['shift','variant'],values='auroc')
for v in ['fp32','qat','ptq']:
    if ('causalin',v) in S: print('causalin',v, ct(S[('causalin',v)]-S[('clean',v)]))
# decision metrics clean
m=C.groupby('variant')[['auroc','bal_acc','sens','spec','ece','agree_fp32']].mean().round(3)
print(m.loc[[v for v in ['fp32','ptq','ptq_cle','ptq_a16','qat','qat_sr','fp32_ft'] if v in m.index]])
# H2 within + TOST
Cc,Mm=table(D)
print('H2 within',ct((Cc.ptq-Mm.ptq)-(Cc.fp32-Mm.fp32),margin=0.02))
# pooled vs within for FP32; temperatures
for r in runs:
    T=[json.loads(l) for l in open(U+r+'/calib.jsonl')]
    print(r,'T fp32 per fold',[round(t['calib']['fp32'][0],2) for t in T],'thr',[round(t['calib']['fp32'][1],2) for t in T])
