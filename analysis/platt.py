import json, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as A
from sklearn.linear_model import LogisticRegression
U='/mnt/user-data/uploads/PARKINSON/reports/'
for r in sys.argv[1].split(','):
    pp=pd.read_csv(U+r+'/patient_preds_PD_vs_HC_raw.csv').rename(columns={'Unnamed: 0':'sid'})
    fold=pp['meta|fold'].values; y=pp.y.values
    cal=[json.loads(l) for l in open(U+r+'/calib.jsonl')]
    out={}
    for v in ['fp32','ptq','qat']:
        lt=np.zeros(len(y)); pl=np.zeros(len(y)); pc=pp['clean|'+v].values
        for c in cal:
            k=c['fold']; m=fold==k; T,thr=c['calib'][v]
            lg=T*np.log(np.clip(pc[m],1e-9,1-1e-9)/np.clip(1-pc[m],1e-9,1))   # raw (uncalibrated) participant logit
            lt[m]=lg
            vp=np.array(c['val_p'][v]); vl=np.log(np.clip(vp,1e-9,1-1e-9)/np.clip(1-vp,1e-9,1))
            lr=LogisticRegression(C=1e6).fit(vl[:,None],np.array(c['val_y']))
            pl[m]=lr.predict_proba(lg[:,None])[:,1]
        per=np.mean([A(y[fold==k],pc[fold==k]) for k in range(5)])
        out[v]=dict(pooled_T=round(A(y,pc),3),pooled_raw=round(A(y,lt),3),pooled_platt=round(A(y,pl),3),within=round(per,3))
    # fold location of raw logits
    loc=[(c['fold'],round(float(np.median(np.log(np.clip(pp['clean|fp32'].values[fold==c['fold']],1e-9,1)/np.clip(1-pp['clean|fp32'].values[fold==c['fold']],1e-9,1))*c['calib']['fp32'][0])),2)) for c in cal]
    print(r,out,'median raw fp32 logit per fold',loc)
