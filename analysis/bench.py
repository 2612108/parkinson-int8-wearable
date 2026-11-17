import pandas as pd,numpy as np
from sklearn.metrics import roc_auc_score as A
from sklearn.model_selection import StratifiedKFold
from folds import P, lab
U='/mnt/user-data/uploads/PARKINSON/reports/'
def folds(task,seed):
    neg='HC' if task=='PD_vs_HC' else 'DD'
    sel=np.where(lab.isin(['PD',neg]))[0]; y=(lab=='PD').astype(int).to_numpy(); f={}
    for k,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(sel,y[sel])):
        for i in sel[te]: f[int(P.id[i])]=k
    return f
M=[('RF','p3','rf',[0],''),('XGBoost','p3','xgb',[0],''),('DS-CNN','p4cd','ds_cnn_raw_meanprob',[0],''),('1-D CNN','p4cd','cnn1d_raw_meanprob',[0],''),
   ('CNN-LSTM','p4cd','cnn_lstm_raw_meanprob',[0],''),('DS-TCN raw','p4ab','ds_tcn_raw_meanprob',[0],''),('DS-TCN invariant','p4ab','ds_tcn_inv_meanprob',[0],''),('DS-TCN MIL','p4ab','ds_tcn_raw_mil',[0],'')]
rows=[]
for name,dirk,stem,seeds,_ in M:
    r={'model':name}
    for task in ['PD_vs_HC','PD_vs_DD']:
        if dirk=='p3': dd='p3'
        elif dirk=='p4cd': dd='p4c' if task=='PD_vs_HC' else 'p4d'
        else: dd='p4a' if task=='PD_vs_HC' else 'p4b'
        po=[];wi=[];sd=[]
        for s in seeds:
            fn=f'{dd}/oof_{task}_{stem}_seed{s}.csv' if dirk!='p3' else f'p3/oof_{task}_{stem}_seed{s}.csv'
            d=pd.read_csv(U+fn,dtype={'sid':str}); d['fold']=d.sid.astype(int).map(folds(task,s))
            per=[A(g.y,g.p) for _,g in d.groupby('fold')]
            po.append(A(d.y,d.p)); wi.append(np.mean(per)); sd.append(np.std(per,ddof=1))
        r[task+'_pooled']=round(np.mean(po),3); r[task+'_within']=round(np.mean(wi),3); r[task+'_foldSD']=round(np.mean(sd),3)
    rows.append(r)
T=pd.DataFrame(rows); print(T.to_string(index=False)); T.to_csv('bench_within.csv',index=False)
