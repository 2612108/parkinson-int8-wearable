import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
U='/mnt/user-data/uploads/PARKINSON/reports/'
P=pd.read_csv(U+'pads_patients.csv',dtype={'id':str})
lab=P.condition.map(lambda c:'PD' if c=="Parkinson's" else ('HC' if c=='Healthy' else 'DD'))
print(lab.value_counts().to_dict())
def folds_for(seed):
    sel=np.where(lab.isin(['PD','HC']))[0]; y=(lab=='PD').astype(int).to_numpy()
    f={}
    for k,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(sel,y[sel])):
        for i in sel[te]: f[int(P.id[i])]=k
    return f
def load(run,seed):
    d=pd.read_csv(U+f'{run}/patient_preds_PD_vs_HC_raw.csv'); d=d.rename(columns={'Unnamed: 0':'id'})
    if 'fold' not in d.columns:
        # blocks by id reset
        b=(np.diff(d.id.values)<0).cumsum(); d['fold_blk']=np.r_[0,b]
        f=folds_for(seed); d['fold']=d.id.map(f)
        print(run,'blocks==recon:',(d.fold_blk.values==d.fold.values).mean(), 'n',len(d))
    return d
if __name__=='__main__':
    for r,s in [('p56a',0),('p56s1',1),('p56s2',2)]: load(r,s)
    for r in ['p56c0','p56c1','p56c2']:
        d=pd.read_csv(U+f'{r}/patient_preds_PD_vs_HC_raw.csv'); print(r,d.columns[:5].tolist(),d.shape)
