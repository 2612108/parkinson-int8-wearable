import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score as A
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from folds import P, U
age=dict(zip(P.id.astype(int),P.age)); sex=dict(zip(P.id.astype(int),P.gender))
print(P.gender.value_counts().to_dict())
for r in ['p56r0','p56r1','p56r2','p56k0','p56k1','p56k2']:
    d=pd.read_csv(U+f'{r}/patient_preds_PD_vs_HC_raw.csv').rename(columns={'Unnamed: 0':'id'})
    f=d['meta|fold'].values; y=d.y.values; p=d['clean|fp32'].values
    rn=np.zeros(len(p))
    for k in range(5): m=f==k; rn[m]=(rankdata(p[m])-0.5)/m.sum()
    sx=d.id.map(sex).values; out=[r]
    for g in ['female','male']: m=sx==g; out.append((g,round(A(y[m],rn[m]),3),int(m.sum()),int((y[m]==0).sum())))
    X0=np.c_[d.id.map(age).values,(sx=='female').astype(float)]; X1=np.c_[X0,rn]; p0=np.zeros(len(y)); p1=np.zeros(len(y))
    for k in range(5):
        tr=f!=k; te=f==k
        p0[te]=LogisticRegression(max_iter=1000).fit(X0[tr],y[tr]).predict_proba(X0[te])[:,1]; p1[te]=LogisticRegression(max_iter=1000).fit(X1[tr],y[tr]).predict_proba(X1[te])[:,1]
    out+=['age+sex',round(np.mean([A(y[f==k],p0[f==k]) for k in range(5)]),3),'+score',round(np.mean([A(y[f==k],p1[f==k]) for k in range(5)]),3)]
    print(out)
