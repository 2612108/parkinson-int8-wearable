import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from r1_stats import units, SH
out='/mnt/user-data/outputs/PARKINSON/paper_bspc'
plt.rcParams.update({'font.size':9,'font.family':'serif'})
D=units(['p56r0','p56r1','p56r2']); C=D[D['shift']=='clean'].pivot_table(index=['seed','fold'],columns='variant',values='auroc')
diff=C.sub(C.fp32,axis=0)
fig,ax=plt.subplots(figsize=(7.2,3.4))
for j,(idx,row) in enumerate(diff.iterrows()):
    a=row[[c for c in row.index if c.startswith('ptqd300')]].values; b=row[[c for c in row.index if c.startswith('ptqd2k')]].values
    ax.scatter(np.full(len(a),j-0.15),a,s=10,facecolors='none',edgecolors='C0',lw=0.8,label='PTQ, 300-window draws' if j==0 else None)
    ax.scatter(np.full(len(b),j+0.15),b,s=10,marker='s',facecolors='none',edgecolors='C1',lw=0.8,label='PTQ, 2000-window draws' if j==0 else None)
    ax.scatter([j],[row.ptq_a16],marker='_',s=120,color='k',lw=1.5,label='PTQ 16x8' if j==0 else None)
    ax.scatter([j],[row.qat],marker='^',s=22,color='C2',label='QAT' if j==0 else None)
    ax.plot([j-0.32,j+0.02],[a.mean()]*2,color='C0',lw=1.2,label='Mean of 300-window draws' if j==0 else None)
    ax.scatter([j],[row.ptq],marker='x',s=30,color='k',lw=1.0,label='PTQ, original conversion' if j==0 else None)
ax.axhline(0,color='0.5',lw=0.8); ax.axhspan(-0.02,0.02,color='0.92',zorder=0)
ax.set_xticks(range(len(diff))); ax.set_xticklabels([f's{s}f{f}' for s,f in diff.index],rotation=90,fontsize=7)
ax.set_xlabel('Trained model (seed, fold)'); ax.set_ylabel('AUROC minus FP32 (same fold)')
fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=7,frameon=False)
fig.tight_layout(rect=(0,0.11,1,1)); fig.savefig(f'{out}/Figure_3.pdf'); fig.savefig('/tmp/Figure_3.png',dpi=110)
# Figure 3: per-shift difference from FP32, zero-phase (r runs) and causal (k runs)
labels={'fs25':'25 Hz','jitter10':'Jitter 10%','jitter30':'Jitter 30%','adc4g12b':'±4 g, 12 bit','adc2g10b':'±2 g, 10 bit','adc2g8b':'±2 g, 8 bit','noise_lo':'Noise low','noise_hi':'Noise high','rot15':'Rot. 15°','rot45':'Rot. 45°','rot180':'Rot. 180°','mirror':'Other wrist','nogyro':'No gyro','clean':'Clean'}
order=['clean']+[s for s in SH if s!='fs50']
fig,axes=plt.subplots(1,2,figsize=(7.2,3.6),sharey=True,sharex=True)
VV=[('ptq','PTQ','C3','s'),('qat','QAT','C2','^'),('fp32_sr','FP32 + SR','C0','o'),('qat_sr','QAT + SR','C4','D')]
for ax,(runs,title) in zip(axes,[(['p56r0','p56r1','p56r2'],'Zero-phase filter'),(['p56k0','p56k1','p56k2'],'Causal (device) filter')]):
    D=units(runs); P=D.pivot_table(index=['seed','fold'],columns=['shift','variant'],values='auroc')
    for j,(v,lab,col,mk) in enumerate(VV):
        mu=[];ci=[]
        for s in order:
            x=(P[(s,v)]-P[(s,'fp32')]).values; n=len(x); mu.append(x.mean()); ci.append(2.145*np.sqrt((1/n+0.25)*x.var(ddof=1)))
        y=np.arange(len(order))+(j-1.5)*0.18
        ax.errorbar(mu,y,xerr=ci,fmt=mk,ms=3.5,color=col,lw=0.8,capsize=1.5,label=lab)
    ax.axvline(0,color='0.5',lw=0.8); ax.set_title(title,fontsize=9); ax.set_xlabel('Within-fold AUROC minus FP32')
    ax.set_yticks(range(len(order))); ax.set_yticklabels([labels[s] for s in order]); ax.invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=7,frameon=False)
fig.tight_layout(rect=(0,0.07,1,1)); fig.savefig(f'{out}/Figure_2.pdf'); fig.savefig('/tmp/Figure_2.png',dpi=110)
