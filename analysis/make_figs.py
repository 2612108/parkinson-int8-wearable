import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from r1_stats import units, table, SH
zp=sys.argv[1].split(','); ca=sys.argv[2].split(','); outdir=sys.argv[3]
plt.rcParams.update({'font.size':9,'font.family':'serif'})
V=[('ptq','PTQ int8'),('ptq_c2k','PTQ, 2000 calib.'),('ptq_ft','PTQ of fine-tuned$^a$'),('ptq_stemf','PTQ, float stem'),('ptq_a16','PTQ 16x8'),('qat','QAT'),('qat_sr','QAT + SR')]
fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),sharey=True)
for ax,(runs,title) in zip(axes,[(zp,'Zero-phase filter'),(ca,'Causal (device) filter')]):
    c,_=table(units(runs))
    for i,(v,lab) in enumerate(V):
        if v not in c: continue
        ref='fp32_ft' if v=='ptq_ft' else 'fp32'
        d=(c[v]-c[ref]).values; seeds=c.index.get_level_values(0)
        for s,mk in zip(range(3),['o','s','^']):
            m=seeds==s; ax.scatter(d[m],np.full(m.sum(),i)+(s-1)*0.18,marker=mk,s=14,facecolors='none',edgecolors=['C0','C1','C2'][s],lw=0.9,label=f'seed {s}' if i==0 else None)
        ax.plot([d.mean()]*2,[i-0.35,i+0.35],'k-',lw=1.6)
    ax.axvline(0,color='0.5',lw=0.8); ax.axvspan(-0.02,0.02,color='0.9',zorder=0)
    ax.set_yticks(range(len(V))); ax.set_yticklabels([l for _,l in V]); ax.invert_yaxis()
    ax.set_xlabel('Within-fold AUROC minus FP32 reference'); ax.set_title(title,fontsize=9)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=7,frameon=False)
fig.tight_layout(rect=(0,0.06,1,1)); fig.savefig(f'{outdir}/Figure_2.pdf'); fig.savefig(f'/tmp/Figure_2.png',dpi=110)
# Figure 3: per shift within-fold AUROC (zero-phase)
D=units(zp); D=D[D['shift']!='fs50']
order=['clean']+[s for s in SH if s!='fs50']
labels={'clean':'Clean','fs25':'25 Hz','jitter10':'Jitter 10%','jitter30':'Jitter 30%','adc4g12b':'±4 g, 12 bit','adc2g10b':'±2 g, 10 bit','adc2g8b':'±2 g, 8 bit','noise_lo':'Noise low','noise_hi':'Noise high','rot15':'Rot. 15°','rot45':'Rot. 45°','rot180':'Rot. 180°','mirror':'Other wrist','nogyro':'No gyro'}
fig,ax=plt.subplots(figsize=(7.2,3.0))
VV=[('fp32','FP32','C0','o'),('ptq','PTQ','C3','s'),('qat','QAT','C2','^'),('qat_sr','QAT + SR','C4','D')]
for j,(v,lab,col,mk) in enumerate(VV):
    g=D[D.variant==v].groupby('shift').auroc; mu=g.mean().reindex(order); se=g.std().reindex(order)/np.sqrt(g.count().reindex(order))
    x=np.arange(len(order))+(j-1.5)*0.17
    ax.errorbar(x,mu,yerr=2.145*se*np.sqrt(1+0.25*15),fmt=mk,ms=3.5,color=col,lw=0.8,capsize=1.5,label=lab)
ax.set_xticks(range(len(order))); ax.set_xticklabels([labels[s] for s in order],rotation=35,ha='right')
ax.set_ylabel('Within-fold AUROC'); ax.legend(ncol=4,fontsize=7,frameon=False,loc='lower left'); ax.grid(axis='y',color='0.9')
fig.tight_layout(); fig.savefig(f'{outdir}/Figure_3.pdf'); fig.savefig('/tmp/Figure_3.png',dpi=110)
