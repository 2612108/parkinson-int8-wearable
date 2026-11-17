import sys, numpy as np, re
from r1_stats import units, table, ct
zp=['p56z0','p56z1','p56z2']; ca=sys.argv[1].split(',')
Cz,Mz=table(units(zp)); Cc,Mc=table(units(ca))
rows=[('FP32','fp32',None),('PTQ int8','ptq','fp32'),('PTQ, 2000 calibration windows','ptq_c2k','fp32'),('PTQ of fine-tuned FP32$^{a}$','ptq_ft','fp32_ft'),
      ('PTQ $16\\times8$','ptq_a16','fp32'),('QAT int8','qat','fp32'),('FP32 + SR','fp32_sr','fp32'),('PTQ + SR$^{b}$','ptq_sr','fp32_sr'),('QAT + SR','qat_sr','fp32')]
def cell(C,v,ref):
    if ref is None: return f"{C[v].mean():.3f} & -- & --"
    d=C[v]-C[ref]; r=ct(d)
    return f"{C[v].mean():.3f} & ${r['mean']:+.3f}$ (${r['lo']:+.3f}$, ${r['hi']:+.3f}$) & ${d.min():+.3f}$"
lines=[f"{n} & {cell(Cz,v,ref)} & {cell(Cc,v,ref)} \\\\" for n,v,ref in rows]
lines.append("\\midrule")
lines.append(f"H3 (QAT+SR $-$ PTQ, shift mean) & -- & ${ct(Mz['qat_sr']-Mz['ptq'])['mean']:+.3f}$ (${ct(Mz['qat_sr']-Mz['ptq'])['lo']:+.3f}$, ${ct(Mz['qat_sr']-Mz['ptq'])['hi']:+.3f}$) & -- & -- & ${ct(Mc['qat_sr']-Mc['ptq'])['mean']:+.3f}$ (${ct(Mc['qat_sr']-Mc['ptq'])['lo']:+.3f}$, ${ct(Mc['qat_sr']-Mc['ptq'])['hi']:+.3f}$) & -- \\\\")
ztab="\n".join(lines)
f=lambda C,v,ref: ct(C[v]-C[ref])
vals=dict(CA_PTQ=f"{-f(Cc,'ptq','fp32')['mean']:.3f}", CA_PTQMIN=f"{-min((Cz.ptq-Cz.fp32).min(),(Cc.ptq-Cc.fp32).min()):.3f}",
          CA_C2K=f"{-f(Cc,'ptq_c2k','fp32')['mean']:.3f}", CA_QAT=f"{f(Cc,'qat','fp32')['mean']:+.3f}", CA_SR=f"{f(Cc,'fp32_sr','fp32')['mean']:+.3f}")
a16=np.r_[(Cz.ptq_a16-Cz.fp32).values,(Cc.ptq_a16-Cc.fp32).values]
st=[]; 
for C in (Cz,Cc):
    m=C.ptq_stemf.notna(); st+=list(zip((C.ptq_stemf-C.fp32)[m].round(3),(C.ptq-C.fp32)[m].round(3)))
vals['STEMF']="float stem minus FP32: "+", ".join(f"{a:+.3f}" for a,b in st)+"; int8 PTQ in the same folds: "+", ".join(f"{b:+.3f}" for a,b in st)
print(vals); print('a16 min',a16.min(),'a16 tost', ct(a16,margin=0.02))
print('zero-phase', {k:ct(Cz[k]-Cz['fp32']) for k in ['ptq','ptq_c2k','qat','fp32_sr']})
print('H2 within z',ct((Cz.ptq-Mz.ptq)-(Cz.fp32-Mz.fp32)),'c',ct((Cc.ptq-Mc.ptq)-(Cc.fp32-Mc.fp32)))
print('c2k worse than ptq per run z', (Cz.ptq_c2k-Cz.ptq).groupby(level=0).mean().round(3).tolist(),'c',(Cc.ptq_c2k-Cc.ptq).groupby(level=0).mean().round(3).tolist())
if len(sys.argv)>2:
    p='/mnt/user-data/outputs/PARKINSON/paper_bspc/main.tex'; s=open(p).read()
    s=s.replace('\\textbf{ZTAB}',ztab)
    for k,v in vals.items(): s=s.replace('\\textbf{'+k+'}',v)
    open(p,'w').write(s); print('filled')
else: print(ztab)
