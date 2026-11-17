import sys
from r1_stats import units, table
runs=[('Zero-phase','0 (planned)','p56a','yes'),('Zero-phase','0 (repeat)','p56z0','yes'),('Zero-phase','1','p56z1','yes'),('Zero-phase','2','p56z2','yes'),
      ('Causal','0','p56k0','yes'),('Causal','1','p56k1','yes'),('Causal','2',sys.argv[1] if len(sys.argv)>1 else None,'yes'),
      ('Zero-phase','1','p56s1','no'),('Zero-phase','2','p56s2','no'),('Causal','0','p56c0','no'),('Causal','1','p56c1','no'),('Causal','2','p56c2','no')]
L=[]
for filt,seed,r,bn in runs:
    if r is None: continue
    c,_=table(units([r])); d=c.ptq-c.fp32; q=c.qat-c.fp32
    a16=f"${(c.ptq_a16-c.fp32).mean():+.3f}$" if 'ptq_a16' in c else '--'
    c2k=f"${(c.ptq_c2k-c.fp32).mean():+.3f}$" if 'ptq_c2k' in c else '--'
    L.append(f"{filt} & {seed} & {bn} & {c.fp32.mean():.3f} & ${d.mean():+.3f}$ (${d.min():+.3f}$) & {c2k} & {a16} & ${q.mean():+.3f}$ (${q.min():+.3f}$) \\\\")
print("\n".join(L))
