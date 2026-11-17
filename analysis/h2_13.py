import numpy as np
from scipy import stats
from r1_stats import units, ct, SH
def h2(runs, sh):
    D=units(runs)
    c=D[D['shift']=='clean'].pivot_table(index=['seed','fold'],columns='variant',values='auroc')
    m=D[D['shift'].isin(sh)].groupby(['seed','fold','variant']).auroc.mean().unstack()
    x=((c['ptq']-m['ptq'])-(c['fp32']-m['fp32'])).values
    n=len(x); se=np.sqrt((1/n+0.25)*x.var(ddof=1)); t90=stats.t.ppf(0.95,n-1)
    r=ct(x,margin=0.02); r['ci90']=(round(x.mean()-t90*se,3),round(x.mean()+t90*se,3)); return r
for name,runs in [('zp r',['p56r0','p56r1','p56r2']),('zp z',['p56z0','p56z1','p56z2']),('causal k',['p56k0','p56k1','p56k2'])]:
    for lab,sh in [('14',SH),('13',[s for s in SH if s!='fs50'])]:
        print(name,lab,h2(runs,sh))
