import sys; sys.argv=['x']
import hyp_rn
for s,r in [(0,'p56a'),(1,'p56z1'),(2,'p56z2'),(0,'p56z0')]:
    for norm in (False,True): hyp_rn.run('zp',r,s,B=2000,norm=norm)
