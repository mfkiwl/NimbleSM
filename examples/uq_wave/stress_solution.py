#!/usr/bin/env python
import os,sys
from math import pi, sqrt, sin, cos

# 1/2 (Cos[a - b] - Cos[a + b]) == Sin[a] Sin[b]
sscale = 1.e-9

if __name__ == "__main__":
  f = open("solution.log","w")
  stdout = sys.stdout
# sys.stdout = f
  ## info in the input file
  pdict = {}
  for line in open("wave.in"):
    name,value = line.split(":")
    pdict[name] = value
  tf = float(pdict["final time"])
  tsteps = int(pdict["number of load steps"])
  ssteps = int(pdict["output frequency"])
  matparam = pdict["material parameters"].split()
  shear_modulus = None
  bulk_modulus = None
  density = None
  for i in range(int(len(matparam)/2)):
    name = matparam[2*i]
    value = matparam[2*i+1]
#   print(name," ", value)
    if    name == "density"       : density = float(value)
    elif  name == "shear_modulus" : shear_modulus = float(value)
    elif  name == "bulk_modulus"  : bulk_modulus = float(value)
  K = bulk_modulus
  G = shear_modulus
  nu = (3.0*K-2.0*G)/(2.0*(3.0*K+G))
  E = 9.0*K*G/(3.0*K+G)  
  modulus = K + 4.0*G/3.0  # modulus = lambda + 2 mu
  print(" modulus {0:9.4g}".format(modulus))
  initialVelocity = 1000.0 # cm/sec NOTE in file but multiple "boundary condition"
  print(" initial velocity {0}".format(initialVelocity))
  ## info in the mesh
  xfile = "x.dat"
  xs = [ float(line.split()[0]) for line in open(xfile).readlines() ]
  x0 = xs[-1]
  length = xs[-1]-xs[0]
  tfile = "t.dat"
  times = [ float(line.split()[0]) for line in open(tfile).readlines() ]
  nsteps = len(times)
  tf = times[-1]
  dt = tf/(nsteps-1)
#  length  = 2.0     
#  dx = 0.02
#  x0 = -0.5*length 
#  nnodes = int(length/dx)+1
#  xs = [x0+i*dx for i in range(nnodes)];
#  zs = [xs[i]-xs[-1] for i in range(nnodes)];
#  nsteps = int(tsteps/ssteps)+1
# dt = tf/(nsteps-1)
# times = [i*dt for i in range(nsteps)];

  nterms = 1000
  c = sqrt(modulus/density)
  k = pi/(2.0*length)
  w = k*c
  A = 8.0*length*initialVelocity/(c*pi*pi)
  
  o = open("solution.dat",'w')
  maxs = 0.0
  for j,t in enumerate(times):
    print(">> {0:4d}/{1:4d} time {2:8.4g}".format(j+1,nsteps,t))
    o.write("# time {0:9.6g}\n".format(t))
    for kk,x in enumerate(xs):
#     z = zs[kk]
      z = x - x0
      u = 0.0
      v = 0.0
      s = 0.0
      for i in range(nterms):
        n = i + 1
        m = 2*n-1
        wn = m*w
        kn = m*k
        An = A/(m*m)
        u = u - An*sin(wn*t)*sin(kn*z) 
        v = v - An*cos(wn*t)*sin(kn*z)*wn 
        s = s - An*sin(wn*t)*cos(kn*z)*kn*modulus # s = E*grad(u)
        maxs = max(s,maxs)
      o.write("{0:7.4f} {1:9.6g} {2:9.6g} {3:9.6g} \n".format(x,u,v,s*sscale))
    o.write("\n\n")
  o.close()
  sys.stdout = stdout
  print("{0}".format(maxs*sscale))
