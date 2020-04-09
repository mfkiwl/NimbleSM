#!/usr/bin/env python
import os,sys
import glob
import math

#########################################################
class kde:
#########################################################
  _npts = 101
  def k(self,x,sig):
    s2 = sig*sig
    return (1./math.sqrt(2.*math.pi*s2))*math.exp(-x*x/(2*s2))
  def std(self,data):
    n = len(data)
    x1 = 0.
    x2 = 0.
    for x in data:
      x1 += x
      x2 += x*x
    s2 = 1./(n-1)*(x2-x1*x1/n)
    return math.sqrt(s2)
  def process(self,data):
    n = len(data)
    xmin = 0.8*min(data)
    xmax = 1.2*max(data)
    h=self.std(data)*math.pow((4./(3.*n)),0.2) # Silverman's rule of thumb
    dx = (xmax-xmin)/(self._npts-1)
    pdf = []
    for i in range(self._npts):
      x = xmin + i*dx
      y = 0.
      for d in data:
        y += self.k(x-d,h)
      pdf.append([x,y])
    return pdf

#########################################################
def read_file(fname):
  f = open(fname)
  data = []
  d = None
  for line in f:
    items = line.split()
    if len(items) > 0:
      if items[0] == "#":
#       print("# time {0}".format(items[2]))
        if d is not None: data.append(d)
        d = []
      else:
        d.append(list(map(float,items)))
    else: pass
  f.close()
  return data

#########################################################
def read_column(fname,col=1):
  f = open(fname)
  data = []
  d = None
  for line in f:
    items = line.split()
    if len(items) > 0:
      if items[0] == "#":
#       print("# time {0}".format(items[2]))
        if d is not None: data.append(d)
        d = []
      else:
        d.append(float(items[col]))
    else: pass
  f.close()
  return data

#########################################################
def diff(v1,v2):
  w = list(map(lambda x,y:(x-y)**2,v1,v2))
  err = math.sqrt(sum(w))
  return err

#########################################################
def collect():
  ufile = "wave.dat"
  rdirs=glob.glob("run_*")
  rdata = []
  fdata = []
  udata = []
  ntrajs = len(rdirs)
  nsteps = 0
  nnodes = 0
  for i in range(ntrajs):
    uq = read_column(ufile,i+4)
    udata.append(uq)
    dname = "run_"+str(i+1)
    ref  = read_column(dname+"/solution.dat")
    rdata.append(ref)
    data = read_column(dname+"/wave.dat")
    fdata.append(data)
    nsteps = len(ref)
    nnodes = len(ref[0])

# print("# shape {0} {1} {2}".format(len(rdata),len(rdata[0]),len(rdata[0][0])))
  f = open("displacement.dat","w")
  for i in range(nsteps):
    f.write("# step {0}\n".format(i))
    for j in range(nnodes):
      for k in range(ntrajs):
        f.write("{0:9.4e} {1:9.4e} {2:9.4e} ".format(rdata[k][i][j],fdata[k][i][j],udata[k][i][j]))
      f.write("\n")
    f.write("\n\n")
  f.close()

#########################################################
def compute_distributions():
  data = read_file("displacement.dat")
  ntimes = len(data)
  nnodes = len(data[0])
  m = len(data[0][0])
  nsamples = m//3
  print("# shape {0} {1} {2}".format(len(data),len(data[0]),len(data[0][0])))
  nt = 4
  dt = ntimes/nt
  times = [ int(i*dt) for i in range(1,nt) ]
  print("# time samples {0}".format(times))
  nn = 4
  dn = nnodes/nn
  nodes = [ int(i*dn) for i in range(1,nn) ]
  print("# node samples {0}".format(nodes))
  for i in times:
    for j in nodes:
      f = open("dist_t="+str(i)+"_n="+str(j)+".dat","w")
      items = data[i][j]
      rs = [items[3*i  ] for i in range(nsamples) ]
      k1 = kde().process(rs)
      rs = [items[3*i+1] for i in range(nsamples) ]
      k2 = kde().process(rs)
      rs = [items[3*i+2] for i in range(nsamples) ]
      k3 = kde().process(rs)
      m = len(k1)
      for k in range(m):
        f.write("{0:9.3e} {1:9.3e} {2:9.3e} {3:9.3e} {4:9.3e} {5:9.3e}\n".format(k1[k][0],k1[k][1],k2[k][0],k2[k][1],k3[k][0],k3[k][1]))
      f.close()

#########################################################
def compute_error():
  ufile = "wave.dat"
  rdirs=glob.glob("run_*")
  f = open("error.dat","w")
  ntraj = len(rdirs)
  for i in range(ntraj):
    dname = "run_"+str(i+1)
    udata = read_column(ufile,i+4)
    ref  = read_column(dname+"/solution.dat")
    data = read_column(dname+"/wave.dat")
    nsteps = len(ref)
    for i in range(nsteps):
      errFE = diff(data[i],ref[i])
      errSU = diff(udata[i],ref[i])
      errUQ = diff(udata[i],data[i])
      f.write("{0:3d} {1:9.4e} {2:9.4e} {3:9.4e}\n".format(i,errFE,errSU,errUQ))
    f.write("\n\n")
  f.close()

#########################################################
def process():
# compute_error()
# collect()
  compute_distributions()

#########################################################
if __name__ == "__main__":
#########################################################
  process()
