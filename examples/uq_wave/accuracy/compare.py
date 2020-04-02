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
  def process(self,data,fname):
    n = len(data)
    xmin = 0.
    xmax = 1.2*max(data)
    h=self.std(data)*math.pow((4./(3.*n)),0.2) # Silverman's rule of thumb
    dx = (xmax-xmin)/(_npts-1)
    pdf = []
    for i in range(_npts):
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
def process():
  ufile = "wave.dat"
  rdirs=glob.glob("run_*")
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
      print("{0:3d} {1:9.4e} {2:9.4e} {3:9.4e} ".format(i,errFE,errSU,errUQ))
    print("\n")

#########################################################
if __name__ == "__main__":
#########################################################
  process()
