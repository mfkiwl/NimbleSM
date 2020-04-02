#! /usr/bin/env python
import os,sys
import multiprocessing
import timeit
import queue 
import math
import random

"""
example input files:

% comment
## SEED 1*10*1000
## X    0.5:100.0:10
## LOGY 0.5&100.0&10

# P1 P2 P3
1. 2. 3.
2. 2. 3.
1. 5. 3.
"""
## individual/specialized run script
_run="run.sh"
if not os.path.exists(_run): _run="run.py"

#################################################################
class Worker(multiprocessing.Process):
  def __init__(self, work_queue, result_queue):
    multiprocessing.Process.__init__(self)
    self.work_queue = work_queue
    self.result_queue = result_queue
 
  def run(self):
    while True:
      try: # get a task
        job = self.work_queue.get_nowait()
      except queue.Empty:
        break
      index,parameters = job
      print("# starting {0} ...".format(index))
      f = open("in."+str(index),"w")
      f.write("{0} variables\n".format(len(parameters)))
      ps = []
      for p in parameters:
        f.write("{0} {1}\n".format(p[0],p[1]))
        ps.append(p[0])
      f.close()
      t0 = timeit.default_timer()
      if not os.path.exists("out."+str(index)):
        os.system("./"+_run+" in."+str(index)+" out."+str(index))
      else:  
        print("!!! out."+str(index)+" exists !!!")
      t1 = timeit.default_timer()
      # collect result
      f = open("out."+str(index))
      lines = f.readlines()
      f.close()
      rs = [ (line.split())[0] for line in lines ]
      result = [index,t1-t0,ps,rs]
      self.result_queue.put(result)

##############################################################
def nlist(sizes):
  if not len(sizes):
    yield ()
    return
  size = sizes.pop(0)
  for outer in nlist(sizes):
    for inner in range(size):
      #yield outer + (inner,)
       yield (inner,) + outer

################################################################
def parse(pfile):
    names = []
    samples = []
    f = open(pfile)
    lines = open(pfile).readlines()
    items = lines[0].split()
    f.close()
    if   items[0] == "##": # generate samples from ranges
      ps = dict()
      names = []
      sizes = []
      for line in lines:
        items = line.split()
        name = items[1]
        names.append(name)
        pp   = items[2]
        if items[0] == "%": # comment
          pass
        else:
          if ":" in pp: # linear range
            pmin,pmax,np = pp.split(":")
            pmin = float(pmin)
            pmax = float(pmax)
            np = int(np)
            dp = (pmax-pmin)/(np-1)
            prange = []
            for i in range(np):
              p = pmin + dp*i
              prange.append(p)
          elif "&" in pp: # log range
            pmin,pmax,np = pp.split("&")
            pmin = float(pmin)
            pmax = float(pmax)
            np = int(np)
            dp = (pmax-pmin)/(np-1)
            prange = []
            for i in range(np):
              p = math.pow(10.,pmin + dp*i)
              prange.append(p)
          elif "*" in pp: # random sample
            try: 
              pmin,pmax,np = map(int,pp.split("*"))
              prange = random.sample(range(pmin,pmax),np)
            except:
              items = pp.split("*")
              pmin = float(items[0])
              pmax = float(items[1])
              np = int(items[2])
              prange = [ random.uniform(pmin,pmax) for i in range(np) ]
#         elif pp == "int": 
#           prange = map(int,  items[3:])
#         elif pp == "float": 
#           prange = map(float,items[3:])
          else: # list
            prange = items[2:]
          ps[name] = prange
          n = len(prange)
          sizes.append(n)
          print("# {0:8s} {1:3d} : {2}".format(name,n,prange))
      # tensor product sample space
      indices = list(nlist(sizes))
      for idx in indices:
        sample = []
        for i,j in enumerate(idx):
          p = ps[names[i]][j]
          sample.append(p)
        samples.append(sample)
    elif items[0] == "#": # give name order in the 1st line
      names = items[1:]
      samples = [ line.split() for line in lines[1:] ]
    ## echo parsed 
    print("# parameter names",names)
    pts = pfile+".pts"
    f = open(pts,"w")
    f.write("# ")
    for name in names:
      f.write("{0} ".format(name))
    f.write("\n")
    for sample in samples:
      for s in sample:
        f.write("{0}".format(s))
      f.write("\n")
    f.close()
    print("# >> wrote samples to:",pts)
    return names, samples

################################################################
if __name__ == "__main__":
################################################################
    pfile = sys.argv[1]
    nprocs = multiprocessing.cpu_count()
    if len(sys.argv) > 2:
      nprocs= int(sys.argv[2])
    names,samples = parse(pfile)
#   print(samples)

    njobs = len(samples)
    work_queue = multiprocessing.Queue()
    print("#--- {0} samples with {1} processes ---".format(njobs,nprocs))
    for i,ps in enumerate(samples):
       job = []
       for j,p in enumerate(ps):
         job.append([p,names[j]])
       work_queue.put([i+1,job])
    result_queue = multiprocessing.Queue()
 
    for i in range(nprocs):
      worker = Worker(work_queue, result_queue)
      worker.start()
 
    f = open("samples.dat","w")
    total_time = 0.0
    results = []
    for i in range(njobs):
      index,dt,parameters,result = result_queue.get()
      print("# job {0:4d} finished in {1:.1f} seconds".format(index,dt))
      total_time += dt
      for p in parameters:
        msg = "{0:12} ".format(p)
        print(msg,end="")
        f.write(msg)
      for r in result:
        msg = "{0:12} ".format(r)
        print(msg,end="")
        f.write(msg)
      print("")
      f.write("\n")
      f.flush()
    print("## average time {0:.1f} seconds".format(total_time/njobs))
