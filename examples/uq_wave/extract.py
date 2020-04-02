#!/usr/bin/env python
import os,sys
f = open("exodus.stdout","w")
stdout = sys.stdout
sys.stdout = f
sys.path.append('../../tools')
sys.path.append('../../../tools')
sys.path.append('../../../../tools')
import exodus

sscale = 1.0e-9
blk=1
fname = sys.argv[1]
oname = fname.split(".")[0]+".dat"
#db = exodus.exodus(fname,mode='r',array_type='numpy')
db = exodus.exodus(fname)
times = list(db.get_times())
nsteps = len(times)
print("  {0} times {1}".format(nsteps, [ float("{0:7.4g}".format(t)) for t in times]))
o = open("t.dat","w")
for t in times: o.write("{0}\n".format(t))
o.close()
blks = list(db.get_elem_blk_ids())
nelms = db.num_elems()
nnodes = db.num_nodes()
print("  blocks {0} elements {1}".format(blks,nelms))
print("  nodes {0} coordinate names {1}".format(nnodes,db.get_coord_names()))
print("  nodal variables {0}".format(db.get_node_variable_names()))
print("  element variables {0}".format(db.get_element_variable_names()))
ntraj = 0
for name in db.get_node_variable_names():
  if name[:21] == "off_nom_displacement_" and name[-2:] == "_x":
    print(name)
    ntraj += 1
print(" number of adjacent trajectories {0}".format(ntraj))

xs,ys,zs = db.get_coords()
Xs = [ xs[i] for i in range(0,nnodes,4) ]
o = open("x.dat","w")
for x in Xs: o.write("{0}\n".format(x))
o.close()

elem_conn, num_blk_elems, num_elem_nodes = db.get_elem_connectivity(blk)
conn = []
k = 0
for i in range(num_blk_elems):
  c = []
  for j in range(num_elem_nodes):
    c.append(elem_conn[k]-1)
    k += 1
  conn.append(c)

#rint(len(conn))
#rint(num_blk_elems)
  
counts = nnodes*[0]
for c in conn:
  for i in c:  counts[i] += 1
weights = nnodes*[0.0]
for i,c in enumerate(counts): 
  if c > 0: weights[i] = 1.0/c

o = open(oname,"w")
maxs = 0.0
for i in range(db.num_times()):
  step = i + 1
  print("time: {}".format(times[i]))
  stress = db.get_element_variable_values(blk,'stress_xx',step)
# print(len(stress))
  for s in stress: maxs = max(s,maxs)
  ss = nnodes*[0.0]
  for j,c in enumerate(conn):
    for k in c:
#     print(k, " ",j)
      ss[k-1] += stress[j]
  for j,w in enumerate(weights): ss[j] *= w
  us = db.get_node_variable_values("displacement_x",step)
  vs = db.get_node_variable_values("velocity_x",step)
  wws = [ db.get_node_variable_values("off_nom_displacement_"+str(j)+"_x",step) for j in range(ntraj) ]
  o.write("# time {0}\n".format(times[i]))
  for j in range(0,nnodes,4):
    o.write("{0} {1} {2} {3} ".format(xs[j],us[j],vs[j],ss[j]*sscale))
    for ws in wws:
      o.write("{0} ".format(ws[j]))
    o.write("\n")
  o.write("\n\n")
o.close()
db.close()
maxs *= sscale 
print("{0} max_stress".format(maxs))
sys.stdout = stdout
print("{0}".format(maxs))
