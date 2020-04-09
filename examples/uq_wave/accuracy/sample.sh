#!/bin/bash 
nimble="/projects/uq/build_NimbleSM_UQ/src/NimbleSM_Serial"
seed="1234"
seed="1235"
seed="1236"
seed="1237"
parameter="G"
mean=1.5e12
#std=1.0e9
std=1.0e10
#std=1.0e11
#nsamples=10
#nsamples=100
nsamples=30  # 40 stalls
nsamples=40
addfields=""
for i in `seq $nsamples`; do
  j=$((i-1))
  addfields="$addfields off_nom_displacement_$j"
done
#echo $addfields

function clean() {
  echo "!! cleaning"
  rm -rf out.* in.* run_*
}

function run_uq() {
  echo ">> running uq $nsamples $mean +/- $std"
  sed "s/\#uq/uq/g;s/{ADDFIELDS}/$addfields/g;s/{NSAMPLES}/$nsamples/g;s/{STD}/$std/g;s/{$parameter}/$mean/g;s/{SEED}/$seed/g" wave.in.tmpl > wave.in
  $nimble wave.in &> uq.log
  ../extract.py wave.serial.e > extract.log
# cmd="$nimble wave.in &> uq.log"
# { time $cmd ; } 2>&1 | grep user | awk '{print $2}'
  echo "# ${parameter}" > sample.in
  tail -n$nsamples parameter_samples.dat | awk '{print $2}' >> sample.in
}

function run_exact() {
  echo ">> running exact"
  ../../../tools/sample.py sample.in > sample.log
# cmd="../../../tools/sample.py sample.in > sample.log"
# { time $cmd ; } 2>&1 | grep user | awk '{print $2}'
}

function compare() {
  compare.py > compare.log
}

clean
time run_uq
time run_exact
compare
