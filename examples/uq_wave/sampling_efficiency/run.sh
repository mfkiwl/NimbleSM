#!/bin/bash
idx=${1/in./}
in=$1
out=$2
nimble="/projects/uq/build_NimbleSM_UQ/src/NimbleSM_Serial"

tag=""
while read line ; do
  items=($line)
  if [ ${items[1]} == "N"   ]; then N=${items[0]} ; fi
done < $in

dir="run_$idx"
if [ -e $dir ]; then rm -rf $dir ; fi
mkdir $dir
cd $dir
sed "s/{N}/$N/g" ../wave.in.tmpl > wave.in
cp ../../bar.g .
( time -p $nimble wave.in ) 2> time.out > stdout
t=`grep real time.out | awk '{print $2}'`
echo $t  >  ../$out

