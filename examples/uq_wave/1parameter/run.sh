#!/bin/bash
idx=${1/in./}
in=$1
out=$2

nimble="/projects/uq/build_NimbleSM_UQ/src/NimbleSM_Serial"

tag=""
while read line ; do
  items=($line)
  if [ ${items[1]} == "G"   ]; then G=${items[0]} ; fi
done < $in

dir="run_$idx"
if [ -e $dir ]; then rm -rf $dir ; fi
mkdir $dir
cd $dir
sed "s/{G}/$G/g" ../wave.in.tmpl > wave.in
cp ../../bar.g .
$nimble wave.in >& stdout
maxstress=`../../extract.py wave.serial.e`
echo $maxstress  >  ../$out
maxstress=`../../stress_solution.py | tail -n1`
echo $maxstress  >> ../$out

