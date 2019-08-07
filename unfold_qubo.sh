#!/bin/bash

obs="peak"
nruns=20
reg=0
backend=qpu

while [ $# -gt 1 ] ; do
case $1 in
-o)   obs=$2     ; shift 2;;
-n)   nruns=$2   ; shift 2;;
-b)   backend=$2 ; shift 2;;
-l)   reg=$2     ; shift 2;;
#-f)   file=$2    ; shift 2;;
*) shift 1 ;;
esac
done

csvfile="results.obs_${obs}.${backend}.reg_${reg}.csv" 
rm -f ${csvfile}

for i in $(seq ${nruns})
do
   ./unfolding_qubo.py -o ${obs} -l ${reg} -b ${backend} -f ${csvfile} | tail -n 1
done
