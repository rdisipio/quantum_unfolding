#!/bin/bash

nruns=500
reg=0
gamma=1
backend=qpu

while [ $# -gt 1 ] ; do
case $1 in
-n)   nruns=$2   ; shift 2;;
-b)   backend=$2 ; shift 2;;
-l)   reg=$2     ; shift 2;;
-g)   gamma=$2   ; shift 2;;
*) shift 1 ;;
esac
done

for i in $(seq ${nruns})
do
   ./unfolding_qubo_syst.py -l ${reg} -g ${gamma} -b ${backend} | tail -n 1
done
