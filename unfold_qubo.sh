#!/bin/bash

nruns=10
reg=0
backend=qpu


while [ $# -gt 1 ] ; do
case $1 in
-n)   nruns=$2   ; shift 2;;
-b)   backend=$2 ; shift 2;;
-l)   reg=$2     ; shift 2;;
*) shift 1 ;;
esac
done

for i in $(seq ${nruns})
do
   ./unfolding_qubo.py -l ${reg} -b ${backend} | tail -n 1
done
