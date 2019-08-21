#!/bin/bash

obs="peak"
nruns=20
reg=0
gamma=1
backend=qpu
enc=4 #nbits

while [ $# -gt 1 ] ; do
case $1 in
-o)   obs=$2     ; shift 2;;
-n)   nruns=$2   ; shift 2;;
-b)   backend=$2 ; shift 2;;
-l)   reg=$2     ; shift 2;;
-g)   gamma=$2   ; shift 2;;
-e)   enc=$2     ; shift 2;;
*) shift 1 ;;
esac
done

[ ! -d csv ] && mkdir -p csv
csvfile="csv/results_syst.obs_${obs}.${backend}.reg_${reg}.gamma_${gamma}.${enc}bits.csv" 
rm -f ${csvfile}

for i in $(seq ${nruns})
do
   ./unfolding_qubo_syst.py -e ${enc} -o ${obs} -l ${reg} -g ${gamma} -b ${backend} -f ${csvfile} | tail -n 1
done
