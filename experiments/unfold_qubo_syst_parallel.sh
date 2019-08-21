#!/bin/bash

obs="peak"
nruns=20
reg=0
gamma=1
backend=qpu
enc=4 #nbits
num_reads=1000

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
csvfile="data/results_syst.obs_${obs}.${backend}.reg_${reg}.gamma_${gamma}.${enc}bits.csv" 
rm -f ${csvfile}

cmdfile="parallel_cmd_syst.sh"
touch $cmdfile
chmod +x $cmdfile
for i in $(seq ${nruns})
do
  echo "unfolding_qubo_syst.py -e ${enc} -o ${obs} -l ${reg} -b ${backend} -g ${gamma} -f ${csvfile} -n ${num_reads}" >> $cmdfile
done
parallel --progress < $cmdfile
rm -f $cmdfile
echo "INFO: finished."
