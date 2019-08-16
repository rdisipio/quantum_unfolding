#!/bin/bash

obs="peak"
nruns=20
reg=0
backend=qpu_lonoise
enc=4 #nbits

while [ $# -gt 1 ] ; do
case $1 in
-o)   obs=$2     ; shift 2;;
-n)   nruns=$2   ; shift 2;;
-b)   backend=$2 ; shift 2;;
-l)   reg=$2     ; shift 2;;
-e)   enc=$2     ; shift 2;;
#-f)   file=$2    ; shift 2;;
*) shift 1 ;;
esac
done

[ ! -d csv ] && mkdir -p csv
csvfile="csv/results.obs_${obs}.${backend}.reg_${reg}.${enc}bits.csv" 
rm -f ${csvfile}
 
cmdfile="parallel_cmd.sh"
touch $cmdfile
chmod +x $cmdfile
for i in $(seq ${nruns})
do
  echo "./unfolding_qubo.py -e ${enc} -o ${obs} -l ${reg} -b ${backend} -f ${csvfile}" >> $cmdfile
done
parallel < $cmdfile
rm -f $cmdfile
echo "INFO: finished."
