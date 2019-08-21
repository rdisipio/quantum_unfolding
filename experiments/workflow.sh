#!/bin/bash

# 4-bits encoding
for obs in peak falling
do
  for backend in qpu_lonoise qpu_hinoise sim
  do
    for lmbd in 0 0.5 1
    do
      ./unfold_qubo_parallel.sh -e 4 -o ${obs} -b ${backend} -l ${lmbd}
    done
  done
done

# 8-bits encoding
for backend in qpu_lonoise qpu_hinoise sim
do
   ./unfold_qubo_parallel.sh -e 8 -o peak -b ${backend}
done

# run with systematics

for obs in peak falling
do
  for backend in qpu_lonoise qpu_hinoise sim
  do
    for gamma in 0 0.5 1
    do
      ./unfold_qubo_syst_parallel.sh -e 4 -o ${obs} -b ${backend} -g ${gamma}
    done
  done
done

