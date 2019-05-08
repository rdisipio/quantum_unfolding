#!/usr/bin/env python3

import argparse
import numpy as np
from decimal2binary import discretize_vector, discretize_matrix, compact_vector, laplacian
import dimod
from dwave.system import FixedEmbeddingComposite, DWaveSampler
from dwave_tools import get_embedding_with_short_chain, make_reverse_anneal_schedule
import neal
np.set_printoptions(precision=1, linewidth=200, suppress=True)


def print_results(result, n):
    energy_bestfit = result.first.energy
    q = np.array(list(result.first.sample.values()))
    y = compact_vector(q, n)
    print("INFO: best-fit:   ", q, "::", y, ":: E =", energy_bestfit)


def main(args):
    num_reads = int(args.nreads)
    dry_run = bool(args.dry_run)
    if dry_run:
        print("WARNING: dry run. There will be no results at the end.")

    # truth-level:
    x = [5, 8, 12, 6, 2]

    # response matrix:
    R = [[1, 2, 0, 0, 0],
         [1, 2, 1, 1, 0],
         [0, 1, 3, 2, 0],
         [0, 2, 2, 3, 2],
         [0, 0, 0, 1, 2]
         ]

    # pseudo-data:
    d = [12, 32, 40, 15, 10]

    # convert to numpy arrays
    x = np.array(x, dtype='uint8')
    R = np.array(R, dtype='uint8')
    b = np.array(d, dtype='uint8')

    # closure test
    b = np.dot(R, x)

    n = 4
    N = x.shape[0]

    print("INFO: N bins:", N)
    print("INFO: n-bits encoding:", n)

    lmbd = np.uint8(args.lmbd)  # regularization strength
    D = laplacian(N)

    # convert to bits
    x_b = discretize_vector(x, n)
    b_b = discretize_vector(b, n)
    R_b = discretize_matrix(R, n)
    D_b = discretize_matrix(D, n)

    print("INFO: Truth-level x:")
    print(x, x_b)
    print("INFO: pseudo-data b:")
    print(b, b_b)
    print("INFO: Response matrix:")
    print(R)
    print(R_b)
    print("INFO: Laplacian operator:")
    print(D)
    print(D_b)
    print("INFO: regularization strength:", lmbd)

    # Create QUBO operator

    # linear constraints
    h = {}
    for j in range(n*N):
        idx = (j)
        h[idx] = 0
        for i in range(N):
            h[idx] += (R_b[i][j]*R_b[i][j] -
                       2*R_b[i][j] * b[i] +
                       lmbd*D_b[i][j]*D_b[i][j])

    # quadratic constraints
    J = {}
    for j in range(n*N):
        for k in range(j+1, n*N):
            idx = (j, k)
            J[idx] = 0
            for i in range(N):
                J[idx] += 2*(R_b[i][j]*R_b[i][k] + lmbd*D_b[i][j]*D_b[i][k])

    # QUBO
    bqm = dimod.BinaryQuadraticModel(linear=h,
                                     quadratic=J,
                                     offset=0.0,
                                     vartype=dimod.BINARY)
    print("INFO: solving the QUBO model...")
    print("INFO: running on QPU")
    hardware_sampler = DWaveSampler()
    print("INFO: finding optimal minor embedding...")
    embedding = get_embedding_with_short_chain(J,
                                               tries=1,
                                               processor=hardware_sampler.edgelist,
                                               verbose=True)
    if embedding is None:
        raise ValueError("ERROR: could not find embedding")

    sampler = FixedEmbeddingComposite(hardware_sampler, embedding)
    print("INFO: creating DWave sampler...")
    print("INFO: annealing (n_reads=%i) ..." % num_reads)
    print("INFO: Connected to", hardware_sampler.properties['chip_id'])
    print("INFO: max anneal schedule points: {}".format(hardware_sampler.properties["max_anneal_schedule_points"]))
    print("INFO: annealing time range: {}".format(hardware_sampler.properties["annealing_time_range"]))
    schedule = make_reverse_anneal_schedule(s_target=0.99, hold_time=1,
                                            ramp_up_slope=0.2)
    neal_sampler = neal.SimulatedAnnealingSampler()
    initial_result = sampler.sample(bqm, num_reads=num_reads).aggregate()
    neal_result = neal_sampler.sample(bqm, num_reads=num_reads).aggregate()
    reverse_anneal_params = dict(anneal_schedule=schedule,
                                 initial_state=initial_result.first.sample,
                                 reinitialize_state=True)
    solver_parameters = {'num_reads': num_reads,
                         'auto_scale': True,
                         **reverse_anneal_params
                         }

    print(schedule)
    result = sampler.sample(bqm, **solver_parameters)

    reverse_anneal_params = dict(anneal_schedule=schedule,
                                 initial_state=neal_result.first.sample,
                                 reinitialize_state=True)
    solver_parameters = {'num_reads': num_reads,
                         'auto_scale': True,
                         **reverse_anneal_params
                         }

    result2 = sampler.sample(bqm, **solver_parameters)
    print("Neal results")
    print_results(neal_result, n)
    print("Reverse annealing on Neal results")
    print_results(result2, n)
    print("QPU results")
    print_results(initial_result, n)
    print("Reverse annealing on QPU results")
    print_results(result, n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Quantum unfolding")
    parser.add_argument('-l', '--lmbd', default=0.00)
    parser.add_argument('-n', '--nreads', default=10000)
    parser.add_argument('-b', '--backend', default='qpu')  # [cpu, qpu, sim, sim_embed]
    parser.add_argument('-d', '--dry-run', action='store_true', default=False)
    arg = parser.parse_args()
    main(arg)
