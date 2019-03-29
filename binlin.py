import numpy as np
import dimod
from dwave.system import EmbeddingComposite, DWaveSampler

# sampler = dimod.SimulatedAnnealingSampler()
sampler = EmbeddingComposite(DWaveSampler())


def setup_random(n):
    A = np.random.rand(n, n) / n
    b = np.random.rand(n)
    return A, b


def setup_easy(att=0.1, off=0.9):

    # x = [0, 1, 0]
    A, b = setup_random(3)
    A[:, 0] *= att
    A[:, 2] *= att
    b = off * A[:, 1]

    return A, b


def bruteforce(A, b):

    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0] == len(b)

    n = len(b)
    minnorm = float('inf')
    bestx = None

    for i in range(2**n):
        x = np.array([int(d) for d in np.binary_repr(i, n)])
        newnorm = np.linalg.norm(np.matmul(A, x) - b)
        if newnorm < minnorm:
            minnorm = newnorm
            bestx = x.copy()

    return bestx, minnorm


def binlin(A, b, scaling=1 / 8.0):

    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0] == len(b)

    n = len(b)
    bqm = dimod.BQM.empty(dimod.BINARY)

    for i in range(n):
        for j in range(n):
            bqm.add_variable(j, scaling * A[i, j] * (A[i, j] - 2 * b[i]))
            for k in range(j):
                bqm.add_interaction(j, k, scaling * 2 * A[i, j] * A[i, k])

    return bqm


# setup
n = 3
A, b = setup_random(n)
# A, b = setup_easy()
scaling = 1.0 / n
chain_strength = 1.0
num_reads = 1000
# sampling
bqm = binlin(A, b, scaling=scaling)
samples = sampler.sample(bqm, num_reads=num_reads).aggregate()
# brute-force exact solution
bestx, minnorm = bruteforce(A, b)
# results
print("A =", A)
print("b =", b)
print("Brute-force solution:", bestx, "norm:", minnorm)
print()

idxbest = None
for idx, sample in enumerate(samples.record.sample):
    isvalid = np.linalg.norm(sample - bestx) == 0
    if isvalid:
        idxbest = idx
        print("Solution #{} (valid = {})".format(idx, isvalid))
        print("Sample:", sample)
        print("Energy:", samples.record[idx].energy)
        print("Occurrences:", samples.record[idx].num_occurrences)
        print("Norm:", np.linalg.norm(np.matmul(A, sample) - b))
        print()

print("Distinct samples:", len(samples))
print("Uniform sampling baseline:", 100.0 / 2**len(b), "%")
if idxbest is not None:
    print("Valid solutions",
          100.0 * samples.record[idxbest].num_occurrences / num_reads, "%")
else:
    print("Valid solution not found.")
