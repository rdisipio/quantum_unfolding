"""Quantum unfolding
=====

Provides
 1. Tools to convert a regularized least-squares minimization to a quadratic unconstrained binary optimization
 2. Functions to solve the optimization using standard classical tools, simulated annealing, and quantum annealing

"""
__version__ = "1.0.0"
from .decimal2binary import  binary_matmul, compact_vector, d2b, discretize_matrix,  discretize_vector, BinaryEncoder, laplacian
from .unfolder import Backends, QUBOUnfolder, StatusCode
from .dwave_tools import anneal_sched_custom, get_embedding_with_short_chain, get_energy, make_reverse_anneal_schedule, merge_substates
from .input_data import input_data, R0, sigma_syst

__all__ = ['anneal_sched_custom',
           'Backends',
           'BinaryEncoder',
           'binary_matmul',
           'compact_vector',
           'd2b',
           'discretize_matrix',
           'discretize_vector',
           'get_embedding_with_short_chain', 
           'get_energy',
           'input_data',
           'laplacian',
           'make_reverse_anneal_schedule',
           'merge_substates',
           'R0',
           'sigma_syst',
           'StatusCode']
