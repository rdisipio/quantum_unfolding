"""Quantum unfolding
=====

Provides
 1. Tools to convert a regularized least-squares minimization to a quadratic unconstrained binary optimization
 2. Functions to solve the optimization using standard classical tools, simulated annealing, and quantum annealing

"""
__version__ = "1.0.0"
from .decimal2binary import  binary_matmul, d2b, discretize_matrix,  discretize_vector, BinaryEncoder, laplacian
from .unfolder import Backends, QUBOUnfolder, StatusCode
from .dwave_tools import get_energy

__all__ = ['Backends',
           'BinaryEncoder',
           'binary_matmul',
           'd2b',
           'discretize_matrix',
           'discretize_vector',
           'get_energy',
           'laplacian',
           'StatusCode']
