"""
Ncpol2SDPA: A converter from commutative and noncommutative polynomial
optimization problems to sparse SDP input formats.
"""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
setup(
    name='quantum_unfolding',
    version='1.0.0',
    packages=['quantum_unfolding'],
    url='https://github.com/rdisipio/quantum_unfolding',
    keywords=[
        'unfolding',
        'quantum annealing'],
    license='LICENSE',
    description='Unfolding as quantum annealing',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python'
    ]
)
