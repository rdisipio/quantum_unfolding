"""
Unfolding as quantum annealing
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
    ],
    scripts=[   'experiments/unfold_qubo_parallel.sh',
                'experiments/unfold_qubo_syst_parallel.sh', 
                'experiments/unfold_qubo.sh',
                'experiments/unfold_qubo_syst.sh',
                'experiments/unfolding_baseline.py', 
                'experiments/unfolding_qubo_syst.py',
                'experiments/unfolding_qubo.py',
                'plotting/plot_unfolded.py',
                'plotting/plot_unfolded_syst.py',
                'plotting/plot_unfolded_bits.py',
                'plotting/plot_unfolded_bits_violin.py',
                'plotting/plot_correlations.py',
            ]
)
