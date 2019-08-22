"""
Unfolding as quantum annealing
"""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("quantum_unfolding/__init__.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")
    
setup(
    name='quantum_unfolding',
    version=version,
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
    install_requires=requirements,
    scripts=[   'experiments/unfold_qubo_parallel.sh',
                'experiments/unfold_qubo_syst_parallel.sh', 
                'experiments/unfold_qubo.sh',
                'experiments/unfold_qubo_syst.sh',
                'experiments/unfolding_baseline.py', 
                'experiments/unfolding_qubo_syst.py',
                'experiments/unfolding_qubo.py',
                'experiments/systematics_analyzer.py',
                'plotting/plot_unfolded.py',
                'plotting/plot_unfolded_syst.py',
                'plotting/plot_unfolded_bits.py',
                'plotting/plot_unfolded_bits_violin.py',
                'plotting/plot_correlations.py',
            ]
)
