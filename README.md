# Quantum Unfolding

Unfolding with quantum computing

## Check out package
```
git clone https://github.com/rdisipio/quantum_unfolding.git
```

## Additional packages
```
pip3 install dwave-system dwave-neal
```

## Test out calculation

In the file `toy_unfolding_classical.py` you can modify the definition of xedges, of the truth-level vector (`x`) and the response matrix (`R`). The code compares the product `Rx=y` carried out with decimal and binary representation, the latter to be used for the quantum computation.

The response matrix is converted to binary by recognizing that the linear vector space is spanned by `(n_cols X n_bits)` standard basis vectors `v`, i.e.:
```
( 0, 0, ..., 1 )
( 0, 1, ..., 0 )
( 1, 0, ..., 0 )
```
Multiplying `Rv` "extracts" the column corresponding to the non-zero element, e.g. `R x (1,0,...,0)^T` returns the first column of `R`. By iteration, we can convert R from decimal to binary.

The matrix-vector multiplication is carried out as usual, with the only exception that the carry bit has to be taken into account. 

```
./test_matrix_multiplication.py
```

To test the unfolding:

```
./toy_unfolding_classical.py
```

Test QUBO unfolding using DWave dimod package:
```
./unfolding_qubo.py -l 0 # no regularization
./unfolding_qubo.py -l 1 # regularization strength = 1
./unfolding_qubo.py -l 2 # regularization strength = 2
# etc..
```

Chose your backend wisely!
```
./unfolding_qubo.py -l 0 -b cpu # use CPU 
./unfolding_qubo.py -l 0 -b sim -n 2000 # use simulated annealer NEAL
./unfolding_qubo.py -l 0 -b qpu -n 2000 # use real QPU: you need a DWave Leap token
```

# Closure test

Using e.g. a 5x5 matrix:
 
```
./unfolding_qubo.py -l 0 -n 10000 -b sim
```

# Unfold with standard methods

Install RooUnfold:

```
cd $HOME/development
svn co https://svnsrv.desy.de/public/unfolding/RooUnfold/trunk RooUnfold
cd RooUnfold
make
```

This will create a library called ```libRooUnfold.so``` . You need to create
links to this library and the directory with the headers:

```
cd -
ln -s $HOME/development/RooUnfold/libRooUnfold.so .
ln -s $HOME/development/RooUnfold/src/ .
```

Then run:
```
./unfolding_baseline.py
```

With systematics:
```
./unfolding_baseline_syst.py # this gives you D'Agostini etc.

# Neal
./unfolding_qubo_syst.py -n 5000 -b sim -l 0.0
./unfolding_qubo_syst.py -n 5000 -b sim -l 0.5
./unfolding_qubo_syst.py -n 5000 -b sim -l 1.0

# QPU
./unfolding_qubo_syst.py -n 5000 -b qpu -l 0.0
./unfolding_qubo_syst.py -n 5000 -b qpu -l 0.5  
./unfolding_qubo_syst.py -n 5000 -b qpu -l 1.0
```

Make plots:
```
./plot_unfolded.py # unfolded only nominal, no systematics
./plot_unfolded_syst.py # unfolded w/ systematics
```
