# Quantum Unfolding

Unfolding with quantum computing

## Check out package
```
git clone https://github.com/rdisipio/quantum_unfolding.git
```

# Test out calculation

In the file `toy_unfolding_classical.py` you can modify the definition of xedges, of the truth-level vector (`x`) and the response matrix (`R`). The code compares the product `Rx=y` carried out with decimal and binary representation, the latter to be used for the quantum computation.

The response matrix is converted to binary by repeateadly multiply it by one-hot vectors `(1,0,...,0), (0,1,0,...,0)` which are the standard basis of the linear operator `R`. The matrix-vector multiplication is carried out as usual, with the only exception that the carry bit has to be taken into account. 

```
./toy_unfolding_classical.py
```

