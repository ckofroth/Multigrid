# 1D Multigrid README

This folder contains code to solve the 1D Poisson problem on a line with Dirichlet boundary conditions. The code is written in Python. The precise contents of this folder are the following files:

1. A Jupyter notebook background.pynb which contains mathematical preliminaries to explain the multigrid method 
2. A Python file 1DPoissonMG.py which contains code with the following functionality:
   - Solves prescribed problem with arbitrary boundary data and forcing
   - Provides comparisons to Numpy's numpy.linalg.solve() direct solver function and outputs execution time comparisons and Euclidean 2-norm errors in a PrettyTable format 
   - Tests optimality of relaxation parameter and outputs Euclidean 2-norm errors between multigrid and direct solver function in PrettyTable format and on a graph
   - Tests limits of grid size for multigrid to run and outputs runtime in PrettyTable format 

This was my way of starting to learn Python!

