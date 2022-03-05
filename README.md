# 1D Multigrid README

This folder contains code to solve the 1D Poisson problem on a line with Dirichlet boundary conditions. The code is written in Python. The precise contents of this folder are the following files:

1. A Jupyter notebook `BackgroundandResults.pynb` which contains mathematical preliminaries to explain the problem, a brief description of the multigrid method, and features of the code (along with some sample output)
2. A pdf of the Juptyer notebook
3. A Python file `1DPoissonMG.py` which contains code with the following functionality:


   - Solves prescribed problem with arbitrary boundary data and forcing
   - Provides comparisons to Numpy's numpy.linalg.solve() direct solver function and outputs execution time comparisons and Euclidean 2-norm errors in a PrettyTable format 
   - Tests optimality of relaxation parameter and outputs Euclidean 2-norm errors between multigrid and direct solver function in PrettyTable format and on a graph
   - Tests limits of grid size for multigrid to run and outputs runtime in PrettyTable format 
   - Compares multigrid solution to the analytical solution when the forcing is sin(x) and the boundary data is arbitrary via a pointwise error plot and outputs Euclidean 2-norm error

All of the functions in the code are carefully documented in the header of`1DPoissonMG.py` in the same format as NumPy - a brief description of the functionality and a description of all of the variables (both in data type and purpose). 

This was my way of starting to learn Python! It is a 1D implementation of a project that I performed in my first year of graduate school, although I added in new error diagnostic features.

