#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multgrid Solver for 1D Poisson Problem
By Collin Kofroth 

This code creates a full Multigrid V-cycle solver for the 1D Dirchlet Poisson 
problem

                    -u''=f, u(0)=g(0), u(1)=g(1),

where g(0) and g(1) are given boundary data. It has the capabilities of taking 
in random forcing f and boundary data g. We wll dscretize the probleem so that 
there aree N-1 interior notes, where ewe will take N to be a power of 2 larger 
than 1 (N^m, and m>1). The methods needed for the multgrid solver are 
the following:

    
Jacobi(x, f, omega, N)
    
    returns result of one iterations of the damped Jacobi method with matrix 
    given by finite differrence approximation of 2nd order derivative operator

    Parameters: x : array_like
                    iniitial guess
                    
                f : array_like
                    forcing
                    
                omega : scalar
                        damping parameter
                N : int
                    grid size 
                    
residual(x, f, N)
    
    returns residual f-Ax where A is the  matrix given by finite differrence 
    approximation of 2nd order derivative operator

    Parameters: x : array_like
                    approximate solution 
                    
                f : array_like
                    forcing

                N : int
                    grid size 
                    
coarsen(res_fine, N)
    
    returns coarsened/restricted vector along with coarsened grid sizee

    Parameters: res_fine : array_like
                           fine grid residual

                N : int
                    fine grid size 
                    
refine(res_coarse, N)
    
    returns refined/interpolated vector along with refined grid size

    Parameters: res_coarse : array_like
                             coarse grid residual 
                             
                N : int
                    coarse grid size                     
   
V_cycle(x, f, N, sweep_start, sweep_end, omega)

    returns the result of performing one multigrid v-cycle 

    Parameters: x : array_like
                    initial guess 
                    
                f : array_like
                    forcing

                N : int
                    grid size (SHOULD BE OF THE FORM 2^M, M>1 int)
                    
                sweep_start : int
                              number of inital Jacobi sweeps
                             
                sweep_end: int
                             number of final Jacobi sweeps
                             
                omega : scalar
                        damping parameter


We call our multigrid solver using the Multigrid class, which has the 
following methods:
    
    
solver_comp(self, N_array)

    returns None; compute the execution times for multigrid and NumPy direct
    solver and the absolute error for an array of grid sizes, then prints 
    this data in two tables (one table for execution times, one table 
    for errors)

    Parameters: self : object instance 
                             
                N_array : array_like
                          grid size array 

relax_test(self, omega_array, N)

    returns None; compute the absolute error for an array of relaxation 
    parameters, then prints a table of these errors and outputs a plot of this 
    data; omega = 2/3 is theoretically optimal, but this is not so noticable 
    for small perturbations
                    
    Parameters: self : object instance 
                             
                N_array : array_like
                          grid size array 

mg_limit(self, N_array)

    returns None; compute the execution time for multigrid for an array of
    grid sizes and prints this data in a table; allows one to solve much 
    larger problems (since we are not using the direct solver at the top level)
                    
    Parameters: self : object instance 
                             
                N_array : array_like
                          grid size array 

ana_comp(self, N, g):
    
    returns None; compare multigrid solution to the analytical solution when 
    f(x)=sin(x); generates plot of pointwise error vector and prints the 
    Euclidean error of this vector
                     
    Parameters: self : object instance 
                             
                N : int
                    rid size
                    
                g : array_like
                    boundary data    
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from prettytable import PrettyTable
from scipy.sparse import diags

          
# run weighted Jacobi method for 1D Poisson with parameter omega

def Jacobi(x, f, omega, N): 
    
    dx2        = 1/(N**2)
    omega_min1 = 1.0-omega;
    omega_div2 = 0.5*omega;
    x_new      = x[:]
    
    for i in range(1,N):
        
        x_new[i] = omega_min1*x[i] \
            + omega_div2*(x[i-1] + x[i+1] + dx2*f[i])

    return x_new 


# compute residual f-Ax for an approximate solution x to the equation Ay=b

def residual(x, f, N): 
    
    res      = np.zeros(N+1)
    res[1:N] = [f[i] - N**2 * (2*x[i] - x[i-1] - x[i+1])\
                for i in range(1, N)]
        
    return res


# coarsen vector and grid where vector is defined through restriction

def coarsen(res_fine, N): 
    
    M          = int((N+1) / 2)
    res_coarse = np.zeros(M+1)
    res_coarse = [res_fine[2*i] for i in range(0,M+1)] 
    
    return res_coarse, M


# refine vector and grid wherervector is defined through interpolation

def refine(res_coarse, N):     
    
    res_fine = np.zeros(2*N+1)  
    res_fine[0:2*N:2] = [res_coarse[i] for i in range(0, N)]
    res_fine[1:2*N:2] = [0.5*(res_coarse[i] + res_coarse[i+1])\
                         for i in range(0, N)]    
    res_fine[2*N]     = res_coarse[N]

    return res_fine, 2*N 


# perform multi-grid V-cycle 

def V_cycle(x, f, N, sweep_start, sweep_end, omega): 
       
    # starting sweeps to damp high frequency modes
    
    for i in range(sweep_start):
        
        y = Jacobi(x,f, omega, N)
        x = y[:]
    
    # coompute residual, then coarsen it
    
    res             = residual(y, f, N) 
    [res_coarse, N] = coarsen(res, N)
    
    
    if N>4: # if problem is sufficiently large, recursively apply V-cyccle to 
            # residual equation, then refine
        
        x0          = np.zeros(N+1)
        x_coarse    = V_cycle(x0, res_coarse, N, sweep_start, \
                             sweep_end, omega)      
        [x_fine, N] = refine(x_coarse, N)
        
    else: # if N is small enough, solve directly
        
        # set up Laplacian matrix A, then do direct solve on residual equation
        
        band         = [-1*np.ones(N-2), 2*np.ones(N-1), -1*np.ones(N-2)]
        offset       = [-1, 0, 1]
        A            = (N**2) * diags(band,offset).toarray()
        exact        = np.zeros(N+1)
        exact[1:N]   = np.linalg.solve(A, res_coarse[1:N])
        [x_fine,  N] = refine(exact, N) 
    
    # correct approximation with error estimation 
    
    y       = np.add(x_fine, y)  
    x_final = np.zeros(N+1)
    
    # relax again
    
    for i in range(sweep_end):
        
        x_final = Jacobi(y, f, omega,N)
        y       = x_final[:]
        
    return x_final


# Multigrid class

class Multigrid:
    
    def solver_comp(self, N_array):
        
        # set up multigrid parameters and boundary
        
        omega       = 2/3
        sweep_start = 3
        sweep_end   = 3
        cycles      = 8
        g           = np.random.rand(2)
        
        # instantiate data tables
        
        table1 = PrettyTable()
        table2 = PrettyTable()
        
        # label column headings
        
        table1.field_names = ["Grid Size", "Multigrid Execution Time (s)",\
                              "NumPy Direct Solver Execution Time (s)"]
        table2.field_names = ["Grid Size", "Absolute Euclidean Error"]

        # right-align headings 
        
        table1.align["Grid Size"]                              = "r"
        table1.align["Multigrid Execution Time (s)"]           = "r"
        table1.align["NumPy Direct Solver Execution Time (s)"] = "r"               
        table2.align["Grid Size"]                              = "r"
        table2.align["Absolute Euclidean Error"]               = "r" 
        
        for N in N_array: # sweep through array of grid sizes
            
            # set up initial guess and forcing 
            
            x    = np.zeros(N+1)
            f    = np.random.rand(N+1)
            f[0] = 0
            f[N] = 0
            x[0] = g[0] # get left boundary correct
            x[N] = g[1] # get right boundary correct
            tic  = time.perf_counter() # start multigrid timer
            
            for i in range(cycles): # do "cycles" V-cycles
                
                y  = V_cycle(x, f, N, sweep_start, sweep_end, omega)
                x  = y[:]
                
            toc     = time.perf_counter() # end multigrid timer
            mg_time = toc - tic # compute execution time for multigrid
            tic     = time.perf_counter() # start direct solve timer
            
            # set up Laplacian matrix A, then do direct solve 
            
            band       = [-1*np.ones(N-2), 2*np.ones(N-1), -1*np.ones(N-2)]
            offset     = [-1, 0, 1]
            A          = diags(band, offset).toarray()
            ff         = (N**-2) * f[1:N]
            x_dir      = np.zeros(N+1)
            x_dir[0]   = g[0]
            x_dir[N]   = g[1]
            gg         = np.zeros(N-1)
            gg[0]      = g[0]
            gg[N-2]    = g[1]
            RHS        = np.add(ff, gg)
            x_dir[1:N] = np.linalg.solve(A, RHS)

            toc         = time.perf_counter() # end direct solve timer
            direct_time = toc - tic # compute execution time for direct solve
            
            # compute 2-norm of error vector 
            
            err = np.linalg.norm(np.subtract(x, x_dir))
            
            # add data to the tables
            
            table1.add_row([N+1, mg_time, direct_time])
            table2.add_row([N+1, err])
        
        # print tables
        
        print(table1)
        print(table2)
        
        return None
        
    def relax_test(self, omega_array, N): 

        # set up multigrid parameters
        
        sweep_start = 3
        sweep_end   = 3
        cycles      = 8      
        
        table = PrettyTable()  # instantiate data table 
        
        # label column headings
        
        table.field_names = ["Grid Size", "Relaxation Parameter",\
                             "Error"]
           
        # right-align headings
            
        table.align["Grid Size"]            = "r"
        table.align["Relaxation Parameter"] = "r"
        table.align["Error"]                = "r" 
            
        # generate initial guess wth both high and low frequency components
        
        k1 = int((N/2)-1)
        k2 = 2
        I  = np.linspace(0, N, N+1)
        x  = [0.5*( np.sin(np.pi*i*k1 / N) + np.sin(np.pi*i*k2 / N) )\
              for i in I] 
            
        # generate boundary data and forcing
            
        g       = np.random.rand(2)    
        f       = np.random.rand(N+1)
        f[0]    = 0
        f[N]    = 0
        x[0]    = g[0] # get left boundary correct
        x[N]    = g[1] # get right boundary correct
        err_vec = [ ]  # initialize error vector (each component is 2-norm 
                       # error for given relaxation parameter)
        
        for omega in omega_array: # sweep through relaxation parameter array
            
            for i in range(cycles): # do "cycles" V-cycles
                
                y   = V_cycle(x, f, N, sweep_start, sweep_end, omega)
                x   = y[:]   
            
            # set up Laplacian matrix A, then do direct solve 

            band       = [-1*np.ones(N-2), 2*np.ones(N-1), -1*np.ones(N-2)]
            offset     = [-1, 0, 1]
            A          = diags(band, offset).toarray()
            ff         = (N**-2)*f[1:N]
            x_dir      = np.zeros(N+1)
            x_dir[0]   = g[0]
            x_dir[N]   = g[1]
            gg         = np.zeros(N-1)
            gg[0]      = g[0]
            gg[N-2]    = g[1]
            RHS        = np.add(ff, gg) 
            x_dir[1:N] = np.linalg.solve(A, RHS)
            
            # compute 2-norm of error then add to error vector
            
            err = np.linalg.norm(np.subtract(x, x_dir)) 
            err_vec.append(err) #
            
            table.add_row([N+1,omega,err]) # add error to table        
        
        print(table) # print table
        
        # generate semilog plot (log in y) of the error vs. 
        # the relaxation parameter; mark data points with red asterisk
        
        plt.semilogy(omega_array, err_vec)
        plt.semilogy(omega_array, err_vec, '*r')
        plt.xlabel("Relaxation Parameter $\omega$")
        plt.ylabel("Absolute Error")
        plt.title("Relaxation Parameter Optimality")
        plt.grid()
        plt.show()
        
        return None
        
    def mg_limit(self, N_array):
        
        # set up multigrid parameters and boundary

        omega       = 2/3
        sweep_start = 3
        sweep_end   = 3
        cycles      = 8
        g           = np.random.rand(2)
        
        table = PrettyTable() # instantiate data table 
        
        # label column headings
        
        table.field_names = ["Grid Size", "Multigrid Execution Time (s)"]
        
        # right-align headings
        
        table.align["Grid Size"]                    = "r"
        table.align["Multigrid Execution Time (s)"] = "r" 
        
        for N in N_array: # sweep through array of grid sizes
        
            # initialize initial guess and forcing 
            
            x    = np.zeros(N+1)
            f    = np.random.rand(N+1)
            f[0] = 0
            f[N] = 0
            x[0] = g[0] # get left boundary correct
            x[N] = g[1] # get right boundary correct
            tic  = time.perf_counter() # start multigrid timer
            
            for i in range(cycles): # do "cycles" V-cycles 
                
                y = V_cycle(x, f, N, sweep_start, sweep_end, omega)
                x = y[:]
                
            toc     = time.perf_counter() # end multigrid timer
            mg_time = toc - tic # compute multigrid execution time
            table.add_row([N+1, mg_time])  # add data to table 
       
        print(table) # print table 
        
        return None
      
    def ana_comp(self, N, g):
        
        # initialize grid and analytical solution
        
        grid = np.linspace(0, 1, N+1)
        x_ana = [g[0] - g[0]*j + g[1]*j - np.sin(1)*j + np.sin(j) \
                        for j in grid]
        
        # initialize multgrid slution and forcing
        
        x_mg    = np.zeros(N+1)
        f       = [np.sin(j) for j in grid]
        f[0]    = 0
        f[N]    = 0
        x_mg[0] = g[0]
        x_mg[N] = g[1]
        
        # set up multigrid parameters
        
        omega       = 2/3
        sweep_start = 3
        sweep_end   = 3
        cycles      = 4
        
        for i in range(cycles): # do "cycles" V-cycles
            
            y    = V_cycle(x_mg, f, N, sweep_start, sweep_end, omega)
            x_mg = y[:]
        
        # generate pointwise error vector   
        
        err_vec = np.subtract(x_mg, x_ana)
        err_vec = [abs(err) for err in err_vec]
        
        # create error plot 
        
        plt.plot(grid, err_vec, linewidth = 2, label = r'$|u(x_j)-u_{mg}[j]|$')
        plt.grid()
        plt.legend()
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.title(r"Error Vector for Analytical and Multigrid Comparison For" 
                  "\n"
                  "$f(x)=\sin(x)$ and $u(0)=$%f, $u(1)=$%f" % (g[0], g[1]))        
        plt.tight_layout()
        plt.show()
        
        # print 2-norm error
        
        print("Euclidean error is", np.linalg.norm(err_vec))
        
        return None
    
##############################################################################
####                                                                      ####
####                         Run multigrid tests                          ####
####                                                                      ####
##############################################################################


drive  = Multigrid() # instantiate multigrid class

# test solver_comp

# N_array1    = [2**n for n in range(6,14)] # array of grid sizes
# drive.solver_comp(N_array1)

# test relax_test 

# omega_array = [0.5, 0.6, 2/3, 0.7, 0.8] # array of relaxation parameters
# N           = 2**8
# drive.relax_test(omega_array, N)


# test mg_limit

# N_array2    = [2**n for n in range(6,16)]  # array of grid sizes
# drive.mg_limit(N_array2)

# test ana_comp

N = 2**6   # grid size
g = [-1,1] # boundary data
drive.ana_comp(N, g)