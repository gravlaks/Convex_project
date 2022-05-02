'''
Algorithm 1 = Randomized Gauss-Newton Algorithm with Per-Sample Jacobian Projection
'''

import numpy as np


# data - to be loaded
m = 100
n = 10
k = 1
A = np.zeros(m,n)
y = np.zeros(m,1)

# initialization 
x = 0
t = 0
eps = 0.1
lam = 1

# core 
dist = 1
while dist > eps:
    # random generation of S_t 
    S = np.identity(k)
    # x_new = argmin SUM_i ...
    dist = np.linalg.norm(x_new - x ,2)  # L2 norm of x_new - x
    x = x_new
return x


