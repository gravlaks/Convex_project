import numpy as np
import cvxpy as cp
import random

# Fix seed for reproductability

def solve(A,b):
    '''
    auxiliary function that solves Ax=b and returns x
    A is a PSD matrix
    b is a vector
    This is done by minimizing the convex function xTAx - 2xTb
    (conjugate gradient descent method)
    '''
    assert(A.shape[0]==b.size, "error in auxiliary function solve: input dimensions don't match")
    x = cp.Variable(b.size)
    prob = cp.Problem(cp.Minimize((x.T)@(A@x) - 2*(x.T)@b), [])  # no constraint
    prob.solve()
    return x.value


def random_matrix(k,l,mode="sampling"):
    '''
    Generate a random projection matrix
    k is the number of rows, l the number of columns 
    mode is the way the matrix is generated 
    '''
    assert(k<l+1, "random matrix should reduce dimension, not increase it")
    R = np.zeros(k,l)
    if mode == "sampling": 
        # means we sample some components of the vector
        indexes = random.sample(range(l), k)  # sampling without repetition
        indexes.sort()
        for i in range(k):
            R[i, indexes[i]] = 1
    elif mode == "gaussian":
        # Each component is independantly sampled from N(0,1)
        # Then we divide by sqrt(k) to ensure columns have norm 1 (in expected value)
        R = np.random.randn(k,l) / np.sqrt(k)
    elif mode == "Hadamard":
        # only for k=l=power of 2 ??
        pass
    return R

'''
Algorithm 1 = Randomized Gauss-Newton Algorithm with Per-Sample Jacobian Projection
'''
def train_1(g,       # g is the model
        A,           # input training data
        y,           # output training data
        k = 10,      # projection dimension
        eps=0.1,     # stopping criterion
        lam=1,       # regularization parameter
        x_init = 0,  # starting point
        mode="sampling"):  # generation of random matrices
    '''
    trains the model g, ie. optimizes the parameters ('x' below) and return them
    g should have attributes
    - g.param_count --> integer
    - g.jac --> takes and returns np.array
    - g.forward --> takes and returns np.array
    '''
    assert(A.shape[0]==y.shape[0], "A and y don't have the same number of observations in function train")
    m, l = y.shape[0], y.shape[1]  # data set size and output dimension
    # initialization of model's parameters 
    if x_init == 0:
        x = np.zeros(g.param_count)  # x is not a scalar but an array of coefficients...
    else:
        assert(x.size==g.param_count, "initial parameters size and model's parameters size don't match")
        x_init = x
    # core optimization loop
    dist = eps + 1.0
    while dist > eps:
        # random generation of S_t 
        S = random_matrix(k,l,mode=mode)
        # x_new = argmin SUM_i ...
        # --> x_new satisfies A_matrix @ x_new = b_vector
        # we derive x_new here
        A_matrix = lam*np.identity(x.size)
        b_vector = lam*x
        for i in range(m):
            Ji = g.jac(A[i,:], y[i])        # jacobian of model's parameters for data point i
            SJ = np.matmul(S, Ji)           # SJ = S_t J_ti, S_t projection matrix, J_ti Jacobian
            A_matrix += np.matmul(SJ.T, SJ)
            residual = g.forward(x, A[i,:]) - np.matmul(Ji, x) - y[i,:]
            b_vector += np.matmul(SJ.T, np.matmul(S, residual))
        x_new = solve(A_matrix, b_vector)
        # compute the L2 norm of x_new - x
        dist = np.linalg.norm(x_new - x, 2)
        x = x_new  # x is updated 
    return x

'''
Algorithm 2 = Randomized Gauss-Newton Algorithm with Full Jacobian Projection
'''
def train_2():
    # to be done
    pass


if __name__ == "__main__":

    # data - to be loaded
    m = 100
    n = 10
    k = 1
    A = np.zeros(m,n)
    y = np.zeros(m,1)
    g = "model" # to be specified

    # initialization
    x = 0
    t = 0
    eps = 0.1
    lam = 1

    print("Model and data defined. Starting training now.")
    train_1(g, A, y)

