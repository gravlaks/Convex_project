
import numpy as np
import cvxpy as cp
import random
import torch
import sys
sys.path.append("../functions")
from tqdm import tqdm

def solve(A,b):
    '''
    auxiliary function that solves Ax=b and returns x
    A is a PSD matrix
    b is a vector
    This is done by minimizing the convex function xTAx - 2xTb
    (conjugate gradient descent method)
    '''
    def is_pos_def(A):
        x = (A==A.T)
        y = np.all(np.linalg.eigvals(A) > 0)
        return (x.all() and y)

    assert A.shape[0]==b.size, "error in auxiliary function solve: input dimensions don't match"
    assert is_pos_def(A), "matrix is not PSD"
    x = cp.Variable(b.size)
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, A) - 2*b.T@x), [])  # no constraint
    prob.solve()
    return x.value


def random_matrix(k,l,mode="sampling"):
    '''
    Generate a random projection matrix
    k is the number of rows, l the number of columns 
    mode is the way the matrix is generated 
    '''
    assert k<l+1, "random matrix should reduce dimension, not increase it"
    R = np.zeros((k,l))
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
        #eps=0.1,     # stopping criterion
        steps = 150, # 2nd stoping criterion: max number of iterations
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
    assert A.shape[0]==y.shape[0], "A and y don't have the same number of observations in function train"
    m, l = y.shape[0], y.shape[1]  # data set size and output dimension
    # initialization of model's parameters 
    if x_init.all() == 0:
        x = np.zeros(g.param_count)  # x is not a scalar but an array of coefficients...
    else:
        assert x_init.size==g.param_count, "initial parameters size and model's parameters size don't match"
        x = x_init
    # core optimization loop
    # dist = eps + 1.0
    # while dist > eps:
    for _ in tqdm(range(steps)):
        # random generation of S_t 
        S = random_matrix(k,l,mode=mode)
        # x_new = argmin SUM_i ...
        # --> x_new satisfies A_matrix @ x_new = b_vector
        # we derive x_new here
        A_matrix = lam*np.identity(x.size)
        b_vector = lam*x
        for i in range(m):
            Ji = g.jac(A[i,:], x)        # jacobian of model's parameters for data point i
            SJ = np.matmul(S, Ji)           # SJ = S_t J_ti, S_t projection matrix, J_ti Jacobian
            A_matrix += np.matmul(SJ.T, SJ)
            residual = g.forward(A[i,:], x) - np.matmul(Ji, x) - y[i,:]
            b_vector += np.matmul(SJ.T, np.matmul(S, residual))
        x_new = solve(A_matrix, b_vector)
        # compute the L2 norm of x_new - x
        # dist = np.linalg.norm(x_new - x, 2)
        x = x_new  # x is updated 
    return x


if __name__ == "__main__":

    N, n = 100, 2
    m = 1
    torch.manual_seed(1)

    #Construct Neural Network
    Ws0, bs0 = get_initial_params(hidden_layer_count=0, m=m, n=n, hidden_neurons=1)
    Ws_true, bs_true = get_initial_params(hidden_layer_count=0, m=m, n=n, hidden_neurons=1)
    X_true = (Ws_true, bs_true)
    X0 = (Ws0, bs0)

    nn = NN(X0)

    ## Generate X and Y labels
    A = np.random.randn(N, n)
    A[0, :] = np.ones((n,))
    Y = np.zeros((N, m))
    for i in range(N):
        a = A[i, :].reshape((n, 1))
        y_pred =  nn.forward(a, nn.flatten(X_true)).flatten()
        Y[i, :] = y_pred
    
    ## Run algorithm 1
    X_est = train_1(nn, A, Y, x_init = nn.flatten(X0), k=1)
    ## Print results
    for i in range(N):
        a = A[i, :].reshape((n, 1))
        y_true = nn.forward(a, nn.flatten(X_true)).flatten()
        y_est = nn.forward(a, X_est)
        print("Y_true", y_true, "Y_est", y_est)

