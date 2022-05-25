import math
import numpy as np
import scipy.sparse.linalg as linalg
import random
import torch
import sys
sys.path.append("../functions")
from functions.modular_nn import NN, get_initial_params
from tqdm import tqdm


def solve(A,b):
    '''
    A is a PSD matrix
    b is a vector
    auxiliary function that solves Ax=b and returns x

    This is done by minimizing the convex function (1/2)xTAx - xTb
    via stochastic gradient descent (conjugate gradient descent method),
    which is faster than inverting A
    '''
    def is_pos_def(A):
        # check that A is PSD
        x = (A==A.T)
        y = np.all(np.linalg.eigvals(A) > 0)
        return (x.all() and y)
    assert A.shape[0]==b.size, "error in auxiliary function solve: input dimensions don't match"
    assert is_pos_def(A), "matrix is not PSD"

    # definition of the optimization problem
    # x = cp.Variable(b.size)
    # prob = cp.Problem(cp.Minimize((1./2)*cp.quad_form(x, A) - np.dot(b.T, x)), [])  # no constraint here
    # cvxpy library uses stochastic gradient descent (SGD) to solve it:
    # prob.solve()
    # return x.value

    x, info = linalg.cg(A,b)
    assert info==0, "conjugate gradient algorithm unsuccesful in algo1"

    return x

def random_matrix(k,l,mode="sampling"):
    '''
    Generate a random projection matrix
    k is the number of rows
    l the number of columns
    mode is the way the matrix is generated
    ("sampling", "gaussian" or "Hadamard")
    '''
    if l==1:
        return np.ones((1,1))
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
        # first we construct the classical Hadamard matrices
        def square_hadamard(n):
            # returns a Hadamard matrix of dimension (2^n, 2^n)
            assert n>=0, "index of Hadamard matrix needs to be an integer"
            if n==0:
                return np.ones((1,1))
            elif n==1:
                M = np.ones((2,2))
                M[1,1] = -1
                return M
            else:
                M = square_hadamard(n-1)
                u = 2 ** (n-1)
                N = np.ones((2*u, 2*u))
                N[:u, :u] = M
                N[:u, u:] = M
                N[u:, :u] = M
                N[u:, u:] = -M
                return N
        # when projection dimension is not exactly a power of 2, 
        # we pad the matrix with zeros
        n = int(math.log(k,2))
        M = square_hadamard(n)
        R[:2**n, :2**n] = M
    else:
        print("ERROR in random_matrix function of algo1: this mode is not defined")
    return R


def MSE(g, X, A, Y):
    '''
    g is the model
    X is its parameters 
    A is input training data
    Y is output training data

    Return the mean squared error (MSE) of model g 
    with parameters X on training set (A, Y)
    '''
    return np.mean( 
        ( np.array([g.forward(A[i,:], X) for i in range(A.shape[0])]).reshape((-1, 1)) - Y.reshape((-1, 1)) )**2 )



'''
Algorithm 1 = Randomized Gauss-Newton Algorithm with Per-Sample Jacobian Projection
'''
def train_1(g,              # g is the model
        X0,                 # starting point for the parameters
        A,                  # input training data
        Y,                  # output training data
        A_test = None,      # input testing data
        Y_test = None,      # output testing data
        k = 10,             # projection space dimension
        epsilon = 0.0,      # 1st stopping criterion (has to be >0 to be used over the second one)
        steps = 150,        # 2nd stoping criterion: max number of iterations
        lambd = 0.1,        # regularization parameter
        mode = "sampling"): # way random matrices are generated 
    '''
    trains the model g, ie. optimizes the parameters ('X' below) and return them
    g should have attributes
    - g.param_count --> integer
    - g.jac --> takes and returns np.array
    - g.forward --> takes and returns np.array
    '''
    assert A.shape[0]==Y.shape[0], "A and y don't have the same number of observations in function train"
    m, l = Y.shape[0], Y.shape[1]  # data set size and output dimension
    # initialization of model's parameters 
    assert X0.size==g.param_count, "initial parameters size and model's parameters size don't match"
    X = X0
    # to track optimization progress
    train_errors = []
    test_errors = []
    # core optimization loop now
    if epsilon == 0.0:
        # second stopping crietrion here
        for _ in tqdm(range(steps)):
            # random generation of S_t 
            S = random_matrix(k,l,mode=mode)
            # x_new = argmin SUM_i ...
            # --> x_new satisfies A_matrix @ x_new = b_vector
            # we derive x_new here
            A_matrix = lambd*np.identity(X.size)
            b_vector = lambd*X
            for i in range(m):
                Ji = g.jac(A[i,:], X)        # jacobian of model's parameters for data point i
                SJ = np.matmul(S, Ji)           # SJ = S_t J_ti, S_t projection matrix, J_ti Jacobian
                A_matrix += np.matmul(SJ.T, SJ)
                residual = g.forward(A[i,:], X) - np.matmul(Ji, X) - Y[i,:]
                b_vector += np.matmul(SJ.T, np.matmul(S, residual))
            x_new = solve(A_matrix, b_vector)
            X = x_new  # x is updated 
            train_mse = MSE(g, X, A, Y)
            train_errors.append(train_mse)
            print("Train error: ", train_mse )
            if A_test is not None: 
                test_mse = MSE(g, X, A_test, Y_test)
                test_errors.append(test_mse)
                print("Test error: ", test_mse)
        return X, train_errors, test_errors
    else:
        # first stopping crietrion here
        dist = epsilon + 1.0
        while dist > epsilon:
            # random generation of S_t 
            S = random_matrix(k,l,mode=mode)
            # x_new = argmin SUM_i ...
            # --> x_new satisfies A_matrix @ x_new = b_vector
            # we derive x_new here
            A_matrix = lambd*np.identity(X.size)
            b_vector = lambd*X
            for i in range(m):
                Ji = g.jac(A[i,:], X)        # jacobian of model's parameters for data point i
                SJ = np.matmul(S, Ji)           # SJ = S_t J_ti, S_t projection matrix, J_ti Jacobian
                A_matrix += np.matmul(SJ.T, SJ)
                residual = g.forward(A[i,:], X) - np.matmul(Ji, X) - y[i,:]
                b_vector += np.matmul(SJ.T, np.matmul(S, residual))
            x_new = solve(A_matrix, b_vector)
            # compute the L2 norm of x_new - x
            dist = np.linalg.norm(x_new - X, 2)
            X = x_new  # x is updated 
            train_mse = MSE(g, X, A, Y)
            train_errors.append(train_mse)
            print("Train error: ", train_mse )
            if A_test is not None: 
                test_mse = MSE(g, X, A_test, Y_test)
                test_errors.append(test_mse)
                print("Test error: ", test_mse)
        return X, train_errors, test_errors



if __name__ == "__main__":
    # Script to test that everything works fine

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
    X_est = train_1(nn, nn.flatten(X0), A, Y, k=1)
    ## Print results
    for i in range(N):
        a = A[i, :].reshape((n, 1))
        y_true = nn.forward(a, nn.flatten(X_true)).flatten()
        y_est = nn.forward(a, X_est)
        print("Y_true", y_true, "Y_est", y_est)

