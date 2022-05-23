
import numpy as np
from scipy.sparse.linalg import lsmr
from torch.autograd import Variable
import torch
import sys
sys.path.append("../functions")
from tqdm import tqdm
from datetime import datetime, timedelta

def get_block(nn, X_t, a, y):
    """
    Retrieves Jacobian for single datapoint and 
    calculates one block of the least squares objective:

    ||g(x_t, ai) + J(x_t, ai)delta_x  - yi||_2^2

    ||J(x, ai)delta_x - (g(x_t, ai)+yi)||_2^2 
    
    ||Aidelta_x - bi||_2^2

    returns Ai, bi

    A_i = m x param_count 
    b_i = m x 1
    
    """
    g = nn.forward(a, X_t)
    jac = nn.jac(a, X_t)

    return jac, -g.flatten() + y.flatten()

def take_step(nn, X_t, A, Y, lambd, step_size):
    """
    Take one step in least squares direction
    Line 4-5 in Ergen

    Solving : ||A_ls delta_x -b||_2^2 + lambd||delta_x||_2^2

    ||g(x, A) + Jdelta_x - y||

    = 
    ||g(x, a_1)+J_2delta_x - y_1||
      g(x, a_2)+J_2delta_x - y_2
      ...
      g(x, a_N)+J_Ndelta_x - y_2

    """

    A_ls = np.zeros((len(Y[0])*len(Y),X_t.shape[0] ))
    b_ls = np.zeros((len(Y[0])*len(Y),1))
    output_dim = len(Y[0])

    for i, (a, y) in enumerate(zip(A, Y)):
        #y = y.reshape((m, 1))
        #a = a.reshape((n, 1))
        A_bl, b_bl = get_block(nn, X_t, a, y)
        A_ls[i*output_dim:(i+1)*output_dim, :] = A_bl
        b_ls[i*output_dim:(i+1)*output_dim, :] = b_bl.reshape((-1, 1))
    delt = lsmr(A_ls, b_ls, damp=np.sqrt(lambd))[0]
    alpha = 0.4
    val = np.sum(
        #[np.linalg.norm(g.forward(a, X)-y)**2 for a, y in zip(A, Y)]
        np.linalg.norm(nn.forward(A, X_t).reshape((-1, 1))-Y, axis=1)**2
    )/2
    print(val)
    ## Line search: 
    t = 1
    next_X = X_t.flatten()+t*delt.flatten()
    error =  np.sum(
        #[np.linalg.norm(g.forward(a, X)-y)**2 for a, y in zip(A, Y)]
        np.linalg.norm(nn.forward(A, next_X).reshape((-1, 1))-Y.reshape((-1, 1)), axis=1)**2
    )/2
    i = 0
    max_back_track = 40
    beta = 0.95
    while error  > val + alpha*t*(A_ls.T@(A_ls@X_t-b_ls.flatten())).T@delt :
        i+=1
        if i == max_back_track:
            print("Max back track")
            break
        t = beta*t
        next_X = X_t.flatten()+t*delt.flatten()

        error =  np.sum(
        #[np.linalg.norm(g.forward(a, X)-y)**2 for a, y in zip(A, Y)]
        np.linalg.norm(nn.forward(A, next_X).reshape((-1, 1))-Y.reshape((-1, 1)), axis=1)**2
        )/2
    X_upd = X_t.flatten()+t*delt.flatten()
    
    return X_upd
def mse(g, X, A, Y):
    return np.sum(
        #[np.linalg.norm(g.forward(a, X)-y)**2 for a, y in zip(A, Y)]
        np.linalg.norm(g.forward(A, X).reshape((-1, 1))-Y.reshape((-1, 1)), axis=1)**2
        )/A.shape[0]

def optimize(g, X0,  A, Y, lambd=0.1, epsilon = 1e-3, step_size=0.999, X_test=None, Y_test=None, steps=150, max_time=300, batch_size=100):
    """
    g: generic class that provides a forward function and jacobian function
    X0 : initial parameter guess for parameters in g
    A (N, n): data
    Y (N, m): labels
    lambd: regression parameter for Gauss Newton
    epsilon: tolerance for when to quit iteration

    Algorithm 2 (currently without projection)
    Gauss Newton for training Neural network
    """
    train_errors = []
    test_errors = []
    X_t = X0
    MAX_ITER = steps
    N = A.shape[0]
    t1 = datetime.now()
    for k in tqdm(range(MAX_ITER)):
        if datetime.now()-t1>timedelta(seconds=max_time):
            print("timeout")
            break
        X_tm1 = np.copy(X_t)
        #np.random.seed(0)
        random_indices = np.random.choice(N,
                                  size=batch_size,
                                  replace=False)
        X_t = take_step(g, X_t, A[random_indices, :], Y[random_indices, :], lambd, step_size)

        train_mse = mse(g,X_t, A, Y)
        train_errors.append(train_mse)
        print("Train error:", train_mse )

        if X_test is not None: 
            test_mse = mse(g, X_t, X_test, Y_test)
            test_errors.append(test_mse)
            print("Test error:", test_mse )


        #if np.linalg.norm(X_tm1-X_t) <= epsilon:
            #break

    return X_t, train_errors, test_errors

