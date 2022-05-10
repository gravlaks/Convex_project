import numpy as np
from scipy.sparse.linalg import lsmr
from torch.autograd import Variable
import torch
from modular_nn import NN, get_initial_params
from tqdm import tqdm

def get_block(nn, X_t, a, y):
    """
    Retrieves Jacobian for single datapoint and 
    calculates one block of the least squares objective:

    ||g(x_t, ai) + J(x, ai)delta_x  - yi||_2^2
    ||J(x, ai)delta_x - (g(x_t, ai)+yi)||_2^2 
    
    ||Aidelta_x -bi||_2^2

    returns Ai, bi
    
    """
    g = nn.forward(a, X_t)
    jac = nn.jac(a, X_t)

    return jac, -g.flatten() + y.flatten()

def take_step(nn, X_t, A, Y, lambd):
    """
    Take one step in least squares direction
    Line 4-5 in Ergen
    """
    N, m = Y.shape
    N, n = A.shape
    A_ls = np.zeros((len(Y[0])*len(Y),X_t.shape[0] ))
    b_ls = np.zeros((len(Y[0])*len(Y),1))
    output_dim = len(Y[0])
    for i, (a, y) in enumerate(zip(A, Y)):
        y = y.reshape((m, 1))
        a = a.reshape((n, 1))
        A_bl, b_bl = get_block(nn, X_t, a, y)
        A_ls[i*output_dim:(i+1)*output_dim, :] = A_bl
        b_ls[i*output_dim:(i+1)*output_dim, :] = b_bl.reshape((-1, 1))
    delt = lsmr(A_ls, b_ls, damp=np.sqrt(lambd))[0]
    
    X_upd = X_t.flatten()+delt.flatten()
    
    return X_upd

def optimize(g, X0,  A, Y, lambd=0.0001, epsilon = 1e-5):
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

    X_t = X0
    MAX_ITER = 100
    for k in tqdm(range(MAX_ITER)):
        X_tm1 = np.copy(X_t)
        X_t = take_step(g, X_t, A, Y, lambd)

        if np.linalg.norm(X_tm1-X_t) <= epsilon:
            break

    return X_t

if __name__ == '__main__':
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
    
    ## Run algorithm 2
    X_est = optimize(nn, nn.flatten(X0), A, Y)



    ## Print results
    print("X_true", nn.flatten(X_true))
    print("X_est", X_est)
    for i in range(N):
        a = A[i, :].reshape((n, 1))
        y_true = nn.forward(a, nn.flatten(X_true)).flatten()
        y_est = nn.forward(a, X_est)
        print("Y_true", y_true, "Y_est", y_est)
