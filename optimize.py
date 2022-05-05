import numpy as np
from scipy.sparse.linalg import lsmr
from torch.autograd import Variable
import torch
from modular_nn import NN, get_initial_params
from tqdm import tqdm

def get_block(nn, X_t, a, y):
    
    g = nn.forward(a, X_t)

    jac = nn.jac(a, X_t)
    #X_flat_t = nn.flatten(X_t)
    if np.any(np.isnan(jac)) or np.any(np.isnan(g)):
        print("nan")
    return jac, -g.flatten() + y.flatten()

def take_step(nn, X_t, A, Y, lambd):
    
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
    if np.any(np.isnan(A_ls)) or np.any(np.isnan(b_ls)):
        
        print("a or b")
        raise Exception
    delt = lsmr(A_ls, b_ls, damp=np.sqrt(lambd))[0]
    #delt = np.linalg.solve(A_ls.T@A_ls+lambd*np.eye(A_ls.shape[1]), A_ls.T@b_ls)
    if np.any(np.isnan(delt)):
        print("delt")
        print(A_ls, b_ls)
        raise Exception
    #delt = lsmr(A_ls, b_ls, damp = np.sqrt(lambd))
    X_upd = X_t.flatten()+delt.flatten()
    
    return X_upd

def optimize(nn, X0,  A, Y, lambd=0.001, epsilon = 0.00005):
   

    X_t = X0
    MAX_ITER = 500
    for k in tqdm(range(MAX_ITER)):
        X_tm1 = np.copy(X_t)
        X_t = take_step(nn, X_t, A, Y, lambd)

        if np.linalg.norm(X_tm1-X_t) <= epsilon:
            break

    return X_t

if __name__ == '__main__':
    N, n = 80, 2
    m = 2
    layer_count = 2
    torch.manual_seed(1)
    Ws0, bs0 = get_initial_params(layer_count, m, n)
    Ws_true, bs_true = get_initial_params(layer_count, m, n)
    X_true = (Ws_true, bs_true)
    X0 = (Ws0, bs0)

    nn = NN(X0)

    A = np.random.randn(N, n)
    A[0, :] = np.ones((n,))
    Y = np.zeros((N, m))
    for i in range(N):
        a = A[i, :].reshape((n, 1))
        y_pred =  nn.forward(a, nn.flatten(X_true)).flatten()
        Y[i, :] = y_pred
    

    X_est = optimize(nn, nn.flatten(X0), A, Y)
    print("X_true", nn.flatten(X_true))
    print("X_est", X_est)

    print(nn.forward(A[1], nn.flatten(X_true)))
    print(nn.forward(A[1], X_est))

    for i in range(N):
        a = A[i, :].reshape((n, 1))
        y_true = nn.forward(a, nn.flatten(X_true)).flatten()
        y_est = nn.forward(a, X_est).flatten()
        print("Y_true", y_true, "Y_est", y_est)
