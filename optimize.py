import numpy as np
from scipy.sparse.linalg import lsmr
from torch.autograd import Variable
import torch
from modular_nn import NN
from tqdm import tqdm

def get_block(nn, X_t, a, y):
    
    g = nn.forward(a, X_t)
    jac = nn.jac(g, X_t)
    
    #X_flat_t = nn.flatten(X_t)
    return jac, g - y

def take_step(nn, X_t, A, Y, lambd=0.1):
    X_flat_t = nn.flatten(X_t)
    N, m = Y.shape
    N, n = A.shape
    A_ls = torch.zeros((len(Y[0])*len(Y),X_flat_t.shape[0] ))
    b_ls = torch.zeros((len(Y[0])*len(Y),1))
    output_dim = len(Y[0])

    for i, (a, y) in enumerate(zip(A, Y)):
        y = y.reshape((m, 1))
        a = a.reshape((n, 1))
        A_bl, b_bl = get_block(nn, X_t, a, y)
        A_ls[i*output_dim:(i+1)*output_dim, :] = A_bl
        b_ls[i*output_dim:(i+1)*output_dim, :] = b_bl

    delt = torch.linalg.solve(A_ls.T@A_ls+lambd*torch.eye(A_ls.shape[1]), A_ls.T@b_ls)
    #delt = lsmr(A_ls, b_ls, damp = np.sqrt(lambd))
    X_upd = X_flat_t+delt.flatten()
    X_upd = X_upd.reshape((-1, 1))
    return X_upd

def optimize(nn, X0,  A, Y, lambd=0.1, epsilon = 0.5):
   

    X_t = X0
    MAX_ITER = 100
    for k in tqdm(range(MAX_ITER)):
        X_tm1_flat = torch.clone(nn.flatten(X_t))
        X_t = take_step(nn, X_t, A, Y)

        if torch.linalg.norm(X_tm1_flat-X_t) <= epsilon:
            break

        X_t = nn.unflatten(X_t, X0)
    return X_t

if __name__ == '__main__':
    N, n = 40, 2
    m = 2
    Ws0 = [Variable(torch.randn(m, n), requires_grad=True),
        Variable(torch.randn(m, m), requires_grad=True)]
    bs0 = [Variable(torch.randn(m, 1), requires_grad=True) for _ in range(len(Ws0))]



    Ws_true = [Variable(torch.randn(m, n), requires_grad=True),
        Variable(torch.randn(m, m), requires_grad=True)]
    bs_true = [Variable(torch.randn(m, 1), requires_grad=True) for _ in range(len(Ws_true))]
    X_true = (Ws_true, bs_true)
    X0 = (Ws0, bs0)

    nn = NN(X0)

    A = torch.randn(N, n)
    Y = torch.zeros((N, m))
    for i in range(N):
        a = A[i, :].reshape((n, 1))
        y_pred =  nn.forward(a, X_true).flatten()
        Y[i, :] = y_pred
    X_est = optimize(nn, X0, A, Y)
    print("X_true", X_true)
    print("X_est", X_est)