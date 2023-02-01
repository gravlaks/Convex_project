import numpy as np
from scipy.sparse.linalg import lsmr
from torch.autograd import Variable
import sys
sys.path.append("../functions")
from tqdm import tqdm
from datetime import datetime, timedelta
import time

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

def get_block_dropout(nn, X_t, a, y, keep_prob):
     

    g = nn.forward(a, X_t)
    jac = nn.jac(a, X_t)
    return jac, -g.flatten() + y.flatten()

def backtrack_step(X_t, A,Y,  A_ls, b_ls, delt, nn,  beta=0.95, alpha = 0.4, maxi=100):
        ## Test on random sample of 300

        def get_val(X):
            random_indices = np.random.choice(len(A),
                                  size=min(150, len(A)),
                                  replace=False)
            val = np.sum(
            np.linalg.norm(nn.forward(A[random_indices], X).reshape((-1, Y.shape[1]))-Y[random_indices], axis=1)**2
        )/2
            return val
        val = get_val(X_t)
        i = 0
        t = 1
        error = get_val(X_t + t*delt)

        while error  > val + alpha*t*(A_ls.T@(A_ls@X_t-b_ls.flatten())).T@delt :
            i+=1
            if i == maxi:
                print("Max back track")
                return np.zeros_like(X_t)
            
            t = beta*t
            next_X = X_t.flatten()+t*delt.flatten()

            error =  get_val(next_X)

        return t*delt.flatten()
    
    

def take_step(nn, X_t, A, Y, lambd, backtrack):
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
        A_bl, b_bl = get_block(nn, X_t, a, y)
        A_ls[i*output_dim:(i+1)*output_dim, :] = A_bl
        b_ls[i*output_dim:(i+1)*output_dim, :] = b_bl.reshape((-1, 1))
    delt = lsmr(A_ls, b_ls, damp=np.sqrt(lambd))[0]
    if backtrack:
        step = backtrack_step(X_t, A,Y, A_ls, b_ls, delt, nn)
    else:
        step = delt.flatten()
    return X_t.flatten()+step



def take_step_gaussian(nn, X_t, A, Y, lambd, backtrack):
    
   
    A_ls = np.zeros((len(Y[0])*len(Y),X_t.shape[0] ))
    b_ls = np.zeros((len(Y[0])*len(Y),1))
    output_dim = len(Y[0])
    #full_jac = get_block(nn, X_t, A, Y)
    
    for i, (a, y) in enumerate(zip(A, Y)):
        A_bl, b_bl = get_block(nn, X_t, a, y)

        A_ls[i*output_dim:(i+1)*output_dim, :] = A_bl
        b_ls[i*output_dim:(i+1)*output_dim, :] = b_bl.reshape((-1, 1))

    k = 100
    S = np.random.randn(k, A_ls.shape[0]) / (A_ls.shape[0])
    delt = lsmr(S@A_ls, S@b_ls, damp=np.sqrt(lambd))[0]

    
    if backtrack:
        step = backtrack_step(X_t, A, Y, A_ls, b_ls, delt, nn)
    else:
        step = delt.flatten()
    return X_t.flatten()+step
    
def take_step_sample_columns(nn, X_t, A, Y, lambd, backtrack):
    """
    Sample columns of matrix

    """
    A_ls = np.zeros((len(Y[0])*len(Y),X_t.shape[0] ))
    b_ls = np.zeros((len(Y[0])*len(Y),1))
    output_dim = len(Y[0])
    keep_prob = 0.99

    indices = np.random.choice(A_ls.shape[1], size=int(A_ls.shape[1]*keep_prob), replace=False)
    X_t_dropout = np.zeros_like(X_t)
    X_t_dropout[indices] = X_t[indices]


    

    t1 = time.time()

    for i, (a, y) in enumerate(zip(A, Y)):
        A_bl, b_bl = get_block_dropout(nn, X_t, a, y, keep_prob)
        A_ls[i*output_dim:(i+1)*output_dim, :] = A_bl
        b_ls[i*output_dim:(i+1)*output_dim, :] = b_bl.reshape((-1, 1))
    t2 = time.time()
    A_ls_sampled = A_ls[:, indices]
    delt_sampled = lsmr(A_ls_sampled, b_ls, damp=np.sqrt(lambd))[0]*keep_prob

    t3 = time.time()
    delt = np.zeros_like(X_t)
    delt[indices] = delt_sampled
    if backtrack:
        step = backtrack_step(X_t, A,Y, A_ls, b_ls, delt, nn)
    else:
        step = delt.flatten()
    t4 = time.time()

    # print(t2-t1)
    # print(t3-t2)
    # print(t4-t3)
    return X_t.flatten()+step




def mse(g, X, A, Y):
    return np.sum(
        #[np.linalg.norm(g.forward(a, X)-y)**2 for a, y in zip(A, Y)]
        np.linalg.norm(g.forward(A, X).reshape((-1, 1))-Y.reshape((-1, 1)), axis=1)**2
        )/A.shape[0]


def optimize(g, X0, A, Y, A_test=None, Y_test=None, lambd=0.1, epsilon = 1e-3, steps=150, max_time=300, batch_size=100, backtrack=True, optimization_method="Random"):
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
    print("Parameter count", g.param_count)
    for k in tqdm(range(MAX_ITER)):
        if datetime.now()-t1>timedelta(seconds=max_time):
            print("timeout")
            break
        #X_tm1 = np.copy(X_t)
        t_start = time.time()

        random_indices = np.random.choice(N,
                                  size=batch_size,
                                  replace=False)
        if optimization_method=="Random":
            random_indices = np.random.choice(N,
                                  size=batch_size,
                                  replace=False)
            X_t = take_step(g, X_t, A[random_indices, :], Y[random_indices, :], lambd, backtrack)
        elif optimization_method == "Gaussian":

            X_t = take_step_gaussian(g, X_t, A[random_indices], Y[random_indices], lambd, backtrack)
        elif optimization_method == "Random columns":


            X_t = take_step_sample_columns(g, X_t, A[random_indices], Y[random_indices], lambd, backtrack)

        else:
            raise NotImplementedError
        train_mse = mse(g,X_t, A[random_indices], Y[random_indices])
        train_errors.append(train_mse)
        if k%10==1:
            print("Train error: ", train_mse )
            print("Epoch time: ", time.time()-t_start)
        if A_test is not None: 
            test_mse = mse(g, X_t, A_test, Y_test)
            test_errors.append(test_mse)
            print("Test error: ", test_mse )
        #if np.linalg.norm(X_tm1-X_t) <= epsilon:
            #break
        
    return X_t, train_errors, test_errors

