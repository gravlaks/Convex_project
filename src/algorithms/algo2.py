import numpy as np
from scipy.sparse.linalg import lsmr
from torch.autograd import Variable
import sys
sys.path.append("../functions")
from tqdm import tqdm
from datetime import datetime, timedelta
import time
from utils.plotting import visualize_step

 
    


class Stepper():
    def __init__(self, optimization_method, nn, backtrack, batch_size, optim_params, visualize, A, Y):
        self.optim_params = optim_params
        if optim_params is None:
            self.optim_params = {
                "keep_prov": 1,
                "max_backtrack": 30,
                "momentum": 0}
        

        self.full_A = A
        self.full_Y = Y
        self.optimization_method=optimization_method
        self.nn = nn
        self.backtrack=backtrack
        self.lambd = 1
        self.batch_size = batch_size
        self.visualize = visualize
        self.keep_prob = self.optim_params.get("keep_prob", 1)
        self.max_backtrack = self.optim_params.get("max_backtrack", 30)
        
        self.optim_params["momentum"] = self.optim_params.get("momentum", 0)

        self.backtracks = []

        self.prev_step = np.zeros_like(self.nn.get_X())
        self.first_iteration = True

    def get_block(self, a, y):
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
    

        g = self.nn.forward(np.expand_dims(a, 0), self.X_t)
        #jac = self.nn.jac(np.expand_dims(a, 0), self.X_t)
        #self.jac = jac
        return -g.flatten() + y.flatten()

    def backtrack_step(self, A, Y,  A_ls, b_ls, delt,  beta=0.95, alpha = 0.1, maxi=30):
        

        def get_f(step):
            ret = np.linalg.norm(self.nn.forward(self.full_A, self.X_t + step)-self.full_Y.flatten())**2 # + self.lambd*np.linalg.norm(step)**2
            return ret
        #val = get_val(self.X_t)
        i = 0
        t = 1
        #error = get_val(self.X_t + t*delt)

        old_error = get_f(np.zeros_like(self.X_t))
        new_error = get_f(delt)

        error_vec = self.nn.forward(A, self.X_t + delt)-Y.flatten()
        nabla_f = 2*A_ls.T@error_vec #+ 2*self.lambd*delt

        min_lambd, max_lambd = 1e-15, 500
        lambd_multiplier = 4
        while new_error > old_error + alpha*t*(nabla_f).T@delt :
            i+=1

            
            if i == maxi:
                print("Max back track")
                if self.lambd<max_lambd:
                    self.lambd*=lambd_multiplier
                self.backtracks.append(i)
                return np.zeros_like(self.X_t)
                break
                #return np.zeros_like(self.X_t)
            
            t = beta*t
            step = t*delt.flatten()

            new_error =  get_f(step)
        print(i)
        self.backtracks.append(i)

        if i < 1 and self.lambd>min_lambd:
            self.lambd/=lambd_multiplier
        elif self.lambd < max_lambd:
            self.lambd*=lambd_multiplier

        
        return t*delt.flatten()  

    def take_step(self, X_t, A, Y):
        self.X_t = X_t
        A_ls = np.zeros((len(Y[0])*len(Y), self.X_t.shape[0] ))
        b_ls = np.zeros((len(Y[0])*len(Y),1))
        output_dim = len(Y[0])



        ## For random columnsØ
        self.X_t_dropout = self.X_t.copy()
        indices = np.random.choice(A_ls.shape[1], size=int(A_ls.shape[1]*self.keep_prob), replace=False)

        # if self.optimization_method == "Random columns":
        #     self.X_t_dropout = np.zeros_like(self.X_t)
        #     self.X_t_dropout[indices] = self.X_t[indices]
        t1 = time.time()
        g = self.nn.forward(A, self.X_t_dropout)
        #jac = self.nn.jac(A, self.X_t)
        #A_ls = jac
        b_ls = -g.flatten() + Y.flatten()
        for i, (a, y) in enumerate(zip(A, Y)):

            #A_bl, b_bl = self.get_block(a, y)
            A_bl = self.nn.jac(np.expand_dims(a, 0), self.X_t_dropout)
            A_ls[i*output_dim:(i+1)*output_dim, :] = A_bl
            #b_ls[i*output_dim:(i+1)*output_dim, :] = b_bl.reshape((-1, 1))

        t2 = time.time()
        if self.optimization_method=="Gaussian":
            k = int(self.batch_size*0.85)
            S = np.random.randn(k, A_ls.shape[0]) / (A_ls.shape[0])
            delt = lsmr(S@A_ls, S@b_ls, damp=np.sqrt(self.lambd), atol=1e-4)[0]

        elif self.optimization_method == "Random columns":
            
            A_ls_sampled = A_ls[:, indices]
            delt_sampled =  lsmr(A_ls_sampled, b_ls, damp=np.sqrt(self.lambd), atol=1e-4)[0]#*self.keep_prob
            delt = np.zeros_like(X_t)
            delt[indices] = delt_sampled
        else:
            delt = lsmr(A_ls, b_ls, damp=np.sqrt(self.lambd), atol=1e-4)[0]

        c = 1#e-1
        delt = c*delt

        if self.first_iteration:
            self.first_iteration = False
        else:
            delt = self.optim_params["momentum"]*self.prev_step + (1-self.optim_params["momentum"])*delt
        
        t3 = time.time()
        if self.backtrack:
            step = self.backtrack_step(A, Y, A_ls, b_ls, delt)
        else:
            step = delt.flatten()

        ## Momentum

        self.prev_step = step;
        t4 = time.time()
        jac_creation = t2-t1
        ls_solve = t3-t2
        backtrack = t4-t3

        if self.visualize:
            visualize_step(step)


        return X_t.flatten()+step, jac_creation, ls_solve, backtrack


def mse(g, X, A, Y):
    Y_pred = g.forward(A, X).reshape((-1, 1))
    mse = ((Y_pred.flatten()- Y.flatten())**2).mean()
    return mse
    # return np.sum(
    #     np.linalg.norm()-Y.reshape((-1, 1)), axis=1)**2
    # )/A.shape[0]


def optimize(g, X0, A, Y, A_test=None, Y_test=None, max_time=300, 
                batch_size=100, backtrack=True, optimization_method="Random",
                optim_params=None, visualize=False):
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
    N = A.shape[0]
    t1 = datetime.now()

    print("Parameter count", g.param_count)
    
    stepper = Stepper(optimization_method, g, backtrack, batch_size=batch_size, optim_params=optim_params, visualize=visualize, A=A, Y=Y)
    k = 0
    timer = {"Jac creation": [], "LS solve": [], "backtrack": []}
    train_mse = mse(g,X_t, A, Y)
    train_errors.append(train_mse)
    while True:
        if datetime.now()-t1>timedelta(seconds=max_time):
            print("timeout")
            break
        #X_tm1 = np.copy(X_t)
        t_start = time.time()

        random_indices = np.random.choice(N,
                                  size=batch_size,
                                  replace=False)
        X_t, jac_creation_time, ls_solve, backtrack = stepper.take_step(X_t, A[random_indices, :], Y[random_indices, :])

        timer["Jac creation"].append(jac_creation_time)
        timer["LS solve"].append(ls_solve)
        timer["backtrack"].append(backtrack)

        train_mse = mse(g,X_t, A, Y)
        train_errors.append(train_mse)
        if k%3==1:
            print("Train error: ", train_mse, "lambd: ", stepper.lambd )
            print("Epoch time: ", time.time()-t_start)
        if A_test is not None: 
            test_mse = mse(g, X_t, A_test, Y_test)
            test_errors.append(test_mse)
            print("Test error: ", test_mse )
        #if np.linalg.norm(X_tm1-X_t) <= epsilon:
            #break
        k+=1


    
    return X_t, train_errors, test_errors, timer, stepper.backtracks

