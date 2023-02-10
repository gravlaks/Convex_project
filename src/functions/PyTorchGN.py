from torch.autograd import Variable, grad
from torch.autograd.functional import jacobian
import numpy as np
import torch
import threading 

class NN_GN:
    def __init__(self, nn, is_linear=True):
        self.nn = nn
        self.param_count = self.get_X().shape[0]
        self.is_linear =is_linear
    def forward(self,a, X ):
        
        return self.__torch_forward(a, X).detach().numpy()
   
    def __replace_params(self, X):
        left = 0
        for W in self.nn.parameters():
            dim = len(W.flatten())
            W_shape = W.shape
            W.data = Variable(
            torch.tensor(X[left:left+dim].reshape(W_shape), dtype=torch.float32)
                , requires_grad=True)
            left += dim
    def __torch_forward(self, a, X):
        self.__replace_params(X)
        if not self.is_linear:
            pass
            #a = a.reshape((-1, a.shape[-1]))
        
        y_pred = self.nn.forward(torch.tensor(a))
        return y_pred.flatten()
        
    def jac(self, a, X):
        self.__replace_params(X)
        y_pred = self.__torch_forward(a, X)
        Jac = np.zeros((y_pred.shape[0],len(X)))
    
        # rows = [np.hstack([r.flatten().detach().numpy() for r in
        #     torch.autograd.grad(y, list(self.nn.parameters()), retain_graph=True)]) for y in y_pred]
        # Jac = np.vstack(rows)
        # return Jac

        # threads = []
        # rows = []
        # def func(y, rows):

        #     row = torch.autograd.grad(y, list(self.nn.parameters()), retain_graph=True)
        #     row = np.hstack([r.flatten().detach().numpy() for r in row])
        #     sem.acquire()

        #     rows.append(row)
        #     sem.release()

        # sem = threading.Semaphore()  
        # for i, y in enumerate(y_pred):
        #     # print("y", y)
        #     p = threading.Thread(target=func, args=(y, rows))
        #     p.start()
        #     threads.append(p)
        # #print("Threads count", len(threads))
        # for p in threads:
        #     p.join()
        # #print(rows)
        # Jac = np.vstack(rows)
        # return Jac
        for i, y in enumerate(y_pred):
            row = torch.autograd.grad(y, list(self.nn.parameters()), retain_graph=True)
            row = np.hstack([r.flatten().detach().numpy() for r in row])
            Jac[i] = row
        return Jac

    def get_X(self):
        X = []
        params = list(self.nn.parameters())
        for W in params:
            X.append(W.detach().numpy().flatten())
        X = np.hstack(X)
        return X


