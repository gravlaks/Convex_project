from torch.autograd import Variable, grad
import numpy as np
import torch

class NN_GN:
    def __init__(self, nn):
        self.nn = nn
        self.param_count = self.get_X().shape[0]

    def forward(self,a, X ):
        
        return self.__torch_forward(a, X).detach().numpy()
    def __replace_params(self, X):
        left = 0
    
        for name, W in self.nn.named_parameters():
                dim = len(W.flatten())
                W_shape = W.shape
                W.data = Variable(
                torch.tensor(X[left:left+dim].reshape(W_shape), dtype=torch.float32)
                    , requires_grad=True)
                left += dim
    def __torch_forward(self, a, X):
        self.__replace_params(X)
        y_pred = self.nn.forward(torch.tensor(a.squeeze()))
        return y_pred.flatten()
        
    def jac(self, a, X):
        
        y_pred = self.__torch_forward(a, X)
        Jac = []
        for y in y_pred:
            row = []
            for name, W in self.nn.named_parameters():
                del_y_del_w = grad(y, W, retain_graph=True)[0].flatten().detach().numpy()

                row.append(del_y_del_w.flatten())
            row = np.hstack(row)
            Jac.append(row)
        Jac = np.vstack(Jac)
        return Jac
    def get_X(self):
        X = []
       
        for name, W in self.nn.named_parameters():
            X.append(W.detach().numpy().flatten())
        X = np.hstack(X)
        return X


