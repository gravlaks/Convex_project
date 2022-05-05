import torch
from torch.autograd import Variable, grad
import numpy as np



class NN():
    def __init__(self,X0):
        Ws, bs = X0
        variable_count = 0
        self.X0 = X0
        for W, b in zip(Ws, bs):
            variable_count += b.shape[0]
            variable_count += W.shape[0]*W.shape[1]
        self.variable_count = variable_count
    def forward(self, a, X):
        Ws, bs = X
        out = a
        for W, b in zip(Ws, bs):
            out = torch.nn.Sigmoid()(W@out+b)
        return out
    
    def jac(self, y_pred, X):
        Ws, bs = X

        output_dim = y_pred.shape[0]
        Jac = torch.zeros((output_dim, self.variable_count ))

        for i, y in enumerate(y_pred):
            left = 0

            for W, b in zip(Ws, bs):
                del_y_del_w = grad(y, W, retain_graph=True)[0].flatten()

                del_y_del_b = grad(y, b, retain_graph=True)[0].flatten()
                dim = del_y_del_w.shape[0]
                dim_r = del_y_del_b.shape[0]

                Jac[i, left:left+dim] = del_y_del_w

                left += dim
                Jac[i, left:left+dim_r] = del_y_del_b
                left += dim_r

        return Jac
    
    def flatten(self, X):
        Ws, bs = X
        flattened = torch.zeros((self.variable_count))
        left = 0
        for W, b in zip(Ws, bs):
            
            dim = W.flatten().shape[0]
            dim_r = b.flatten().shape[0]

            flattened[left:left+dim] = W.flatten()


            left += dim
            flattened[left:left+dim_r] = b.flatten()
            left += dim_r


        return flattened
    
    def unflatten(self, X_flat):
        Ws0, bs0 = self.X0
        Ws, bs = [], []
        left = 0
        
        for w0, b0 in zip(Ws0, bs0):
            len_w = len(w0.flatten())
            len_b = len(b0.flatten())
            
            Ws.append(X_flat[left:left+len_w].reshape((w0.shape)))
            left = left+len_w
            bs.append(X_flat[left:left+len_b].reshape((b0.shape)))
            left = left+len_b
            

        return (Ws, bs)




    
if __name__ == '__main__':
    torch.manual_seed(1)
    x = torch.randn((4,1))#([[0.1, 0.2, 0.3]])

    output_dim = 2
    input_dim = x.shape[0]
    Ws = [Variable(torch.randn(output_dim, input_dim), requires_grad=True),
        Variable(torch.randn(output_dim, output_dim), requires_grad=True)]
    bs = [Variable(torch.randn(output_dim, 1), requires_grad=True) for _ in range(len(Ws))]
    nn = NN(Ws, bs)

    y_pred = nn.forward(x)
    jac = nn.jac(y_pred)
    print(jac)