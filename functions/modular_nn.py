import torch
from torch.autograd import Variable, grad
import numpy as np



class NN():
    """
    Simple multi-layered network of the following structure: 
    Activation function currently sigmoid
    input data a:
    x = a 
    for each layer_i
        x = activation_function(W_i@x + b_i)
    y = x

    The class provides most importantly a 
    - forward function for propagating data through the 
    network
    - jac function for return the jacobian with respect to the parameters
    Both functions expect numpy arrays. 
    """
    def __init__(self,X0):
        """
        X0 tuple(Ws, bs): 
        Ws: list of weight matrices W in neural network
        bs: list of biases b in neural network
        
        initializes param_count: number of parameters in neural network
        """
        Ws, bs = X0
        param_count = 0
        self.X0 = X0
        for W, b in zip(Ws, bs):
            param_count += b.shape[0]
            param_count += W.shape[0]*W.shape[1]
        self.param_count = param_count

    
    def forward(self, a, X):
        """
        a data_row: (n, ) or (n, 1)
        X np.ndarray: (param_count,)

        Inputs the parameters of the neural network and a data point
        Outputs the output of the neural network
        """
        
        X_ = self.unflatten(X)
        out = self.__torch_forward(a, X_)
        out = out.detach().numpy()
        return out.flatten()
    
    def __torch_forward(self, a, X):
        """
        a data_row: (n, ) or (n, 1)
        X tuple(Ws, bs): 
        Ws: list of weight matrices W in neural network (torch tensors)
        bs: list of biases b in neural network (torch tensors)

        Given a tuple of parameters and data, returns output from neural network 
        """
        Ws, bs = X
        out = torch.tensor(a, dtype=torch.float32)
        for W, b in zip(Ws, bs):
            b = b.reshape((-1, 1))
            out = out.reshape((-1, 1))
            out = W@out + b
            out = torch.nn.Sigmoid()(out)

        return out
    def get_projected_J(self, a, X, S):
        """
        a data_row: (n, ) or (n, 1)
        X np.array: (param_count,)
        S np.ndarray: (K, param_count)
        Given data, parameters and a projection matrix S, returns The projected Jacobian 
        """
        return S@self.jac(a, X)
        
    def jac(self, a, X):
        """
        a data_row: (n, ) or (n, 1)
        X np.array: (param_count,)

        Returns jacobian of Neural network with respect to the parameters for a given data point: 
        I.e. del NN(X, a) / del X
        """
        X_ = self.unflatten(X)
        y_pred = self.__torch_forward(a,X_)
        Ws, bs = X_
        output_dim = y_pred.shape[0]
        Jac = torch.zeros((output_dim, self.param_count ))

        """
        As pytorch can only handle scalars, we iterate through each element of the output
        and find the gradient of the scalar with respect to our parameters.
        """
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
        return Jac.detach().numpy()
    
    def flatten(self, X):
        """
        X tuple(Ws, bs): 
        Ws: list of weight matrices W in neural network (torch tensors)
        bs: list of biases b in neural network (torch tensors)

        Convert list of weight matrices and biases to a 1D 
        numpy array
        """
        Ws, bs = X

        flattened = torch.zeros((self.param_count))
        left = 0
        for W, b in zip(Ws, bs):
            
            dim = W.flatten().shape[0]
            dim_r = b.flatten().shape[0]

            flattened[left:left+dim] = W.flatten()


            left += dim
            flattened[left:left+dim_r] = b.flatten()
            left += dim_r
        


        return flattened.detach().numpy()
    
    def unflatten(self, X_flat):
        """
        X_flat np.array: (param_count,)

        Recover list of weights and biases from 1D np array.
        """
        Ws0, bs0 = self.X0
        Ws, bs = [], []
        left = 0
        
        for w0, b0 in zip(Ws0, bs0):
            len_w = len(w0.flatten())
            len_b = len(b0.flatten())
            
            Ws.append(Variable(
                torch.tensor(X_flat[left:left+len_w].reshape((w0.shape)), dtype=torch.float32)
                , requires_grad=True))
            left = left+len_w
            bs.append(Variable(torch.tensor(X_flat[left:left+len_b].reshape((b0.shape)), dtype=torch.float32)
                , requires_grad=True))
            left = left+len_b
            

        return (Ws, bs)


def get_initial_params(hidden_layer_count, m, n, hidden_neurons):
    """
    hidden_layer_count (int): number of hidden layers 
    m (int): output dimension 
    n (int): input dimension
    hidden_neurons (int) : neurons per hidden layer
    """
    if hidden_layer_count == 0:
        Ws0 = [Variable(torch.zeros((m, n)), requires_grad=True)]
        bs0 = [Variable(torch.zeros((m, 1)), requires_grad=True)]
        return Ws0, bs0
    else:
        
        Ws0 = [Variable(torch.zeros((hidden_neurons, n)), requires_grad=True)]
        bs0 = [Variable(torch.zeros((hidden_neurons, 1)), requires_grad=True)]
    for i in range(hidden_layer_count-1):
        w_var = Variable(torch.randn(hidden_neurons, hidden_neurons), requires_grad=True)
        b_var = Variable(torch.randn(hidden_neurons, 1), requires_grad=True)
        Ws0.append(w_var)
        bs0.append(b_var)
    Ws0.append(Variable(torch.randn(m, hidden_neurons), requires_grad=True))
    bs0.append(Variable(torch.randn(m, 1), requires_grad=True))

    return Ws0, bs0

    
if __name__ == '__main__':
    torch.manual_seed(1)
    x = torch.randn((4,1))#([[0.1, 0.2, 0.3]])

    output_dim = 2
    input_dim = x.shape[0]
    X = get_initial_params(2, output_dim, input_dim, 5)
    nn = NN(X)

    y_pred = nn.forward(x, nn.flatten(X))
    jac = nn.jac(x, nn.flatten(X))
    print(jac)