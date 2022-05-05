import torch
from torch.autograd import Variable, grad
import numpy as np
from modular_nn import get_initial_params
torch.manual_seed(1)

output_dim = 2
input_dim = 2

Ws, bs = get_initial_params(1, output_dim, input_dim)
Ws_true, bs_true = get_initial_params(1, output_dim, input_dim)

W = Ws[0]
b = bs[0]
x = torch.ones((1, input_dim))#([[0.1, 0.2, 0.3]])

y_pred = torch.nn.Sigmoid()(Ws[0]@x.T+bs[0])
print("y_pred", y_pred)
n = x.shape[1]

variable_count = (n+1)*output_dim

T = np.zeros((output_dim, (n+1)*output_dim))

for i in range(output_dim):
    vec = torch.hstack((x, torch.tensor([[1.]]))).flatten()
    length = vec.shape[0]
    T[i, i*(length): (i+1)*(length)]= vec
y_pred_np = y_pred.detach().numpy()
Jac = y_pred_np*(1-y_pred_np)*T
print("True jac", Jac)

Jac = torch.zeros((output_dim, variable_count ))

for i, y in enumerate(y_pred):
    del_y_del_w = grad(y, Ws[0], retain_graph=True)[0].flatten()

    del_y_del_b = grad(y, bs[0], retain_graph=True)[0].flatten()
    dim = del_y_del_w.shape[0]
    Jac[i, :dim] = del_y_del_w
    Jac[i, dim:] = del_y_del_b
print(Jac)
    