import torch
from torch.autograd import Variable, grad
import numpy as np
torch.manual_seed(1)
x = torch.randn((1, 4))#([[0.1, 0.2, 0.3]])
y_true = torch.Tensor([[0, 1]])

output_dim = 2
input_dim = x.shape[1]

W = Variable(torch.randn(output_dim, input_dim), requires_grad=True)
b = Variable(torch.randn(output_dim, 1), requires_grad=True)

y_pred = torch.nn.Sigmoid()(W@x.T+b)
n = x.shape[1]
print(y_pred)

variable_count = (n+1)*output_dim

T = np.zeros((output_dim, (n+1)*output_dim))
print(T.shape, y_pred.shape)
print(output_dim)
for i in range(output_dim):
    vec = torch.hstack((x, torch.tensor([[1.]]))).flatten()
    print(i, vec)
    length = vec.shape[0]
    T[i, i*(length): (i+1)*(length)]= vec
y_pred_np = y_pred.detach().numpy()
Jac = y_pred_np*(1-y_pred_np)*T
print(Jac)

Jac = torch.zeros((output_dim, variable_count ))
print("delYdelW", grad(y_pred[0], W, retain_graph=True))
print(grad(y_pred[0], b, retain_graph=True))

for i, y in enumerate(y_pred):
    del_y_del_w = grad(y, W, retain_graph=True)[0].flatten()

    del_y_del_b = grad(y, b, retain_graph=True)[0].flatten()
    dim = del_y_del_w.shape[0]
    Jac[i, :dim] = del_y_del_w
    Jac[i, dim:] = del_y_del_b
print(Jac)
    