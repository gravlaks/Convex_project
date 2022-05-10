import torch
from torch.autograd import Variable, grad
import numpy as np
torch.manual_seed(1)
x = torch.randn((4,1))#([[0.1, 0.2, 0.3]])

output_dim = 2
input_dim = x.shape[0]
layers_cnt = 2
Ws = [Variable(torch.randn(output_dim, input_dim), requires_grad=True),
    Variable(torch.randn(output_dim, output_dim), requires_grad=True)]
bs = [Variable(torch.randn(output_dim, 1), requires_grad=True) for _ in range(layers_cnt)]

def func(x):
    out = x
    for W, b in zip(Ws, bs):
        print(W.shape, out.shape, b.shape)
        out = torch.nn.Sigmoid()(W@out+b)
        print(out.shape)
    return out

print(Ws)
n = x.shape[1]
y_pred = func(x)
print("y_pred", y_pred)
variable_count = 0
for W, b in zip(Ws, bs):
    variable_count += b.shape[0]
    variable_count += W.shape[0]*W.shape[1]
print("Variable count", variable_count)
T = np.zeros((output_dim, variable_count))
print(T.shape, y_pred.shape)
print(output_dim)
for i in range(output_dim):
    vec = torch.vstack((x, torch.tensor([[1.]]))).flatten()
    print(i, vec)
    length = vec.shape[0]
    #T[i, i*(length): (i+1)*(length)]= vec
#y_pred_np = y_pred.detach().numpy()
#Jac = y_pred_np*(1-y_pred_np)*T
#print(Jac)

Jac = torch.zeros((output_dim, variable_count ))
#print("delYdelW", grad(y_pred[0], W, retain_graph=True))
#print(grad(y_pred[0], b, retain_graph=True))

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
print(Jac)
    