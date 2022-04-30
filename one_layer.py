import torch
from torch.autograd import Variable
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



T = np.zeros((output_dim, (n+1)*output_dim))
print(T.shape, y_pred.shape)
print(output_dim)
for i in range(output_dim):
    vec = torch.hstack((x, torch.tensor([[1.]]))).flatten()
    print(i, vec)
    length = vec.shape[0]
    T[i, i*(length): (i+1)*(length)]= vec
y_pred = y_pred.detach().numpy()
Jac = y_pred*(1-y_pred)*T
print(Jac)

