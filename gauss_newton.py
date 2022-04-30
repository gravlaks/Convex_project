from cgi import test
import torch
from torch.autograd.functional import jacobian
import numpy as np

def func(x):
    return x**2 + 3*x


def test_autograd():
    input = torch.tensor([3.])

    expected_output = 2*input + 3

    output = jacobian(func, input)
    print("Jacobians Tensor:", output)
    print("Expected output", expected_output)

test_autograd()

def squared_loss(g, y, X, a):
    return torch.linalg.norm(g(X, a)-y)**2

def g(X, a):
    return X@a
X_t = torch.eye(2)
a = torch.ones((2,))
a.requires_grad = False
input = (X_t, a)
output_g = torch.tensor(g(X_t, a))

J_g = jacobian(g, input)
print(J_g)



def squared_loss_grad(g, y, X, a):
    return g(X, a)-y

def squared_loss_hess(g, y, X, a):
    return 
