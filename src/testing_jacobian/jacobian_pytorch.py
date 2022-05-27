import torch
from torch.autograd import Variable, grad
import numpy as np
torch.manual_seed(1)
import torch.nn.functional as F
import torch.nn as nn


x = torch.randn((28*28,))#([[0.1, 0.2, 0.3]])

class Net(nn.Module):

    def __init__(self,  input_dim, output_dim):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
    
        self.fc1 = nn.Linear(input_dim, 10)  # 5*5 from image dimension
        self.fc2 = nn.Linear(10, output_dim)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

if __name__=='__main__':
    net = Net(28*28, 2)
    print(net)
    y_pred = net(x)
    Jac = []


    for name, W in net.named_parameters():
        print(W)
    #print(Jac.shape)