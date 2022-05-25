import torch
import torch.nn.functional as F
import torch.nn as nn


class BacktrackFC(nn.Module):

    def __init__(self,  input_dim, output_dim):
        super(BacktrackFC, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
    
        self.fc1 = nn.Linear(input_dim, 60)  # 5*5 from image dimension
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 15)
        self.fc4 = nn.Linear(15, 5)
        self.fc5 = nn.Linear(5, output_dim)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x