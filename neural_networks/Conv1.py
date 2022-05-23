import torch
import torch.nn.functional as F
import torch.nn as nn





class Conv1(nn.Module):
    def __init__(self):
        super(Conv1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 10, kernel_size=5)
        self.fc1 = nn.Linear(1000, 1)

    def forward(self, x):
        try:
            x = x.view(1, 28, 28)
        except Exception as e:
            x = x.view(-1, 1,28, 28)
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        #x = F.dropout(x, p=0.5, training=self.training)

        x = x.view( -1, 1,1000 )
        x = self.fc1(x)
        x = x.view(-1, 1)
        return F.sigmoid(x)