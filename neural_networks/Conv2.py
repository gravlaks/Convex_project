import torch.nn.functional as F
import torch.nn as nn

class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 8, kernel_size=5)
        self.conv3 = nn.Conv2d(8,8, kernel_size=5)
        self.fc1 = nn.Linear(3*3*8, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        try:
            x = x.view(1, 28, 28)
        except Exception as e:
            x = x.view(1, 28, 28)

            print(e)
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = x.view(-1,3*3*8 )
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)