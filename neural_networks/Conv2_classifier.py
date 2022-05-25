import torch.nn.functional as F
import torch.nn as nn

class Conv2_Classifier(nn.Module):
    def __init__(self):
        super(Conv2_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16,32, kernel_size=5)
        self.fc1 = nn.Linear(3*3*32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        try:
            x = x.view(1, 28, 28)
        except Exception as e:
            x = x.view(-1, 1,28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = x.view(-1,3*3*32 )
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = F.softmax(x, dim=1)
        return out