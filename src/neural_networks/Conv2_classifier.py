import torch.nn.functional as F
import torch.nn as nn

class Conv2_Classifier(nn.Module):
    def __init__(self):
        super(Conv2_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        #size: ( 28-5+1)x(28-5+1)x5
        self.conv2 = nn.Conv2d(5, 5, kernel_size=1)
        #size: 24x24x5
        self.pool = nn.MaxPool2d(2)
        #size: 12x12x5
        self.fc1 = nn.Linear(12*12*5, 10)

    def forward(self, x):

        out = x.view(-1, 1, 28, 28)
        out = F.relu(self.conv1(out))
        out  = F.relu(self.pool(self.conv2(out)))
        out = out.flatten(start_dim=1)
        out = self.fc1(out)
        out = F.softmax(out,  dim=-1)
        return out