import torch.nn.functional as F
import torch.nn as nn

class Conv2_Classifier(nn.Module):
    def __init__(self):
        super(Conv2_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 5, kernel_size=1)

        #self.conv3 = nn.Conv2d(16,3, kernel_size=1)
        self.fc1 = nn.Linear(720, 10)
        #self.fc2 = nn.Linear(20, 10)

    def forward(self, x):


        out = x.view(-1, 1, 28, 28)
        out = F.relu(self.conv1(out))
        out  = F.relu(F.max_pool2d(self.conv2(out), 2))
        #out = F.relu(F.max_pool2d(self.conv3(out),2))

        out = out.flatten(start_dim=1)
        out = self.fc1(out)
        #out = self.fc2(out)
        out = F.softmax(out,  dim=-1)
        return out