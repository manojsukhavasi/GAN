from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class GAN_D(nn.Module):
    def __init__(self):
        super(GAN_D, self).__init__()
        self.fc1 = nn.Linear(784,300)
        self.fc2 = nn.Linear(300,300)
        self.fc3 = nn.Linear(300,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

class GAN_G(nn.Module):
    def __init__(self):
        super(GAN_G, self).__init__()
        self.fc1 = nn.Linear(100,300)
        self.fc2 = nn.Linear(300,300)
        self.fc3 = nn.Linear(300,784)


    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x
