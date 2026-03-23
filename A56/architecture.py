import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyImageNet_CNN(nn.Module):

    def __init__(self):
        super(TinyImageNet_CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.conv2 = nn.Conv2d(64, 128, 5, 1, 2)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 384, 5, 1, 1)
        self.conv5 = nn.Conv2d(384, 256, 1, 1, 1)

        self.pool = nn.MaxPool2d(2, 2)
        self.adapt_pool = nn.AdaptiveAvgPool2d((3, 3))

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 200)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)

        x = self.adapt_pool(x)

        x = torch.flatten(x, 1)

        x = self.dropout1(x)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        
        return x
