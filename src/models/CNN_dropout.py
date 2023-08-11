import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

SEED = 12345
_ = np.random.seed(SEED)
_ = torch.manual_seed(SEED)

from iftool.image_challenge import ParticleImage2D

# We use a specifically designed "collate" function to create a batch data
from iftool.image_challenge import collate
from torch.utils.data import DataLoader


class CNN_dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5)
        self.dropout2d = nn.Dropout2d(p=0.1)
        self.conv2 = nn.Conv2d(12, 6, kernel_size=3)
        self.max_pool = nn.MaxPool2d(5, 5)
        self.fc1 = nn.Linear(8214, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, 4)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #         x = self.dropout2d(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2d(x)
        x = self.max_pool(x)
        x = x.view(-1, 8214)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
