import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.grid'] = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
torch.multiprocessing.set_start_method('spawn')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import numpy as np
SEED=12345
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)

import h5py as h5
datapath='if-image-train.h5'

# Open a file in 'r'ead mode.
f=h5.File(datapath,mode='r',swmr=True) 

# List items in the file
for key in f.keys():
    print('dataset',key,'... type',f[key].dtype,'... shape',f[key].shape)
    
from iftool.image_challenge import ParticleImage2D
train_data = ParticleImage2D(data_files=[datapath])

train_data = ParticleImage2D(data_files = [datapath],
                             start = 0.0, # start of the dataset fraction to use. 0.0 = use from 1st entry
                             end   = 0.04, # end of the dataset fraction to use. 1.0 = use up the last entry
                            )
val_data = ParticleImage2D(data_files = [datapath],
                             start = 0.1, # start of the dataset fraction to use. 0.0 = use from 1st entry
                             end   = 0.104, # end of the dataset fraction to use. 1.0 = use up the last entry
                            )

# We use a specifically designed "collate" function to create a batch data
from iftool.image_challenge import collate
from torch.utils.data import DataLoader
batch_size = 128
train_loader = DataLoader(train_data,
                          collate_fn  = collate,
                          shuffle     = True,
#                           num_workers = 4,
                          batch_size  = batch_size
                         )
val_loader = DataLoader(val_data,
                          collate_fn  = collate,
                          shuffle     = True,
#                           num_workers = 4,
                          batch_size  = batch_size
                         )
learning_rate = 0.001


class CNN_dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5)
        self.dropout2d = nn.Dropout2d(p=0.1)
        self.conv2 = nn.Conv2d(12, 6, kernel_size=3)
        self.max_pool = nn.MaxPool2d(5, 5)
        self.fc1 = nn.Linear(8214, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024,128)
        self.fc4 = nn.Linear(128,4)
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
    
model = CNN_dropout().to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
n_total_steps = len(train_loader)

train_its = int(len(train_data) / batch_size)
val_its = int(len(val_data) / batch_size)

train_losses = []
val_losses = []
val_accs = []
val_acc_best = 0
epochs = 15

for epoch in range(epochs):
    l_train_epoch = []
    l_val_epoch = []
    model.train()
    pbar = tqdm.tqdm(train_loader, total=train_its)
    for i, batch in enumerate(pbar):
        images = batch["data"]
        labels = batch["label"]
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        l_train_epoch.append(loss.detach().cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#         if (i+1) % 10 == 0:
#             print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Training Loss: {loss.item():.4f} ')
    l_train = np.mean(l_train_epoch)
    train_losses.append(l_train)
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {l_train:.4f} ')
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        pbar = tqdm.tqdm(val_loader, total=val_its)
        for i, batch in enumerate(pbar):
            images = batch["data"]
            labels = batch["label"]
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            l_val_epoch.append(loss.detach().cpu())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

#             if (i+1) % 2 == 0:
#                 print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Val Loss: {loss.item():.4f} ')
    val_acc = correct/total
    print(f"val accuracy: {val_acc}")
    val_accs.append(val_acc)
    
    l_val = np.mean(l_val_epoch)
    print(f'Epoch [{epoch+1}/{epochs}], Val Loss: {l_val:.4f} ')
    if val_acc > val_acc_best:
        val_acc_best = val_acc
        print("new best model")
        torch.save(model.state_dict(), "CNN_dropout_best.pth")
    
    val_losses.append(l_val)

np.save("train_losses.npy", np.array(train_losses))
np.save("val_losses.npy", np.array(val_losses))
np.save("val_accs.npy", np.array(val_accs))