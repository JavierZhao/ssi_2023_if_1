import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import models


import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import yaml

from tqdm import tqdm
from itertools import islice

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

parser = argparse.ArgumentParser()
parser.add_argument("jobID", type=str)
parser.add_argument('-c', '--config-file', required=True, type=str, help="xxx.yaml")
args = parser.parse_args()

# job_id
job_id = args.jobID

# Load config
with open(args.config_file) as file:
    config = yaml.safe_load(file)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Configuration
datapath= = config["dataset"]["root_dir"]
output_dir = config["output"]["output_dir"] + job_id

backbone = config["model"]["backbone"]
batch_size = config["train"]["batch_size"]
epochs = config["train"]["num_epochs"]
learning_rate = config["train"]["lr"]



# torch.multiprocessing.set_start_method('spawn')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SEED=12345
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)


# import h5py as h5

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from iftool.image_challenge import ParticleImage2D

train_data = ParticleImage2D(data_files = [datapath],
                             start = 0.0, # start of the dataset fraction to use. 0.0 = use from 1st entry
                             end   = config["train"]["data_fraction"], # end of the dataset fraction to use. 1.0 = use up the last entry
                            )


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
train_size = int(0.8 * len(train_data))
test_size = len(train_data) - train_size
train_data, test_data = random_split(train_data, [train_size, test_size])

print(f'Number of train examples = {train_size}')
print(f'Number of test examples = {test_size}')

# We use a specifically designed "collate" function to create a batch data
from iftool.image_challenge import collate

train_loader = DataLoader(train_data,
                          collate_fn  = collate,
                          shuffle     = True,
                          num_workers = 4,
                          batch_size  = batch_size
                         )

test_loader = DataLoader(test_data,
                          collate_fn  = collate,
                          shuffle     = False,
                          num_workers = 4,
                          batch_size  = batch_size
                         )
                         
                         
                         
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++        

from vgg16 import VGG16        
        
class My_Model(nn.Module):
    def __init__(self, model_type, num_classes=4):
        super(My_Model, self).__init__()
        
        if model_type == 'vgg16':
            self.model = VGG16(num_classes)
            
        elif model_type == 'resnet152':
            # Load the pretrained ResNet-152 model
            self.model = models.resnet152(pretrained=True)
            
            # Modify the last fully connected layer to match the number of classes
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            
        else:
            raise ValueError("Invalid model_type. Expected 'vgg16' or 'resnet152'")

    def forward(self, x):
        x = self.model(x)
        return x
        
        
        
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Let's use", torch.cuda.device_count(), "GPUs!")

model = My_Model(backbone)
model = model.to(device)
model = nn.DataParallel(model)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion  = nn.CrossEntropyLoss()
    
#------------------------------------------
    
global_progress = tqdm(range(0, epochs), desc='Training')

epoch_list = []
train_loss_list = []
test_loss_list = []

train_label = [] # Keep track of the label list of all epochs (each epoch is different because we shuffle the training set)
train_pred = []  # Keep track of the prediction list of all epochs

test_label = []  # Keep track of the label list (each epoch is the same because we didn't shuffle the test set)
test_pred = []   # Keep track of the prediction list of all epochs


for epoch in global_progress:

    model.train()

    train_local_progress = tqdm(
        train_loader,
        desc = f'Epoch {epoch}/{epochs}',
    )

    correct_predictions, total_predictions, train_loss = 0, 0, 0
    train_label_epoch = torch.empty((0, 4)).to(device)
    train_pred_epoch = torch.empty((0, 4)).to(device)

    for batch in train_local_progress:
        """
        image.shape = torch.Size([batch_size, channels, h, w])
        labels.shape = torch.Size([batch_size])   (long tensor)
        """
        images = batch['data'].to(device)
        labels = batch['label'].to(device)
                

        outputs = model(
            images.float().to(device, non_blocking=True))
        """ outputs.shape = torch.Size([batch_size, num_classes]); one-hot encoded """
        
        # One-hot encoding the long-tensor labels
        labels = nn.functional.one_hot(labels, num_classes=4).to(device, non_blocking=True)
        # 'CrossEntropyLoss' only accepts input tensors in float type
        labels = labels.float()
        """ labels.shape = torch.Size([batch_size, num_classes]) """

        # Compute loss for each batch
        """
        'CrossEntropyLoss' will apply a Softmax to the output (one-hot encoded),
        so the outputs here doesn't need an additional Softmax function
        """ 
        batch_loss = criterion(outputs, labels)
        """ batch_loss.shape = torch.Size([]) """
        train_loss += batch_loss

        # Backward pass and optimize
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        m = nn.Softmax(dim=1)
        outputs_prob = m(outputs)
        """ outputs_prob.shape = torch.Size([batch_size, num_classes]); one-hot encoded """
        
        # Count the correct presictions and total predictions from each batch
        labels_long = labels.argmax(dim=1)
        outputs_prob_long = outputs_prob.argmax(dim=1)
        correct_predictions += (outputs_prob_long == labels_long).sum().item()
        total_predictions += labels_long.size(0)
        
        # Concatenate the labels and predictioins from each batch
        train_label_epoch = torch.cat((train_label_epoch, labels), dim=0)
        train_pred_epoch = torch.cat((train_pred_epoch, outputs_prob), dim=0)

    # Accuracy of each epoch (assuming threshold = 0.5) 
    accuracy = correct_predictions/total_predictions
    # Number of batch in training set
    train_num_batch = train_size/batch_size
    # Average the total loss of all batches
    train_loss /= train_num_batch
    print(f'Epoch {epoch}: Training loss: {train_loss:.6f} / Accuracy: {accuracy:.3f}')

    train_label.append(train_label_epoch.detach().cpu().numpy())
    train_pred.append(train_pred_epoch.detach().cpu().numpy()) 
    np.save(f'{output_dir}/train_label_epoch.npy', train_label)
    np.save(f'{output_dir}/train_pred_epoch.npy', train_pred)

# -------------------------------------------------------------------------------------------------

    # Test the model on test set
    test_local_progress = tqdm(
        test_loader,
        desc = f'Epoch {epoch}/{epochs}',
    )

    
    correct_predictions, total_predictions, test_loss = 0, 0, 0
    
    with torch.no_grad():
        model.eval()
        
        if epoch == 0:
            test_label_0 = torch.empty((0, 4)).to(device)
        test_pred_epoch = torch.empty((0, 4)).to(device)

        for batch in test_local_progress:  
            """
            image.shape = torch.Size([batch_size, channels, h, w])
            labels.shape = torch.Size([batch_size])   (long tensor)
            """
            images = batch['data'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                images.float().to(device, non_blocking=True))
            """ outputs.shape = torch.Size([batch_size, num_classes]); one-hot encoded """
            
            # One-hot encoding the long-tensor labels
            labels = nn.functional.one_hot(labels, num_classes=4).to(device, non_blocking=True)
            # 'CrossEntropyLoss' only accepts input tensors in float type
            labels = labels.float()
            """ labels.shape = torch.Size([batch_size, num_classes]); one-hot encoded """
        
            # Compute loss for each batch
            """
            'CrossEntropyLoss' will apply a Softmax to the output (one-hot encoded),
            so the outputs here doesn't need an additional sigmoid function
            """ 
            batch_loss = criterion(outputs, labels).mean()
            """ batch_loss.shape = torch.Size([]) """
            test_loss += batch_loss        

            m = nn.Softmax(dim=1)
            outputs_prob = m(outputs)
            """ outputs_prob.shape = torch.Size([batch_size, num_classes]); one-hot encoded """
            
            # Count the correct presictions and total predictions from each batch
            labels_long = labels.argmax(dim=1)
            outputs_prob_long = outputs_prob.argmax(dim=1)
            correct_predictions += (outputs_prob_long == labels_long).sum().item()
            total_predictions += labels_long.size(0)
            
            # Concatenate the labels and predictioins from each batch
            if epoch == 0:
                test_label_0 = torch.cat((test_label_0, labels), dim=0)
            test_pred_epoch = torch.cat((test_pred_epoch, outputs_prob), dim=0)

        # Accuracy of each epoch (assuming threshold = 0.5) 
        accuracy = correct_predictions/total_predictions
        # Number of batch in test set
        test_num_batch = test_size/batch_size
        # Average the total loss of all batches
        test_loss /= test_num_batch
        
        print(f'Epoch {epoch}: Test loss: {test_loss:.6f} / Accuracy: {accuracy:.3f}')

        if epoch == 0:
            test_label.append(test_label_0.detach().cpu().numpy())
            np.save(f'{output_dir}/test_label_epoch.npy', test_label)

        test_pred.append(test_pred_epoch.detach().cpu().numpy()) 
        np.save(f'{output_dir}/test_pred_epoch.npy', test_pred)
                

    # Record all losses, make plot, and save the outputs
    epoch_list.append(epoch)
    train_loss_list.append(train_loss.detach().cpu().numpy())
    test_loss_list.append(test_loss.detach().cpu().numpy())

    plt.plot(epoch_list, train_loss_list, marker = 'o', color= 'navy')
    plt.plot(epoch_list, test_loss_list, marker = 'o', color= 'coral')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig(f'{output_dir}/plotter.pdf')

    
    output_dict = {
        'Epoch': epoch_list,
        'Train_loss': train_loss_list,
        'Test_loss': test_loss_list,
    }
    df = pd.DataFrame(output_dict)
    df.to_csv(f'{output_dir}/output_history.csv', index=False)

    
    if epoch == 0:
        lowest_test_loss = test_loss
        
    # Save the model that achieves the lowest loss on the test set
    if test_loss < lowest_test_loss:
        lowest_test_loss = test_loss
        model_save_path = f'{output_dir}/epoch_{epoch}_loss_{lowest_test_loss:.6f}.pt'
        torch.save(model, model_save_path)
