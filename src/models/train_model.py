import argparse
import copy
import glob
import os
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

# define the global base device
world_size = torch.cuda.device_count()
if world_size:
    device = torch.device("cuda:0")
    for i in range(world_size):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    device = "cpu"
    print("Device: CPU")

import numpy as np

SEED = 12345
_ = np.random.seed(SEED)
_ = torch.manual_seed(SEED)

import h5py as h5
from iftool.image_challenge import ParticleImage2D
from iftool.image_challenge import collate
from torch.utils.data import DataLoader
from src.CNN_dropout import CNN_dropout

project_dir = Path(__file__).resolve().parents[2]


def main(args):
    outdir = args.outdir
    label = args.label
    model_loc = f"{outdir}/trained_models/"
    model_perf_loc = f"{outdir}/model_performances/{label}"
    os.system(
        f"mkdir -p {model_loc} {model_perf_loc}"
    )  # -p: create parent dirs if needed, exist_ok

    datapath = args.datapath
    # Open a file in 'r'ead mode.
    f = h5.File(datapath, mode="r", swmr=True)

    # List items in the file
    for key in f.keys():
        print("dataset", key, "... type", f[key].dtype, "... shape", f[key].shape)

    train_data = ParticleImage2D(data_files=[datapath])

    train_data = ParticleImage2D(
        data_files=[datapath],
        start=0.0,  # start of the dataset fraction to use. 0.0 = use from 1st entry
        end=args.train_size,  # end of the dataset fraction to use. 1.0 = use up the last entry
    )
    val_data = ParticleImage2D(
        data_files=[datapath],
        start=1.0
        - args.val_size,  # start of the dataset fraction to use. 0.0 = use from 1st entry
        end=1.0,  # end of the dataset fraction to use. 1.0 = use up the last entry
    )

    # We use a specifically designed "collate" function to create a batch data
    f.close()
    batch_size = args.batch_size
    train_loader = DataLoader(
        train_data,
        collate_fn=collate,
        shuffle=True,
        #                           num_workers = 4,
        batch_size=batch_size,
    )
    val_loader = DataLoader(
        val_data,
        collate_fn=collate,
        shuffle=True,
        #                           num_workers = 4,
        batch_size=batch_size,
    )
    learning_rate = 0.001

    model = CNN_dropout().to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    n_total_steps = len(train_loader)

    train_its = int(len(train_data) / batch_size)
    val_its = int(len(val_data) / batch_size)

    train_losses = []
    val_losses = []
    val_accs = []
    val_acc_best = 0
    epochs = args.epoch

    for epoch in range(epochs):
        l_train_epoch = []
        l_val_epoch = []
        model.train()
        pbar = tqdm.tqdm(train_loader, total=train_its)
        for _, batch in enumerate(pbar):
            images = batch["data"]
            labels = batch["label"]
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.detach().cpu().item()
            l_train_epoch.append(loss)
            pbar.set_description(f"Training loss: {loss:.4f}")

        l_train = np.mean(l_train_epoch)
        train_losses.append(l_train)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {l_train:.4f} ")

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
                loss = loss.detach().cpu().item()
                l_val_epoch.append(loss)
                pbar.set_description(f"Validation loss: {loss:.4f}")
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

        val_acc = correct / total
        print(f"val accuracy: {val_acc}")
        val_accs.append(val_acc)

        l_val = np.mean(l_val_epoch)
        print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {l_val:.4f} ")
        if val_acc > val_acc_best:
            val_acc_best = val_acc
            print("new best model")
            torch.save(model.state_dict(), f"{model_loc}/CNN_dropout_{label}_best.pth")
        torch.save(model.state_dict(), f"{model_loc}/CNN_dropout_{label}_last.pth")
        val_losses.append(l_val)

    # after training, save the losses and accuracies
    np.save(f"{model_perf_loc}/train_losses.npy", np.array(train_losses))
    np.save(f"{model_perf_loc}/val_losses.npy", np.array(val_losses))
    np.save(f"{model_perf_loc}/val_accs.npy", np.array(val_accs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN")
    parser.add_argument(
        "--datapath",
        type=str,
        default=f"{project_dir}/data/if-image-train.h5",
        metavar="N",
        help="path to data",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.9,
        help="Percentage of data to use for training",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Percentage of data to use for validation",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=50,
        help="Epochs to train for",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        action="store",
        dest="outdir",
        default=f"{project_dir}/models/",
        help="Output directory",
    )
    parser.add_argument(
        "--label",
        type=str,
        action="store",
        dest="label",
        default="baseline",
        help="label",
    )
    args = parser.parse_args()
    main(args)
