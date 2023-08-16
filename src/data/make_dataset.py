# -*- coding: utf-8 -*-
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

project_dir = Path(__file__).resolve().parents[2]

import networkx as nx
from scipy.spatial import KDTree
from torch_geometric.utils import from_networkx

def build_graph_from_image(img, distance_threshold=2):
    # Extract non-zero pixels
    nonzero_pixels = np.column_stack(np.where(img > 0))
    
    # Using KDTree to efficiently query nearby points
    kdtree = KDTree(nonzero_pixels)
    pairs = kdtree.query_pairs(distance_threshold)
    
    # Create graph from edges
    G = nx.Graph()
    for i, j in pairs:
        xi, yi = nonzero_pixels[i]
        xj, yj = nonzero_pixels[j]
        distance = np.sqrt((xi-xj)**2 + (yi-yj)**2)
        if distance <= distance_threshold:
            G.add_edge(i, j)
    
    # Add node attributes
    for idx, (x, y) in enumerate(nonzero_pixels):
        G.add_node(idx, pos=(x,y), value=img[x, y])

    data = from_networkx(G)
    return data

def main(args):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    label = args.label
    ratio = args.ratio
    data_path = f"/ssl-jet-vol-v2/ssi_2023_if_1/data/if-image-{args.label}.h5"
    batch_size = 100
    train_data = ParticleImage2D(
            data_files=[data_path],
            start=0.0,  # start of the dataset fraction to use. 0.0 = use from 1st entry
            end=ratio,  # end of the dataset fraction to use. 1.0 = use up the last entry
        )
    train_graphs = []
    for i in range(len(train_data)):
        data = build_graph_from_image(train_data[i]["data"])
        data.y = train_data[i]["label"]  # label for the graph
        train_graphs.append(data)
        if i % 1000 == 0:
            torch.save(train_graphs, f"/ssl-jet-vol-v2/ssi_2023_if_1/data/{label}_graphs_2_{i}.pt")
            print(i)
    torch.save(train_graphs, f"/ssl-jet-vol-v2/ssi_2023_if_1/data/{label}_graphs_2.pt")
    


if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="CNN")
    parser.add_argument(
        "--label",
        type=str,
        action="store",
        dest="label",
        default="train",
        help="train / test",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        action="store",
        dest="ratio",
        default="1.0",
        help="ratio of dataset to load",
    )
    args = parser.parse_args()
    main(args)
