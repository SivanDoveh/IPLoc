import math
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import scipy.io
import pickle
import json


class LocDataset(Dataset):
    def __init__(self,args):
        """
        Args:
            data_list (list): List of your data.
        """
        with open(args.data_path, 'rb') as f:
            data_all = json.load(f)
        # Initialize lists to hold results
        if args.chunks > 1:
            data_all = data_all[args.curr_chunk:-1:args.chunks]
        self.data_all=data_all

    def __len__(self):
        return self.data_all.__len__()

    def __getitem__(self, idx):# every image and desc used once. same order everytime for now
        data = self.data_all[idx]
        element = data["element"]
        bbox = data["bbox"] 
        image_path = data["image_path"] 
        image_id = data["image_id"]      
        return element,bbox,image_path,image_id,data

def get_dataloader(args):
    loc_dataset = LocDataset(args)
    LocDataloader = DataLoader(loc_dataset, batch_size=args.bs, shuffle=False)
    return LocDataloader
