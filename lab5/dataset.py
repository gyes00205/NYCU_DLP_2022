from random import seed
import torch
import os
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

default_transform = transforms.Compose([
    transforms.ToTensor(),
    ])


class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):
        # TODO
        assert mode == 'train' or mode == 'test' or mode == 'validate'
        self.root_dir = args.data_root
        self.data_dir = os.path.join(self.root_dir, mode)
        if mode == 'train':    
            self.ordered = False
        else:
            self.ordered = True
        
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir(os.path.join(self.data_dir, d1)):
                self.dirs.append(os.path.join(self.data_dir, d1, d2))

        self.seq_len = args.n_past + args.n_future
        self.seed_is_set = False
        self.d = 0
        self.transform = transform
        self.cur_dir = ''
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            
    def __len__(self):
        # TODO
        return len(self.dirs)
        
    def get_seq(self):
        # TODO
        if self.ordered:
            self.cur_dir = self.dirs[self.d]
            if self.d == len(self.dirs) - 1:
                self.d = 0
            else:
                self.d+=1
        else:
            self.cur_dir = self.dirs[np.random.randint(len(self.dirs))]

        image_seq = []
        for i in range(self.seq_len):
            fname = os.path.join(self.cur_dir, f'{i}.png')
            im = self.transform(Image.open(fname)).view(1, 3, 64, 64)
            image_seq.append(im)
        image_seq = torch.cat(image_seq, dim=0)
        return image_seq
    
    def get_csv(self):
        d = self.cur_dir
        csv_seq = []
        actions = list(csv.reader(open(os.path.join(d, 'actions.csv'), newline='')))
        endeffector = list(csv.reader(open(os.path.join(d, 'endeffector_positions.csv'), newline='')))
        for i in range(self.seq_len):
            row_list = actions[i]
            row_list.extend(endeffector[i])
            csv_seq.append(row_list)
        csv_seq = torch.tensor(np.array(csv_seq).astype(np.float), dtype=torch.float)
        return csv_seq
    
    def __getitem__(self, index):
        self.set_seed(index)
        seq = self.get_seq()
        cond =  self.get_csv()
        return seq, cond
