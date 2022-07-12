from fileinput import filename
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from tqdm import tqdm

def custom_transform(mode):
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
    # Transformer
    if mode == 'train':
        transformer = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transformer = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    return transformer


def getData(mode, root, test_file='test.json'):
    label_map = json.load(open(os.path.join(root, 'objects.json')))
    filenames = None
    if mode == 'train':
        data_json = json.load(open(os.path.join(root, 'train.json')))
        filenames = list(data_json.keys())
        labels_list = list(data_json.values())
        one_hot_vector_list = []
        for i in range(len(labels_list)):
            one_hot_vector = np.zeros(24, dtype=np.int)
            for j in labels_list[i]:
                one_hot_vector[label_map[j]] = 1
            one_hot_vector_list.append(one_hot_vector)
        labels = np.array(one_hot_vector_list)
        
    else:
        data_json = json.load(open(os.path.join(root, test_file)))
        labels_list = data_json
        one_hot_vector_list = []
        for i in range(len(labels_list)):
            one_hot_vector = np.zeros(24, dtype=np.int)
            for j in labels_list[i]:
                one_hot_vector[label_map[j]] = 1
            one_hot_vector_list.append(one_hot_vector)
        labels = np.array(one_hot_vector_list)

    return filenames, labels


class iclevrDataset(Dataset):

    def __init__(self, mode, root, test_file='test.json'):
        self.filenames = None
        self.labels = None
        self.mode = mode
        self.root = root
        self.filenames, self.labels = getData(mode, root, test_file=test_file)
        self.transform = custom_transform(mode)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.mode == 'train':
            image = Image.open(os.path.join(self.root, self.filenames[idx])).convert('RGB')
            image = self.transform(image)
            label = torch.Tensor(self.labels[idx])
            return image, label
        else:
            label = torch.Tensor(self.labels[idx])
            return label


if __name__ == '__main__':
    dataset = iclevrDataset(mode='train', root='iclevr')
    print(len(dataset))
    train_dataloader = DataLoader(
        iclevrDataset(mode='test', root='iclevr'),
        batch_size=4,
        shuffle=True
    )
    for i, label in enumerate(tqdm(train_dataloader)):
        pass