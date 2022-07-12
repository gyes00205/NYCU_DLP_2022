import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import dataloader as DA
from torchsummary import summary
from model import EEGNet, DeepConvNet
import dataloader as DA
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim
import copy
import argparse


class BCIDataset(Dataset):

    def __init__(self, datalist, labels):
        self.datalist = datalist
        self.labels = labels
    
    def __len__(self):
        return  len(self.datalist)

    def __getitem__(self, idx):
        data = self.datalist[idx]
        label = self.labels[idx]
        return  data, label


def parse_config():
    """Define parse config
    model: which model you would select
    activation: which activation you would select
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EEGNet", type=str)
    parser.add_argument("--activation", default='LeakyReLU', type=str)
    parser.add_argument("--weight", default='checkpoint/EEGNet_LeakyReLU.pth')
    return parser.parse_args()


def test(model):
    
    total_test = 0
    correct_test = 0
    model.eval()
    for i, (x, label) in enumerate(test_dataloader):
        with torch.no_grad():  # don't need gradient
            x = x.to(device, dtype=torch.float32)
            label = label.to(device)
            outputs = model(x)  # predict image
            # calculate testing data accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_test += label.size(0)
            correct_test += (predicted == label).sum()

    test_acc = correct_test / total_test
    print(f'Test acc: {test_acc:.3f}')


if __name__ == '__main__':

    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_data, test_label = DA.read_bci_data()
    test_dataloader = DataLoader(
        BCIDataset(test_data, test_label),
        batch_size=batch_size,
        shuffle=False
    )
    summary(EEGNet().to(device), (1, 2, 750))
    summary(DeepConvNet().to(device), (1, 2, 750))
    for model_name in ['EEGNet', 'DeepConvNet']:
        for activation in ['ReLU', 'ELU', 'LeakyReLU']:
            if model_name == 'EEGNet':
                if activation == 'ReLU':
                    model = EEGNet(activation=nn.ReLU())
                elif activation == 'ELU':
                    model = EEGNet(activation=nn.ELU())
                else:
                    model = EEGNet(activation=nn.LeakyReLU())
            else:
                if activation == 'ReLU':
                    model = DeepConvNet(activation=nn.ReLU())
                elif activation == 'ELU':
                    model = DeepConvNet(activation=nn.ELU())
                else:
                    model = DeepConvNet(activation=nn.LeakyReLU())

            print(f'{model_name} {activation}:')
            model.load_state_dict(torch.load(f'checkpoint/{model_name}_{activation}.pth'))
            model = model.to(device)
            test(model)
            print('----------------------------')