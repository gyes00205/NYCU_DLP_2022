'''Usage
python train.py --model resnet50 --epochs 5 --output_dir checkpoint_implement
'''
import os
import torch
import torchvision
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models
from torch import optim
from torchsummary import summary
import copy
import argparse
from dataloader import RetinopathyLoader
from tqdm import tqdm
from model import ResNet18, ResNet50

def train(model, SAVE_MODEL_PATH):

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    loss_epoch = []
    train_acc, test_acc = [], []
    best_acc = 0.0
    for epoch in range(epochs):
        iter = 0
        correct_train, total_train = 0, 0
        correct_test, total_test = 0, 0
        train_loss = 0.0

        model.train()
        print('epoch: ' + str(epoch + 1) + ' / ' + str(epochs))

        # ---------------------------
        # Training Stage
        # ---------------------------

        for i, (x, label) in enumerate(tqdm(train_dataloader)):
            x = x.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.int64)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # calculate training data accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += label.size(0)
            correct_train += (predicted == label).sum()

            train_loss += loss.item()
            iter += 1

        print(
            'Training epoch: %d | loss: %.3f | acc: %.3f' %
            (epoch + 1, train_loss / iter, correct_train / total_train)
        )

        # --------------------------
        # Testing Stage
        # --------------------------

        model.eval()

        for i, (x, label) in enumerate(tqdm(test_dataloader)):
            with torch.no_grad():  # don't need gradient
                x, label = x.to(device), label.to(device)
                outputs = model(x)  # predict image
                # calculate testing data accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_test += label.size(0)
                correct_test += (predicted == label).sum()

        valid_acc = correct_test / total_test
        print(f'Test acc: {valid_acc:.3f}')
        if valid_acc > best_acc:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = valid_acc
            torch.save(best_model_wts, SAVE_MODEL_PATH)

        train_acc.append(100 * (correct_train / total_train))
        test_acc.append(100 * (correct_test / total_test))
        loss_epoch.append(train_loss / iter)
    
    return train_acc, test_acc, loss_epoch, best_acc


def split_train_test():

    root = 'data'
    train_dataloader = DataLoader(
        RetinopathyLoader(root, 'train'),
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        RetinopathyLoader(root, 'test'),
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, test_dataloader


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50", type=str)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--output_dir", required=True, type=str)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_config()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    lr = 1e-3
    epochs = args.epochs
    train_dataloader, test_dataloader = split_train_test()    

    if args.model == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        model = models.resnet18(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 5)
    if args.model == 'resnet50':
        model.load_state_dict(torch.load('checkpoint_20epochs_16bs/resnet50_pretrained.pt'))
    model = model.to(device)
    train_acc1, test_acc1, train_loss1, best_acc1 = train(model, f'{args.output_dir}/resnet{args.model[-2:]}_pretrained.pt')

    if args.model == 'resnet50':
        model = ResNet50()
    else:
        model = ResNet18()

    model = model.to(device)
    train_acc2, test_acc2, train_loss2, best_acc2 = train(model, f'{args.output_dir}/resnet{args.model[-2:]}_wo.pt')
    
    plt.figure()
    x = np.linspace(1, epochs, epochs)
    plt.plot(x, train_loss1)  # plot your loss
    plt.plot(x, train_loss2)
    plt.title(f'Result comparison (ResNet{args.model[-2:]})')
    plt.ylabel('Loss'), plt.xlabel('Epoch')
    plt.legend(['with pretrained', 'w/o pretrained'], loc='upper left')
    plt.savefig(os.path.join(args.output_dir, f'ResNet{args.model[-2:]}_loss.png'))

    plt.figure()
    plt.plot(x, train_acc1)  # plot your training accuracy
    plt.plot(x, test_acc1)  # plot your testing accuracy
    plt.plot(x, train_acc2)  
    plt.plot(x, test_acc2)  
    plt.title(f'Result comparison (ResNet{args.model[-2:]})')
    plt.ylabel('Accuracy (%)'), plt.xlabel('Epoch')
    plt.legend(['Train(with pretrained)', 'Test(with pretrained)', 'Train(w/o pretrained)', 'Test(w/o pretrained)'])
    plt.savefig(os.path.join(args.output_dir, f'ResNet{args.model[-2:]}_acc.png'))

    print(f'ResNet{args.model[-2:]} with pretrained test accuracy: {best_acc1}')
    print(f'ResNet{args.model[-2:]} with pretrained test accuracy: {best_acc2}')