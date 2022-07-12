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

def train(model_name, activation, SAVE_MODEL_PATH):
    if model_name == 'EEGNet':
        model = EEGNet(activation=activation).to(device)
    else:
        model = DeepConvNet(activation=activation).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_data, train_label, test_data, test_label = DA.read_bci_data()
    train_dataloader = DataLoader(
        BCIDataset(train_data, train_label),
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        BCIDataset(test_data, test_label),
        batch_size=batch_size,
        shuffle=False
    )

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

        for i, (x, label) in enumerate(train_dataloader):
            x = x.to(device, dtype=torch.float32)
            label = label.type(torch.LongTensor)
            label = label.to(device)

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
            'Training epoch: %d / loss: %.3f | acc: %.3f' %
            (epoch + 1, train_loss / iter, correct_train / total_train)
        )

        # --------------------------
        # Testing Stage
        # --------------------------

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

        valid_acc = correct_test / total_test
        print(f'Test acc: {valid_acc:.3f}')
        if valid_acc > best_acc:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = valid_acc
            torch.save(best_model_wts, SAVE_MODEL_PATH)

        # step_lr_scheduler.step()

        train_acc.append(100 * (correct_train / total_train))
        test_acc.append(100 * (correct_test / total_test))
        loss_epoch.append(train_loss / iter)
    
    return train_acc, test_acc, loss_epoch, best_acc


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    lr = 1e-3
    epochs = 300

    SAVE_MODEL_PATH = 'checkpoint/EEGNet_ReLU.pth'
    EEG_ReLU_train_acc, EEG_ReLU_test_acc, EEG_ReLU_train_loss, EEG_ReLU_best_acc = train('EEGNet', nn.ReLU(), SAVE_MODEL_PATH)

    SAVE_MODEL_PATH = 'checkpoint/EEGNet_ELU.pth'
    EEG_ELU_train_acc, EEG_ELU_test_acc, EEG_ELU_train_loss, EEG_ELU_best_acc = train('EEGNet', nn.ELU(), SAVE_MODEL_PATH)

    SAVE_MODEL_PATH = 'checkpoint/EEGNet_LeakyReLU.pth'
    EEG_LeakyReLU_train_acc, EEG_LeakyReLU_test_acc, EEG_LeakyReLU_train_loss, EEG_LeakyReLU_best_acc = train('EEGNet', nn.LeakyReLU(), SAVE_MODEL_PATH)

    plt.figure()
    x = np.linspace(1, epochs, epochs)
    plt.plot(x, EEG_ReLU_train_loss)  # plot your loss
    plt.plot(x, EEG_ELU_train_loss)
    plt.plot(x, EEG_LeakyReLU_train_loss)
    plt.title('Activation function comparison (EEGNet)')
    plt.ylabel('Loss'), plt.xlabel('Epoch')
    plt.legend(['ReLU', 'ELU', 'LeakyReLU'], loc='upper left')
    plt.savefig('EEGNet_loss.png')

    plt.figure()
    plt.plot(x, EEG_ReLU_train_acc)  # plot your training accuracy
    plt.plot(x, EEG_ReLU_test_acc)  # plot your testing accuracy
    plt.plot(x, EEG_ELU_train_acc)  # plot your training accuracy
    plt.plot(x, EEG_ELU_test_acc)  # plot your testing accuracy
    plt.plot(x, EEG_LeakyReLU_train_acc)  # plot your training accuracy
    plt.plot(x, EEG_LeakyReLU_test_acc)  # plot your testing accuracy
    plt.title('Activation function comparison (EEGNet)')
    plt.ylabel('Accuracy (%)'), plt.xlabel('Epoch')
    plt.legend(['ReLU_train', 'ReLU_test', 'ELU_train', 'ELU_test', 'LeakyReLU_train', 'LeakyReLU_test'])
    plt.savefig('EEGNet_acc.png')

    # DeepConvNet
    SAVE_MODEL_PATH = 'checkpoint/DeepConvNet_ReLU.pth'
    Deep_ReLU_train_acc, Deep_ReLU_test_acc, Deep_ReLU_train_loss, Deep_ReLU_best_acc = train('DeepConvNet', nn.ReLU(), SAVE_MODEL_PATH)

    SAVE_MODEL_PATH = 'checkpoint/DeepConvNet_ELU.pth'
    Deep_ELU_train_acc, Deep_ELU_test_acc, Deep_ELU_train_loss, Deep_ELU_best_acc = train('DeepConvNet', nn.ELU(), SAVE_MODEL_PATH)

    SAVE_MODEL_PATH = 'checkpoint/DeepConvNet_LeakyReLU.pth'
    Deep_LeakyReLU_train_acc, Deep_LeakyReLU_test_acc, Deep_LeakyReLU_train_loss, Deep_LeakyReLU_best_acc = train('DeepConvNet', nn.LeakyReLU(), SAVE_MODEL_PATH)

    plt.figure()
    x = np.linspace(1, epochs, epochs)
    plt.plot(x, Deep_ReLU_train_loss)  # plot your loss
    plt.plot(x, Deep_ELU_train_loss)
    plt.plot(x, Deep_LeakyReLU_train_loss)
    plt.title('Activation function comparison (DeepConvNet)')
    plt.ylabel('Loss'), plt.xlabel('Epoch')
    plt.legend(['ReLU', 'ELU', 'LeakyReLU'], loc='upper left')
    plt.savefig('DeepConvNet_loss.png')
    # plt.show()

    plt.figure()
    plt.plot(x, Deep_ReLU_train_acc)  # plot your training accuracy
    plt.plot(x, Deep_ReLU_test_acc)  # plot your testing accuracy
    plt.plot(x, Deep_ELU_train_acc)  # plot your training accuracy
    plt.plot(x, Deep_ELU_test_acc)  # plot your testing accuracy
    plt.plot(x, Deep_LeakyReLU_train_acc)  # plot your training accuracy
    plt.plot(x, Deep_LeakyReLU_test_acc)  # plot your testing accuracy
    plt.title('Activation function comparison (DeepConvNet)')
    plt.ylabel('Accuracy (%)'), plt.xlabel('Epoch')
    plt.legend(['ReLU_train', 'ReLU_test', 'ELU_train', 'ELU_test', 'LeakyReLU_train', 'LeakyReLU_test'])
    plt.savefig('DeepConvNet_acc.png')

    print(f'EEGNet_ReLU best acc: {EEG_ReLU_best_acc}')
    print(f'EEGNet_ELU best acc: {EEG_ELU_best_acc}')
    print(f'EEGNet_LeakyReLU best acc: {EEG_LeakyReLU_best_acc}')
    print(f'DeepConvNet_ReLU best acc: {Deep_ReLU_best_acc}')
    print(f'DeepConvNet_ELU best acc: {Deep_ELU_best_acc}')
    print(f'DeepConvNet_LeakyReLU best acc: {Deep_LeakyReLU_best_acc}')
