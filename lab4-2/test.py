'''Usage
python test.py --model resnet50 --checkpoint checkpoint_20epochs_16bs/resnet50_pretrained.pt
'''
from sklearn.metrics import confusion_matrix
import torchvision.models as models
import argparse
from dataloader import RetinopathyLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import ResNet18, ResNet50

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--pretrained", action="store_true")
    return parser.parse_args()


def plot_confusion_matrix(y_true, y_pred, classes=[0, 1, 2, 3, 4],
                          title=None,
                          cmap=plt.cm.Blues,
                          filename=None):
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize='true')
    fig, ax = plt.subplots()
    sn.heatmap(cm, annot=True, ax=ax, cmap=cmap, fmt='.2f')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.xaxis.set_ticklabels(classes, rotation=45)
    ax.yaxis.set_ticklabels(classes, rotation=0)
    plt.title(title)
    plt.savefig(filename, dpi=300)


def evaluate(model):
    correct_test, total_test = 0, 0
    labels, pred = [], []
    model.eval()
    for i, (x, label) in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():  # don't need gradient
            x, label = x.to(device), label.to(device)
            outputs = model(x)  # predict image
            # calculate testing data accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_test += label.size(0)
            correct_test += (predicted == label).sum()
            
            labels.extend(label.detach().cpu().numpy().tolist())
            pred.extend(predicted.detach().cpu().numpy().tolist())

    test_acc = correct_test / total_test
    print(f'Test acc: {test_acc:.3f}')
    filename = args.checkpoint.split('/')[-1]
    plot_confusion_matrix(labels, pred, title=f'Normalized confusion matrix ({args.model})',
                          filename=f'cm/{filename[:-2]}.png')


if __name__ == '__main__':
    args = parse_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    if args.pretrained:
        if args.model == 'resnet50':
            model = models.resnet50(pretrained=False)
        else:
            model = models.resnet18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 5)
    else:
        if args.model == 'resnet50':
            model = ResNet50()
        else:
            model = ResNet18()

    
    model.load_state_dict(torch.load(args.checkpoint))
    model = model.to(device)

    test_dataloader = DataLoader(
        RetinopathyLoader('data', 'test'),
        batch_size=batch_size,
        shuffle=False
    )

    evaluate(model)
