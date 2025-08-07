import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np

class ResNet18CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.model.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    
    net = ResNet18CIFAR10()
    net.load_state_dict(torch.load('resnet18_cifar10.pth', map_location='cpu'))
    net.eval()

    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = net(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    
    y_true = label_binarize(all_labels, classes=list(range(10)))
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(10):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], all_probs[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], all_probs[:, i])
    
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.reshape(-1), all_probs.reshape(-1))
    average_precision["micro"] = average_precision_score(y_true, all_probs, average="micro")

    
    plt.figure(figsize=(8, 6))
    plt.plot(recall["micro"], precision["micro"], label=f'micro-average (AP={average_precision["micro"]:.2f})', color='navy', linewidth=2)
    for i in range(10):
        plt.plot(recall[i], precision[i], lw=1, label=f'Class {i} (AP={average_precision[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('CIFAR-10 Precision-Recall curve')
    plt.legend(loc='lower left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show() 