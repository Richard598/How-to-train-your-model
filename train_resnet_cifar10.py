import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import os


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.5),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class ResNet18CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')  
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.model.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    net = ResNet18CIFAR10()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    
    model_path = 'resnet18_cifar10.pth'
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
        print(f'model parameters loaded：{model_path}，will keep training...')
    else:
        print('can\'t find model parameters，will train from zero...')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    for epoch in range(1, 11):  
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f'Epoch {epoch}, Step {i+1}, Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        scheduler.step()

        
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Epoch {epoch}, Test Accuracy: {100 * correct / total:.2f}%')

        
        if epoch % 10 == 0:
            torch.save(net.state_dict(), f'resnet18_cifar10_epoch{epoch}.pth')
            print(f'model is saved as resnet18_cifar10_epoch{epoch}.pth')

    print('training is over！')
    torch.save(net.state_dict(), 'resnet18_cifar10.pth')
    print('model is saved as resnet18_cifar10.pth') 
