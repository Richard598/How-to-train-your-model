import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.ToTensor()
])


trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)


testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

print('CIFAR-10 already downloaded！') 
