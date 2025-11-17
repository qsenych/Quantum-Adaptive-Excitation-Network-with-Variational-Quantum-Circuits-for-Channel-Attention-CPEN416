"""
This code was Retrieved from https://www.geeksforgeeks.org/deep-learning/building-a-convolutional-neural-network-using-pytorch/
on November 5, 2025. The origional Source code without any changes can be found in ~/CNN/Cnn.ipynb. This file
has been built on that origional architecture but contains changes from the QERC development team.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys

def main():
    if len(sys.argv) > 1:
        ds = sys.argv[1]
    else:
        print("Usage: base_CNN.py [model]")

    # Prep data
    colour = False
    
    # Select correct dataset
    if ds == "CIFAR-10":
        print("Selected CIFAR-10")
        colour = True
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
        classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        num_epoch = 200
    elif ds == "F-MNIST":
        print("Selected Fashion MNIST")

        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                            download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                        download=True, transform=transform)
        classes = ('zero', 'one', 'two', 'three', 'four', 'five',
                   'six', 'seven', 'eight', 'nine')
        
        num_epoch = 50
    elif ds == "MNIST":
        print("Selected MNIST")

        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
        classes = ('top', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'boot')
        
        num_epoch = 50
    else:
        print("Please specify a valid dataset")


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)
    

    # Create architecture
    # Need to figure out how to adapt this to black and white
    if colour:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 16 * 5 * 5)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
    else: #This needs to be reworked, the dimensions are garbage

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                            shuffle=True, num_workers=2)

        testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                            shuffle=False, num_workers=2)

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 4 * 4, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 16 * 4 * 4)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

    net = Net()

    # Define Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    # Train network
    for epoch in range(num_epoch):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # Test network
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

if __name__ == "__main__":
    main()