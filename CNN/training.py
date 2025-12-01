import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sen
import time

def train_model(data):
    """
    Heavily influenced by this guide right here:
        https://pythonguides.com/pytorch-mnist/
    """
    BATCH_SIZE = 100
    LEARNING_RATE = 0.001
    if data == "CIFAR":
        EPOCHS = 200
    else:
        EPOCHS = 50
    SUBSET_SIZE = 5000

    # history = {
    #     'epoch': [],
    #     'avg_loss': [],
    #     'accuracy': [],
    #     'test_accuracy': [],
    # }
    writer = SummaryWriter('runs/qae_MNIST_1')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on: {device}")
    if(data == "CIFAR"):
        model = sen.SEN_net(model_cifar=True)
    else:
        model = sen.SEN_net(model_cifar=False)
    model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        #Standard MNIST normalization
        transforms.Normalize((0.1307,), (0.3081)) 
    ])
    if data == "MNIST":
        train_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
            )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=False, 
            transform=transform)
    elif data == "F-MNIST":
        train_dataset = torchvision.datasets.FashionMNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
            )
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data', 
            train=False, 
            transform=transform)
    elif data == "CIFAR":
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
            )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=False, 
            transform=transform)
    else:
        print("ERROR: Please Specify a Valid Dataset!")
        return
    
    # Could use subset for training for efficiency
    # train_dataset = torch.utils.data.Subset(train_dataset, range(SUBSET_SIZE)) 
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # TODO: are these the right optimization methods to be using?
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate Loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()


            if batch_idx % 10 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)

                print(f"Epoch {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # history['epoch'].append(epoch + 1)
        # history['avg_loss'].append(avg_loss)
        # history['accuracy'].append(accuracy)
        
        writer.add_scalar('Loss/train_avg_epoch', avg_loss, epoch)
        writer.add_scalar('Accuracy/train_epoch', accuracy, epoch)

    print("\nEvaluating on Test Set...")
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_accuracy = 100. * test_correct / len(test_loader.dataset)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    # history['test_accuracy'].append(test_accuracy)
    writer.add_scalar('Accuracy/test_epoch', test_accuracy, epoch)

    writer.close()
    return test_accuracy

if __name__ == "__main__":
    # Timing retrieved from: https://docs.python.org/3/library/time.html#time.perf_counter
    time1 = time.perf_counter()

    cifar_acc = train_model("CIFAR")
    f_mnist_acc = train_model("F-MNIST")
    mnist_acc = train_model("MNIST")

    time2 = time.perf_counter()

    print("Finished training all datasets. \n Accuracies are:\n" \
    "- CIFAR_10: ", cifar_acc, "\n- F_MNIST:", f_mnist_acc,
    "\n- MNIST: ", mnist_acc,
    "\nTotal time (seconds) was:", time2-time1)