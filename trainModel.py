import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cnn

def trainModel(run_name: str, model_type: str = 'qae', dataset: str = 'MNIST', num_epochs: int = 50, num_layers: int = 1):
    """
    Heavily influenced by this guide right here:
        https://pythonguides.com/pytorch-mnist/
    """
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.001
    # Every BATCH_SAMPLING_RES  batches it records a value
    BATCH_SAMPLING_RES = 6

    writer = SummaryWriter('runs/' + run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on: {device}")

    if dataset == 'CIFAR-10':
        model = cnn.CNN_CIFAR(model_type, vqc_layers=num_layers)
    else:
        model = cnn.CNN(model_type)

    model.to(device)

    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            #Standard MNIST normalization
            transforms.Normalize((0.1307,), (0.3081)) 
        ])

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
    elif dataset == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            #Standard FMNIST normalization
            transforms.Normalize((0.5,), (0.5)) 
        ])

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
    elif dataset == 'CIFAR-10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            #Standard CIFAR-10 normalization
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ])

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


    
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    
    model.train()
    for epoch in range(num_epochs):
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


            if (batch_idx + 1) % BATCH_SAMPLING_RES == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)

                print(f"Epoch {epoch+1} [{(batch_idx + 1) * len(data)}/{len(train_loader.dataset)}] "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        writer.add_scalar('Loss/train_avg_epoch', avg_loss, epoch)
        writer.add_scalar('Accuracy/train_epoch', accuracy, epoch)

    print("\nEvaluating on Test Set...")
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_accuracy = 100. * test_correct / len(test_loader.dataset)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')

    writer.add_scalar('Accuracy/test_epoch', test_accuracy, epoch)

    writer.close()

    return test_accuracy

if __name__ == "__main__":
    trainModel('QAE_MNIST', 'qae', 'MNIST', 1, 1)

