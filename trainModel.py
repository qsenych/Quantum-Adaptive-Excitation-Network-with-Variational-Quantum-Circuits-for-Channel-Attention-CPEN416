import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cnn


def trainModel(
    run_name: str,
    model_type: str = "qae",
    dataset: str = "MNIST",
    num_epochs: int = 50,
    num_layers: int = 1,
):
    """
    This script handles the training and evaluation loops for the project.
    Includes Tensorboard logging for analysis. See README for instructions
    on how to access TensorBoard

    Heavily influenced by this guide right here:
        https://pythonguides.com/pytorch-mnist/

    Args:
        run_name: Enter the name of the run to be logged in TensorBoard
            Repeating a name will overwrite the previous entry.
        model_type: Choose the type of attention mechanism you want to use:
            'qae': quantum attention mechanism
            'sen': classical squeeze and excitation attention mechanism
        dataset: Choose which dataset to train the model on
            'MNIST': Choose the MNIST dataset
            'FashionMNIST': Choose the FashionMNIST dataset
            'CIFAR-10': Choose the CIFAR-10 dataset
        num_epochs: Choose how many epochs to train the model for
            default is 50, however for CIFAR-10 it is recommended to run for 200 to match the paper
        num_layers: number of VQC layers, only relevant if the model_type is 'qae'
            paper only varies the layers for the CIFAR-10 dataset.


    Author: Quinn Senych, Robert Walsh
    """

    BATCH_SIZE = 1024
    LEARNING_RATE = 0.001
    # Every BATCH_SAMPLING_RES batches it records a value
    BATCH_SAMPLING_RES = 6

    writer = SummaryWriter("runs/" + run_name)

    # Configure devices if cuda device available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on: {device}")

    # Initialize the model depending on the dataset
    if dataset == "CIFAR-10":
        model = cnn.CNN_CIFAR(model_type, vqc_layers=num_layers)
    else:
        model = cnn.CNN(model_type)

    model.to(device)

    # Load the appropriate dataset
    if dataset == "MNIST":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # Standard MNIST normalization
                transforms.Normalize((0.1307,), (0.3081)),
            ]
        )

        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, transform=transform
        )
    elif dataset == "FashionMNIST":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # Standard FMNIST normalization
                transforms.Normalize((0.5,), (0.5)),
            ]
        )

        train_dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, transform=transform
        )
    elif dataset == "CIFAR-10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # Standard CIFAR-10 normalization
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, transform=transform
        )

    # pin_memory will cause a warning when not on GPU, but has marginal speed up when using it
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )

    # paper only specifies gradient descent with cross entropy loss,
    # Adam does gradient descent + adaptive moment estimation
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("Starting training")

    # Main training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to GPU if available
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate Loss
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Logging
            if (batch_idx + 1) % BATCH_SAMPLING_RES == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Loss/train_batch", loss.item(), global_step)

                print(
                    f"Epoch {epoch+1} [{(batch_idx + 1) * len(data)}/{len(train_loader.dataset)}] "
                    f"Loss: {loss.item():.4f}"
                )

        # Summarize the epoch
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / len(train_loader.dataset)
        print(
            f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.5f}, Accuracy: {accuracy:.2f}%"
        )

        writer.add_scalar("Loss/train_avg_epoch", avg_loss, epoch)
        writer.add_scalar("Accuracy/train_epoch", accuracy, epoch)

    # Evaluation
    print("\nEvaluating on Test Set...")
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_accuracy = 100.0 * test_correct / len(test_loader.dataset)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Extract number of parameters to compare to paper
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    writer.add_scalar("Accuracy/test_epoch", test_accuracy, epoch)
    writer.close()

    return test_accuracy


if __name__ == "__main__":
    # Quick and dirty testing
    trainModel("QAE_MNIST", "qae", "MNIST", 10, 1)
