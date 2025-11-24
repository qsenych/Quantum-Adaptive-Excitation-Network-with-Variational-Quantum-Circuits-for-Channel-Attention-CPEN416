import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pennylane as qml
import vqc

# WARNING: AI was used for this script
def train_model():
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on: {device}")

    model = vqc.MNISTModel().to(device)

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

    # Use a subset for quick testing if you don't have a GPU
    # train_dataset = torch.utils.data.Subset(train_dataset, range(100)) 
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Setup ---
    model = vqc.MNISTModel()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    
    # --- Training Loop ---
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
                print(f"Epoch {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                      f"Loss: {loss.item():.4f}")

        # --- Epoch Stats ---
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # --- Evaluation ---
    print("\nEvaluating on Test Set...")
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()

    print(f"Test Accuracy: {100. * test_correct / len(test_loader.dataset):.2f}%")

if __name__ == "__main__":
    train_model()