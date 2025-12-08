import torch
import torch.nn as nn
import QAE.qae as qae
import SEN.sen as sen

"""
Defines the backbone CNN for MNIST/FashionMNIST and CIFAR-10 datasets.

Key Design Decision:
    Two separate classes are implemented (`CNN` and `CNN_CIFAR`) because the 
    input image dimensions (28x28 vs 32x32) result in different flattend
    feature vector sizes

Author: Quinn Senych
"""

class CNN(nn.Module):
    """
    Lightweight CNN enhanced with either quantum or SE channel attention.
    For MNIST/FashionMNIST (28x28 inputs)

    Architecture:
        Conv(12) -> maxPool -> attn -> Conv(16) -> maxPool -> FC(256) -> FC(128)

    Args:
        model: Type of attention block to use
            "qae": use a quantum attention block
            "sen": use a classical squeeze and excitation attention block
    """
    def __init__(self, model: str = "qae"):
        super().__init__()
        
        # 1 input channel, kernel size 5 defined by paper 
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5) 
        # Implied pooling layer for dimentions to work out
        self.pool = nn.MaxPool2d(2)

        # Attentino mechanism selection
        if (model == "qae"):
            self.attn = qae.QuantumChannelAttn(channels=12, num_qubits=4, vqc_layers=1)
        elif (model == "sen"):
            # paper implies reduction ratio of 3 to match quantum parameter counts
            self.attn = sen.SEBlock(c=12, r=3)

        self.conv2 = nn.Conv2d(12,16, kernel_size=5)

        # 28x28 -> Conv1(5x5) -> 24x24 -> Pool(2) -> 12x12
        # -> Conv2(5x5) -> 8x8 -> Pool(2) -> 4x4
        # for 16 channels * 4 * 4 = 256
        self.flatten_dim = 16 * 4 * 4
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)

        x = self.attn(x)

        x = self.conv2(x)
        x = self.pool(x)

        x = x.view(-1, self.flatten_dim)
        x = torch.relu(self.fc1(x))
        # Paper mention dropout is applied before classification
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNN_CIFAR(nn.Module):
    """
    Lightweight CNN enhanced with either quantum or SE channel attention.
    For CIFAR-10 (32x32 inputs)

    Architecture:
        Conv(12) -> maxPool -> attn -> Conv(16) -> maxPool -> FC(256) -> FC(128)

    Design Note:
        For CIFAR-10, the larger input size (32x32) creates a flattened vector of 400.
        To match the paper's parameter count, we use an extra FC layer 
        and disable the bias on the first FC layer (400->256).

    Args:
        model: Type of attention block to use
            "qae": use a quantum attention block
            "sen": use a classical squeeze and excitation attention block
        vqc_layers: number of VQC layers if using a "qae" model
            Paper suggests 3 is best but in our implementation 2 is best
    """
    def __init__(self, model: str = "qae", vqc_layers: int = 1):
        super().__init__()

        # CIFAR has 3 input channels (RGB)
        self.conv1 = nn.Conv2d(3, 12, kernel_size=5) 
        self.pool = nn.MaxPool2d(2)

        if (model == "qae"):
            self.attn = qae.QuantumChannelAttn(channels=12, num_qubits=4, vqc_layers=vqc_layers)
        elif (model == "sen"):
            self.attn = sen.SEBlock(c=12)

        self.conv2 = nn.Conv2d(12,16, kernel_size=5)
        
        # must be good for 32x32 input
        # 32x32 -> Conv1 -> 28x28 -> Pool -> 14x14
        # -> Conv2 -> 10x10 -> Pool -> 5x5
        # Final: 16 channels * 5 * 5 = 400
        self.flatten_dim = 16 * 5 * 5
        self.fc1 = nn.Linear(self.flatten_dim, 256, bias=False)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #TODO: Double check that doing .relu is correct
        x = self.conv1(x)
        x = self.pool(x)

        x = self.attn(x)

        x = self.conv2(x)
        x = self.pool(x)

        x = x.view(-1, self.flatten_dim)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x