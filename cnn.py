import torch
import torch.nn as nn
import QAE.qae as qae
import SEN.sen as sen

class CNN(nn.Module):
    """
    Lightweight CNN enhanced with Quantum Channel Attention

    Architecture:
        Conv(12) -> maxPool -> q_attn -> Conv(16) -> maxPool -> FC(256) -> FC(128)
    """
    def __init__(self, model: str = "qae"):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 12, kernel_size=5) #MNIST is 1 channel
        self.pool = nn.MaxPool2d(2)

        if (model == "qae"):
            self.attn = qae.QuantumChannelAttn(channels=12, num_qubits=4, vqc_layers=1)
        elif (model == "sen"):
            self.attn = sen.SEBlock(channels=12, reduction_ratio=3)

        self.conv2 = nn.Conv2d(12,16, kernel_size=5)
        self.flatten_dim = 16 * 4 * 4
        self.fc1 = nn.Linear(self.flatten_dim, 256)
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