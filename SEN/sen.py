import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """
    Implements the Squeeze-and-Excitation block from the provided text.
    
    Equations:
        z = F_sq(u) = GlobalAvgPool(u)  (Eq. 2)
        s = F_ex(z, W) = Sigmoid(W2 * ReLU(W1 * z))  (Eq. 3)
        x_tilde = s * u  (Eq. 4)
    """
    def __init__(self, channels, reduction_ratio=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Calculate hidden dimension (bottleneck size)
        # For channels=12 and ratio=3, hidden_size will be 4 (matching your 4 qubits)
        hidden_size = max(1, channels // reduction_ratio)
        
        self.fc = nn.Sequential(
            # W1: Dimensionality reduction
            nn.Linear(channels, hidden_size, bias=False),
            nn.ReLU(inplace=True),
            # W2: Dimensionality restoration
            nn.Linear(hidden_size, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Squeeze: Global Information Embedding
        y = self.avg_pool(x).view(b, c)
        
        # Excitation: Adaptive Recalibration
        y = self.fc(y).view(b, c, 1, 1)
        
        # Scale: Reweighing the channels
        return x * y

class SENet(nn.Module):
    """
    Classical SENet architecture for comparison.
    Replaces the QuantumChannelAttn with the SEBlock.
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 12, kernel_size=5)
        self.pool = nn.MaxPool2d(2)

        # Replaced QuantumChannelAttn with classical SEBlock
        self.se_block = SEBlock(channels=12, reduction_ratio=3)

        self.conv2 = nn.Conv2d(12, 16, kernel_size=5)
        self.flatten_dim = 16 * 4 * 4
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Standard CNN Backbone with SE Block inserted
        x = self.conv1(x)
        x = self.pool(x)

        # Apply Squeeze-and-Excitation
        x = self.se_block(x)

        x = self.conv2(x)
        x = self.pool(x)

        x = x.view(-1, self.flatten_dim)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x 
