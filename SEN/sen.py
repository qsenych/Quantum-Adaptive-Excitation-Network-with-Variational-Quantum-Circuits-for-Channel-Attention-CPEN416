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
        
        # For channels=12 and ratio=3, hidden_size will be 4 matching the 4 qubits
        hidden_size = max(1, channels // reduction_ratio)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_size, bias=False),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden_size, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Squeeze 
        y = self.avg_pool(x).view(b, c)
        
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y