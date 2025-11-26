import torch
import torch.nn as nn

class SqueezeExciteAttn(nn.Module):
    """
    Build a squeeze and excitation layer to compare performance
    with QAE

    From the paper:
    - Squeeze = global average pooling layer -> condenses spatial info
    - excitation = 2 fully connected layers with non-linear
        activations -> resulting attn weights recalibrate
        channels by emphasizing important features

        Activations -> ReLU, and sigmoid
    """
    def __init__(self, channels = 12): # TODO: Need more params?
        super().__init__()
        self.channels = channels

        """ Squeeze """
        # Use adaptive pooling since I trust pytorch to make
        # a good kernel size
        # Squeeze down to n x 1 -> source 9
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        """ Excitation """
        self.fc1 = nn.Linear() # TODO: figure out correct dims
        self.fc2 = nn.Linear() # TODO: figure out correct dims
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        


    
    def forward(self,x):
        b, c, _, _ = x.size()

        y = self.squeeze(x).view(b,c)
        # TODO: Is this the correct way to apply the activation functions?
        # Also what should dims of the fc layers be?
        y = self.fc1(y)
        y = self.relu(y)

        y = self.fc2(y)
        y = self.sigmoid(y)
 

        #TODO: does this correctly multiply the weights?
        # "Scale and combine" from source 9
        return x * y

class SEN_net(nn.Module):
    """
    Lightweight CNN enhanced with Quantum Channel Attention

    Architecture:
        Conv(12) -> maxPool -> S&E -> Conv(16) -> maxPool -> FC(256) -> FC(128)
    """
    def __init__(self):
        super().__init__()

        # First paremter is # of channels, will go up to 3 for CIFAR-10
        # Does this mean second parameter is 36?
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5) #MNIST is 1 channel

        self.pool = nn.MaxPool2d(2)

        self.squeeze_excite = SqueezeExciteAttn(channels=12, num_qubits=4, vqc_layers=1)

        # 16 as mentioned in the paper
        self.conv2 = nn.Conv2d(12,16, kernel_size=5)
        self.flatten_dim = 16 * 4 * 4
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 128)

        # Note sure of exactly what this does, but it 
        # is mentioned as being done before the classifcation layer in the paper
        self.dropout = nn.Dropout(0.5)
        # Back to 10, because 10 categories
        self.fc3 = nn.Linear(128, 10)
        
    # This actually runs the network
    def forward(self, x):
        #TODO: Double check that doing .relu is correct
        x = torch.relu(self.conv1(x))
        x = self.pool(x)

        x = self.squeeze_excite(x)

        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, self.flatten_dim)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x