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
    def __init__(self, b, c=12):
        super().__init__()
        self.channels = c

        """ Squeeze """
        # Use adaptive pooling since I trust pytorch to make
        # a good kernel size
        # Squeeze down to n x 1 -> source 9
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # TODO: what should dims of the fc layers be?
        """ Excitation """
        self.excitation = nn.Sequential(
            nn.Linear(c, c, bias = False),
            nn.ReLU(inplace = True), # Try to use inplace to improve mem efficiency
            nn.Linear(c, c, bias = False),
            nn.Sigmoid(),
        )


    def forward(self, x):
        b, c, _, _ = x.size()
        #print("Got X size -> b = ", b, "c = ", c)
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c,1,1)

        #TODO: do we want to expand the y in QAE too?
        # "Scale and combine" from source 10
        return x * y.expand_as(x)