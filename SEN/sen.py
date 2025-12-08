import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Sources:
    - https://medium.com/@tahasamavati/squeeze-and-excitation-explained-387b5981f249

        This source provided the source code for the squeeze and excitation block below.
        The only change we made was to define a hidden_size variable to allow
        the network dimensions to fit our application correctly.

    - J. Hu, L. Shen and G. Sun, "Squeeze-and-Excitation Networks,"
        2018 IEEE/CVF Conference on Computer Vision and Pattern
        Recognition, Salt Lake City, UT, USA, 2018, pp. 7132-7141,
        doi: 10.1109/CVPR.2018.00745.

    The origional paper implies the reduction ratio is 3 to match the quantum parameter counts

    Equations:
        z = F_sq(u) = GlobalAvgPool(u)  (Eq. 2)
        s = F_ex(z, W) = Sigmoid(W2 * ReLU(W1 * z))  (Eq. 3)
        x_tilde = s * u  (Eq. 4)

    Authors: Robert Walsh, Quinn Senych
    """

    def __init__(self, c=12, r: int = 3):
        super().__init__()
        self.channels = c

        # Squeeze
        # Use adaptive pooling since we trust pytorch to make
        # a good kernel size
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        hidden_size = max(1, c // r)

        # Excitation
        self.excitation = nn.Sequential(
            nn.Linear(c, hidden_size, bias=False),
            nn.ReLU(inplace=True),  # Try to use inplace to improve mem efficiency
            nn.Linear(hidden_size, c, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)

        # "Scale and combine"
        return x * y.expand_as(x)
