import torch
import torch.nn as nn
import pennylane as qml

""" 
TODO:
    Ensure Layer increases work (Especially with CIFAR)
    Standardize docstrings and improve inline documentation
"""

class QuantumAttnBlk(nn.Module):
    """
    Implements the quantum circuit for channelwise attention.

    Inputs:
        x : tensor of shape (batch_size, num_qubits)
            PauliZ expectation values used to weight the classical channels
    
    Output:
        Tensor of shape (batch_size, num_qubits)

    Circuit structure:
        - starts with H on all qubits
        - encodes inputs per qubit as RZ, RY, RZ
        - applies basic entangling CNOT chain
        - Applies trainable Unitary rotations
        - measures PauliZ on each qubit
    """

    def __init__(self, num_qubits: int = 4 , num_layers: int = 1):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        dev = qml.device("lightning.gpu", wires=num_qubits)

        # TODO: Double check and verify that doing "adjoint" as diff_method is
        # equivelent mathematically to paramater_shift
        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def circuit(inputs, weights):
            # the shape of inputs is: (batch_size, numqubits * 3) -> (batch_size, 12)
            # so we must do some weird indexing to get the angles
            #     wire0 - uses indices 0,1,2
            #     wire1 - uses indices 3,4,5
            #     wire2 - uses indices 6,7,8
            #     wire3 - uses indices 9,10,11
            
            # enc_angles[:, index] gets the column for that parameter across entire batch
            enc_angles = inputs
            for wire in range(self.num_qubits):
                qml.Hadamard(wires=wire)

                i = wire * 3
                a1 = enc_angles[:, i]
                a2 = enc_angles[:, i + 1]
                a3 = enc_angles[:, i + 2]
                qml.RZ(a1, wires=wire)
                qml.RY(a2, wires=wire)
                qml.RZ(a3, wires=wire)

            for l in range(num_layers):
                qml.CNOT(wires=[0, 1]) 
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 3])
                qml.CNOT(wires=[3, 0])
                
                for wire in range(num_qubits):
                    r0, r1, r2 = weights[l, wire]
                    qml.U3(r0, r1, r2, wires=wire)

            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
        
        weight_shapes = {"weights": (num_layers, num_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        return self.qlayer(x)
        
class QuantumChannelAttn(nn.Module):
    """
    Applies the quantum replacement to the SE-block.

    Steps:
        - Squeeze channels using global average pooling
        - Reshape into angle format
        - run through QuantumAttnBlk
        - Expand to channel-wise attention weights
        - Multiply attention weights with input activations

    parameters:
        channels - number of input channels (must be 3 * num_qubits)
        num_qubits - qubits used in quantum circuit (defined as 4 in the paper)
        num_layers - trainable quantum layers
    """
    def __init__(self, channels=12, num_qubits=4, vqc_layers=1):
        super().__init__()
        self.channels = channels
        self.num_qubits = num_qubits

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.quantum_encoder = QuantumAttnBlk(
            num_qubits=num_qubits, 
            num_layers=vqc_layers
        )

        #see eq 7 of the paper
        self.fc = nn.Linear(num_qubits, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)

        # reshape to (Batch_size, 12) instead of (Batch_size, 4, 3)
        y_flattened = y.view(b, self.num_qubits * 3)
        q_out = self.quantum_encoder(y_flattened.float())

        attention_weights = self.sigmoid(self.fc(q_out)).view(b, c, 1, 1)

        return x * attention_weights
    

class QAE_net(nn.Module):
    """
    Lightweight CNN enhanced with Quantum Channel Attention

    Architecture:
        Conv(12) -> maxPool -> q_attn -> Conv(16) -> maxPool -> FC(256) -> FC(128)
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 12, kernel_size=5) #MNIST is 1 channel
        self.pool = nn.MaxPool2d(2)

        self.q_attn = QuantumChannelAttn(channels=12, num_qubits=4, vqc_layers=1)

        self.conv2 = nn.Conv2d(12,16, kernel_size=5)
        self.flatten_dim = 16 * 4 * 4
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #TODO: Double check that doing .relu is correct
        x = torch.relu(self.conv1(x))
        x = self.pool(x)

        x = self.q_attn(x)

        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, self.flatten_dim)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    