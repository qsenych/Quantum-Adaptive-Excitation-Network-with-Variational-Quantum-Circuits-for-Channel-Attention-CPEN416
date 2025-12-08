import torch
import torch.nn as nn
import pennylane as qml

""" 
This module implements the QAE blocked described in the paper
    https://arxiv.org/pdf/2507.11217

The QAE block replaces the classical excitation mechanism in Sqeeze and Excitation
networks with a variational quantum circuit

Components:
    - QuantumAttnBlk: The core VQC implemented with Pennylane
    - QuantumChannelAttn: The full attention block integrating the VQC into a PyTorch nn.module

Authors: Quinn Senych, Evan Nawfal
"""


class QuantumAttnBlk(nn.Module):
    """
    Implements the quantum circuit for channelwise attention.

    This circuit processes the compressed channel information and ouputs expectation
    values that are the attention weights.

    Circuit structure:
        - starts with H on all qubits
        - encodes inputs per qubit as RZ, RY, RZ
        - Variational Layers:
            - applies basic entangling CNOT chain
            - Applies trainable Unitary rotations
        - measures PauliZ expectation values on each qubit

    Inputs:
        num_qubits: number of qubits in the circuit,
                    for this paper this should always be 4
        num_layers: number of variational layers in the model.

    Output:
        Tensor of shape (batch_size, num_qubits)

    """

    def __init__(self, num_qubits: int = 4, num_layers: int = 1):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        # Use default.qubit instead of lightning.gpu which was defined in the paper.
        # This results in significantly faster training times.
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            """
            Quantum circuit definition.
            the shape of inputs is: (batch_size, numqubits * 3) -> (batch_size, 12)
            so we must do some weird indexing to get the angles
                wire0 - uses indices 0,1,2
                wire1 - uses indices 3,4,5
                wire2 - uses indices 6,7,8
                wire3 - uses indices 9,10,11

            parameters:
                inputs: Input features for angle encoding (batch_size x num_qubits*3)
                weights: Trainable parameters for unitaries (num_layers x num_qubits x 3)
            """

            # enc_angles[:, index] gets the column for that parameter across entire batch
            enc_angles = inputs
            for wire in range(self.num_qubits):
                # Uniform superposition
                qml.Hadamard(wires=wire)

                # State preparation
                i = wire * 3
                a1 = enc_angles[:, i]
                a2 = enc_angles[:, i + 1]
                a3 = enc_angles[:, i + 2]
                qml.RZ(a1, wires=wire)
                qml.RY(a2, wires=wire)
                qml.RZ(a3, wires=wire)

            for l in range(num_layers):
                # Basic entangling layer but with unitaries applied after CNOTs (match the paper)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 3])
                qml.CNOT(wires=[3, 0])

                # Trainable rotations
                for wire in range(num_qubits):
                    r0, r1, r2 = weights[l, wire]
                    qml.U3(r0, r1, r2, wires=wire)

            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        weight_shapes = {"weights": (num_layers, num_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        """
        Forward pass of the quantum block, just passes the qml torchlayer
        to pytorch
        """
        return self.qlayer(x)


class QuantumChannelAttn(nn.Module):
    """
    Applies the quantum replacement to the SE-block.
    Replaces the classical 'Excitation' step in an SE block

    Steps:
        - Squeeze channels using global average pooling
        - Reshape into angle format
        - run through QuantumAttnBlk
        - Expand to channel-wise attention weights
        - Multiply attention weights with input activations

    parameters:
        channels: number of input channels (must be 3 * num_qubits)
        num_qubits: qubits used in quantum circuit (defined as 4 in the paper)
        num_layers: trainable quantum layers
    """

    def __init__(self, channels=12, num_qubits=4, vqc_layers=1):
        super().__init__()
        self.channels = channels
        self.num_qubits = num_qubits

        # Squeeze operation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Quantum "excitation"
        self.quantum_encoder = QuantumAttnBlk(
            num_qubits=num_qubits, num_layers=vqc_layers
        )

        # see eq 7 of the paper - restores the dimentions
        # and maps the quantum output to the channel weights
        self.fc = nn.Linear(num_qubits, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward path for pytorch
        """
        b, c, _, _ = x.size()

        # Squeeze operation
        y = self.avg_pool(x).view(b, c)

        # reshape to (Batch_size, 12) instead of (Batch_size, 4, 3)
        y_flattened = y.view(b, self.num_qubits * 3)
        # perform the quantum excitation
        q_out = self.quantum_encoder(y_flattened)

        # Map back to original channel attention iwth a sigmoid activation
        attention_weights = self.sigmoid(self.fc(q_out)).view(b, c, 1, 1)

        # scale
        return x * attention_weights
