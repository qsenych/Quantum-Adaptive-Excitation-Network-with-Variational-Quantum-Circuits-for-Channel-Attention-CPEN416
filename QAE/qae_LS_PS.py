import torch
import torch.nn as nn
import pennylane as qml



class QuantumAttnBlk(nn.Module):
    """
    Quantum circuit for channel-wise attention.
    """

    def __init__(self, num_qubits=4, num_layers=1):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

    
        self.dev = qml.device(
            "lightning.qubit",
            wires=num_qubits,
            
        )

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(inputs, weights):
            # inputs shape: (12,) for 4 qubits × 3 angles
            # weights shape: (layers, qubits, 3)

            # Encoding
            for wire in range(self.num_qubits):
                qml.Hadamard(wire)
                i = wire * 3
                a1, a2, a3 = inputs[i:i+3]
                qml.RZ(a1, wire)
                qml.RY(a2, wire)
                qml.RZ(a3, wire)

            # Trainable layers
            for l in range(self.num_layers):
                qml.CNOT([0, 1])
                qml.CNOT([1, 2])
                qml.CNOT([2, 3])
                qml.CNOT([3, 0])

                for q in range(self.num_qubits):
                    r0, r1, r2 = weights[l, q]
                    qml.U3(r0, r1, r2, wires=q)

            return [qml.expval(qml.PauliZ(w)) for w in range(self.num_qubits)]

        weight_shapes = {"weights": (num_layers, num_qubits, 3)}

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        """
        x: (batch, 12)
        QNode only accepts CPU tensors and single samples.
        """
        batch = x.shape[0]

        # Move to CPU for PennyLane
        x_cpu = x.detach().cpu()

        outputs = []
        for i in range(batch):
            out = self.qlayer(x_cpu[i])   # returns (num_qubits,)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=0)

        # Move back to GPU if needed
        return outputs.to(x.device)




class QuantumChannelAttn(nn.Module):
    """
    Quantum-enhanced SE block.
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

        # Map quantum output → channel weights
        # Eq 7 in QAE paper
        self.fc = nn.Linear(num_qubits, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape

        # Squeeze
        y = self.avg_pool(x).view(b, c)  # (batch, 12)

        # Reshape for quantum encoder (already (b,12))
        q_out = self.quantum_encoder(y.float())  # (batch, 4)

        # Expand to channels
        attention = self.sigmoid(self.fc(q_out)).view(b, c, 1, 1)

        return x * attention
