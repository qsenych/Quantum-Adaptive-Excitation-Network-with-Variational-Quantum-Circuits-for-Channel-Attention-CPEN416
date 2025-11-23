import torch
import torch.nn as nn
import pennylane as qml

class VQC(nn.Module):
    """
      - starts with H on all qubits
      - encodes inputs per qubit as RZ, RY, RZ
      - applies trainable Rot gates and a CNOT chain
      - returns Z expectations (one value per qubit)
    """

    # Leaving as 4 and 1 for now, later we can implement the circuit to take these as arguments
    def __init__(self, num_qubits: int = 4 , num_layers: int = 1):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        self.dev = qml.device("default.qubit", wires=num_qubits)

        # makes weights of the angles to be trained, randn to avoid barren plateaus
        self.weights = nn.Parameter(torch.randn(num_layers, num_qubits, 3))

        self.qnode = qml.QNode(self.quantum_circuit, self.dev, interface="torch", diff_method="parameter-shift")
        
    
    def quantum_circuit(self, enc_angles, weights):
        """ Creates the quantum circuit """

        for wire in range(self.num_qubits):
            qml.Hadamard(wires=wire)

        for wire in range(self.num_qubits):
            angle_0, angle_1, angle_2 = enc_angles[wire]
            qml.RZ(angle_0, wires=wire)
            qml.RY(angle_1, wires=wire)
            qml.RZ(angle_2, wires=wire)

        for l in range(self.num_layers):
            qml.CNOT(wires=[0, 1]) 
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[3, 0])
            
            for wire in range(self.num_qubits):
                rotation_0, rotation_1, rotation_2 = weights[l, wire]
                qml.U3(rotation_0, rotation_1, rotation_2, wires=wire)

        # measures with pauli z observable (-1 to 1)
        expectations = []
        for wire in range(self.num_qubits):
            exp_val = qml.expval(qml.PauliZ(wires=wire))
            expectations.append(exp_val)

        return expectations

    def forward(self, enc_angles):
        batch_outputs = []

        # loop over each sample
        for img in range(enc_angles.shape[0]):
            
            # run circuit on one sample's angles
            result_tensor = self.qnode(enc_angles[img], self.weights)
            batch_outputs.append(result_tensor)

        return torch.stack(batch_outputs, dim=0)
