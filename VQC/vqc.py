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

        # having issues with GPU
        # self.dev = qml.device("lightning.gpu", wires=num_qubits)
        self.dev = qml.device("default.qubit", wires=num_qubits)

        # makes weights of the angles to be trained, randn to avoid barren plateaus
        # self.weights = nn.Parameter(torch.randn(num_layers, num_qubits, 3).cuda())
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
        # Having significant trouble getting this to appear as a tensor instead of a list
        #      Fix was in the forward method, (using torch.stack twice)
        return (
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1)),
            qml.expval(qml.PauliZ(2)),
            qml.expval(qml.PauliZ(3))
        )

    def forward(self, x):
        batch_outputs = []

        # loop over each sample
        for img in range(x.shape[0]):
            
            # run circuit on one sample's angles
            result_tuple = self.qnode(x[img], self.weights)
            result_tensor = torch.stack(result_tuple).float()
            batch_outputs.append(result_tensor)

        return torch.stack(batch_outputs, dim=0)
    
class QAELayer(nn.Module):
    def __init__(self, channels=12, num_qubits=4, vqc_layers=1):
        super().__init__()
        # number of channels must be num_qubits*3. This is 12 as defined by paper
        self.channels = channels
        self.num_qubits = num_qubits

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.vqc = VQC(num_qubits=num_qubits, num_layers=vqc_layers)

        #see eq 7 of the paper
        self.fc = nn.Linear(num_qubits, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)

        y_reshaped = y.view(b, self.num_qubits, 3)

        q_out = self.vqc(y_reshaped)

        attention_weights = self.sigmoid(self.fc(q_out)).view(b, c, 1, 1)

        return x * attention_weights
    

class InitialCNN_QAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Paper Setup: Conv(12) -> QAE -> Conv(16) 
        
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5) # MNIST is 1 channel
        
        # Insert the Quantum Layer here
        self.qae = QAELayer(channels=12, num_qubits=4, vqc_layers=1)
        
        self.conv2 = nn.Conv2d(12, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 256) # Dimension depends on input image size
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        
        # Apply Quantum Attention
        x = self.qae(x)
        
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()


        #conv(12) -> QAE -> conv(16)
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5) #MNIST is 1 channel
        self.pool = nn.MaxPool2d(2)

        self.qae = QAELayer(channels=12, num_qubits=4, vqc_layers=1)

        self.conv2 = nn.Conv2d(12,16, kernel_size=5)
        self.flatten_dim = 16 * 4 * 4
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)



    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)

        x = self.qae(x)

        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, self.flatten_dim)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    

def test_initialCNN():
    model = InitialCNN_QAE()
    test_input = torch.randn(5, 1, 28, 28)
    output = model(test_input)
    # expected output is 5, 10
    print(f"output shape: {output.shape}")

test_initialCNN()