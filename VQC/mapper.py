import torch
import torch.nn as nn

class Mapper(nn.Module):

    def __init__(self, channels: int, num_qubits: int = 4):
        super().__init__()
        self.channels = channels
        self.num_qubits = num_qubits
        # not done yet

    def map_cnn_to_angles(self, cnn_in):
        # not sure how to implement yet, need to understand input format
        return torch.zeros(cnn_in.shape[0], self.num_qubits, 3)

    def map_quantum_to_cnn(self, q_out):
        # not sure how to implement yet, need to understand output format
        return torch.zeros(q_out.shape[0], self.channels, 1, 1)