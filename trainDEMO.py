import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from trainModel import trainModel

""" 
A script to train and compare  accuracy of all models.
"""

if __name__ == "__main__":

    print("Begin training for the demo\n")
    
    print("QAE-Nets:")
    
    print("- MNIST")
    qaemnist_acc = trainModel('QAE_MNIST', 'qae', 'MNIST', 50)
    print(qaemnist_acc)


    print("SEN-Nets:\n")

    print("- MNIST: ")
    senmnist_acc = trainModel('SEN_MNIST', 'sen', 'MNIST', 50)
    print(senmnist_acc, "\n")

    print("FINISHED ALL TRAINING:\n " \
    "-----Summary-----\n" \
    "SEN:\n" \
    # "- CiFAR: ", sencifar_acc, 
    # "- F_MNIST: ", senf_mnist_acc,
    "- MNIST: ", senmnist_acc,
    "QAE:\n" \
    # "- CiFAR1: ", qaecifar1_acc, 
    # "- CiFAR2: ", qaecifar2_acc, 
    # "- CiFAR3: ", qaecifar3_acc, 
    # "- F_MNIST: ", qaef_mnist_acc,
    "- MNIST: ", qaemnist_acc,
    )
