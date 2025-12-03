import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from train import trainQAEMNIST
from train import trainQAECiFAR
from train import trainSENMNIST
from train import trainSENCiFAR
from train import trainSENF_MNIST
from train import trainQAEF_MNIST

""" 
A script to train and compare  accuracy of all models.
"""

if __name__ == "__main__":

    print("Beginning training of All Models.\n")

    print("SEN-Nets:\n")
    
    print("- CiFAR: ")
    sencifar_acc = trainSENCiFAR.trainSENCiFAR()
    print(sencifar_acc, "\n")

    print("- F_MNIST: ")
    senf_mnist_acc = trainSENF_MNIST.trainSENF_MNIST()
    print(senf_mnist_acc, "\n")

    print("- MNIST: ")
    senmnist_acc = trainSENMNIST.trainSENMNIST()
    print(senmnist_acc, "\n")

    # print("QAE-Nets:")

    # print("- CiFAR:")
    # qaecifar_acc = trainQAECiFAR.trainQAECiFAR()
    # print(qaecifar_acc, "\n")
    
    # print("- F_MNIST:")
    # qaef_mnist_acc = trainQAEF_MNIST.trainQAEF_MNIST()
    # print(qaef_mnist_acc, "\n")

    # print("- MNIST")
    # qaemnist_acc = trainQAEMNIST.trainQAEMNIST()
    # print(qaemnist_acc)

    print("FINISHED ALL TRAINING:\n " \
    "-----Summary-----\n" \
    "SEN:\n" \
    "- CiFAR: ", senmnist_acc, 
    "- F_MNIST: ", senf_mnist_acc,
    "- MNIST: ", senmnist_acc,
    # "QAE:\n" \
    # "- CiFAR: ", qaemnist_acc, 
    # "- F_MNIST: ", qaef_mnist_acc,
    # "- MNIST: ", qaemnist_acc,
    )
