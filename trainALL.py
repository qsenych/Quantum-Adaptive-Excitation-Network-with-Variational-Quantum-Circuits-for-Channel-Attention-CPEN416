import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from train import trainQAEMNIST
from train import trainQAECiFAR1Layer
from train import trainQAECiFAR2Layer
from train import trainQAECiFAR3Layer
from train import trainSENMNIST
from train import trainSENCiFAR
from train import trainSENF_MNIST
from train import trainQAEF_MNIST
import time

""" 
A script to train and compare  accuracy of all models.
"""

if __name__ == "__main__":

    print("Beginning training of All Models.\n")

    print("SEN-Nets:\n")
    
    print("- CiFAR: ")
    start = time.time()
    sencifar_acc = trainSENCiFAR.trainSENCiFAR()
    stop = time.time()
    sencifar_time = stop-start
    print(sencifar_acc, "\n")

    print("- F_MNIST: ")
    start = time.time()
    senf_mnist_acc = trainSENF_MNIST.trainSENF_MNIST()
    stop = time.time()
    senf_mnist_time = stop-start
    print(senf_mnist_acc, "\n")

    print("- MNIST: ")
    start = time.time()
    senmnist_acc = trainSENMNIST.trainSENMNIST()
    stop = time.time()
    senmnist_time = stop-start
    print(senmnist_acc, "\n")

    print("QAE-Nets:")

    print("- CiFAR 1 Layer:")
    start = time.time()
    qaecifar1_acc = trainQAECiFAR1Layer.trainQAECiFAR()
    stop = time.time()
    qaecifar1_time = stop-start
    print(qaecifar1_acc, "\n")
    
    print("- CiFAR 2 Layer:")
    start = time.time()
    qaecifar2_acc = trainQAECiFAR2Layer.trainQAECiFAR()
    stop = time.time()
    qaecfiar2_time = stop-start
    print(qaecifar2_acc, "\n")
    
    print("- CiFAR 3 Layer:")
    start = time.time()
    qaecifar3_acc = trainQAECiFAR3Layer.trainQAECiFAR()
    stop = time.time()
    qaecifar3_time = stop-start
    print(qaecifar3_acc, "\n") 
    
    print("- F_MNIST:")
    start = time.time()
    qaef_mnist_acc = trainQAEF_MNIST.trainQAEF_MNIST()
    stop = time.time()
    qaef_mnist_time = stop-start
    print(qaef_mnist_acc, "\n")

    print("- MNIST")
    start = time.time()
    qaemnist_acc = trainQAEMNIST.trainQAEMNIST()
    stop = time.time()
    qaemnist_time = stop-start
    print(qaemnist_acc)

    print("FINISHED ALL TRAINING:\n " \
    "-----Summary-----\n" \
    "SEN:\n" \
    "- CiFAR: ", sencifar_acc,  "TIME: ", sencifar_time,
    "- F_MNIST: ", senf_mnist_acc,  "TIME: ", senf_mnist_time,
    "- MNIST: ", senmnist_acc,  "TIME: ", senmnist_time,
    "QAE:\n" \
    "- CiFAR(1): ", qaecifar1_acc,  "TIME: ", qaecifar1_time,
    "- CiFAR(2): ", qaecifar2_acc,  "TIME: ", qaecfiar2_time,
    "- CiFAR(3): ", qaecifar3_acc,  "TIME: ", qaecifar3_time,
    "- F_MNIST: ", qaef_mnist_acc, "TIME: ", qaef_mnist_time,
    "- MNIST: ", qaemnist_acc, "TIME: ", qaemnist_time 
    )
