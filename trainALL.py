import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from trainModel import trainModel

""" 
A script to train and compare  accuracy of all models.
"""

if __name__ == "__main__":

    print("Beginning training of All Models.\n")
    
    print("QAE-Nets:")

    print("- CiFAR 1 Layer:")
    qaecifar1_acc = trainModel('QAE_CIFAR-10_1-layer', 'qae', 'CIFAR-10', 200, 1)
    print(qaecifar1_acc, "\n")
    
    print("- CiFAR 2 Layer:")
    qaecifar2_acc = trainModel('QAE_CIFAR-10_2-layer', 'qae', 'CIFAR-10', 200, 2)
    print(qaecifar2_acc, "\n")
    
    print("- CiFAR 3 Layer:")
    qaecifar3_acc = trainModel('QAE_CIFAR-10_3-layer', 'qae', 'CIFAR-10', 200, 3)
    print(qaecifar3_acc, "\n") 
    
    print("- F_MNIST:")
    qaef_mnist_acc = trainModel('QAE_F-MNIST', 'qae', 'FashionMNIST', 50)
    print(qaef_mnist_acc, "\n")

    print("- MNIST")
    qaemnist_acc = trainModel('QAE_MNIST', 'qae', 'MNIST', 50)
    print(qaemnist_acc)


    print("SEN-Nets:\n")
    
    print("- CiFAR: ")
    sencifar_acc = trainModel('SEN_CIFAR-10', 'sen', 'CIFAR-10', 200)
    print(sencifar_acc, "\n")

    print("- F_MNIST: ")
    senf_mnist_acc = trainModel('SEN_F-MNIST', 'sen', 'FashionMNIST', 50)
    print(senf_mnist_acc, "\n")

    print("- MNIST: ")
    senmnist_acc = trainModel('SEN_MNIST', 'sen', 'MNIST', 50)
    print(senmnist_acc, "\n")

    print("FINISHED ALL TRAINING:\n " \
    "-----Summary-----\n" \
    "SEN:\n" \
    "- CiFAR: ", sencifar_acc, 
    "- F_MNIST: ", senf_mnist_acc,
    "- MNIST: ", senmnist_acc,
    "QAE:\n" \
    "- CiFAR1: ", qaecifar1_acc, 
    "- CiFAR2: ", qaecifar2_acc, 
    "- CiFAR3: ", qaecifar3_acc, 
    "- F_MNIST: ", qaef_mnist_acc,
    "- MNIST: ", qaemnist_acc,
    )
