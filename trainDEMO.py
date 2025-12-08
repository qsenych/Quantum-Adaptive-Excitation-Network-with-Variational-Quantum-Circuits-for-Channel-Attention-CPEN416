from trainModel import trainModel

""" 
This script runs both our QAEnet and SENet implementation for the purposes
of demoing to the class our model.
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
