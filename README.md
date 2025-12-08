# Quantum Adaptive Excitation Network with Variational Quantum Circuits for Channel Attention 

## Overview
This repo contains an implementation replicating a Quantum Adaptive 
Excitation Network (QAE-Net), a hybrid quantum-classical neural network 
architecture done as a final project in CPEN 416 at UBC in winter 2025.

Implementation based on this paper (https://arxiv.org/pdf/2507.11217): 

Yu-Chao Hsu, Kuan-Cheng Chen, Tai-Yue Li, and Nan-Yow Chen, “Quantum Adaptive Excitation Network with Variational Quantum Circuits for Channel Attention,” arXiv preprint arXiv:2507.11217, 2025. 

The core of the project is replacing the classical Squeeze and Excitation
attention block with a variation quantum circuit (VQC).

## Architeture and Implementation

### Model Structure
There are 2 variations of the CNN backbone to match the necessary differences
between the CIFAR-10 dataset and the 2 *MNIST datasets.
1. MNIST / FashionMNIST Model:
- Input: 28x28 Grayscale images.
- Structure: Conv(12) -> Attention block -> Conv(16) -> Flatten -> FC(256, 128) -> FC(128,10)
- Parameter count: ~39k
2. CIFAR-10 Model:
- Input: 32x32 RGB images.
- Structure: Conv(12) -> Attention block -> Conv(16) -> Flatten -> FC(400,256) -> FC(256, 128) -> FC(128, 10)
- Parameter count: ~142k

The model was built around pytorch as it was what was used by the paper in 
addition to its integration with our quantum framework: Pennylane.

In pennylane, the device we used was `default.qubit` even though the 
paper's implementation used `lightning.gpu`. However, we noticed significant
performance improvment when using `default.qubit`. Additionally, we do not do
parameter shift differentiation as it does not work with our batching implemntation
this drives training times up from ~45 minutes to 50+ hours for CIFAR-10 and
from ~10 minutes to ~20 hours for FashionMNIST and MNIST.

Furthermore, the paper does not specify pooling layers in the CNN, but they 
are necessary. Without the cooing layers the input to the hidden layers 
would be way too large.

## Installation and Use

### Dependencies
- set up virtual environment
    
    ```python3 -m venv .venv```

- Install pennylane

    ```pip3 install pennylane```

- Install PyTorch (adjust index-url for your cuda version or CPU)

    ```pip3 install torch torchvision --index-url (https://download.pytorch.org/whl/cpu)```
- Install tensorboard for viewing 

    ```pip3 install tensorboard```

### Training
To train the model use the provided training scripts.

1. trainModel.py: Contatins the main training and evaluation logic, trainModel.trainModel can be called like this:

```
from trainModel import TrainModel

# Train QAE-Net on CIFAR-10 with 2 vqc layers:
trainModel(run_name="QAE_CIFAR-10_2-layers_Run1", model_type="qae", dataset="CIFAR-10", num_epochs=200, num_layers=2)

# Train classical SENet on MNIST (default number of epochs is 50):
trainModel(SEN_MNIST_Run1", model_type="sen", dataset="MNIST", num_layers=50)
```

### View TensorBoard logs
To view metrics after running run this command:
(Note: One must install tensorboard first)
`tensorboard --logdir runs`.

The data was then downloaded from tensorboard and turned into images using the matlab script csv_to_image.m.

The runs that produced the result are in the runs folder in the repo. The produced csvs are in runs/csv/ though these are manually extracted from tensorboard. 

## Resources:
*Useful Resources*
---
1. Paper on SAE Networks -> https://ieeexplore.ieee.org/document/8578843 
2. Video on SAE Networks -> https://www.youtube.com/watch?v=3b7kMvrPZX8&fbclid=IwY2xjawN4pHZleHRuA2FlbQIxMQBicmlkETFaamJ2cEpidzhSbkxGb0Fpc3J0YwZhcHBfaWQBMAABHqkELARfeVnz2nQ9wWiVohou_sCBeOP10Y0pXJuuzYn8IyqIb7JR-5W_690v_aem_KrmOu12QmV42G2iyyQSMXg
3. Article explaining how to build a CNN w/ pytorch, and example of basic network https://www.geeksforgeeks.org/deep-learning/building-a-convolutional-neural-network-using-pytorch/ 
4. CIFAR 10 Dataset -> https://www.cs.toronto.edu/~kriz/cifar.html Need to cite from here: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf 
5. Fully connected vs convolutional layers -> https://www.geeksforgeeks.org/deep-learning/fully-connected-layer-vs-convolutional-layer/ 
6. F-MNIST details -> https://www.geeksforgeeks.org/deep-learning/fashion-mnist-with-python-keras-and-deep-learning/ 
7. MNIST details -> https://www.geeksforgeeks.org/machine-learning/mnist-dataset/ 
8. Pythorch transforms docs -> https://docs.pytorch.org/vision/0.9/transforms.html 
9. Article on fully connected layers ->
https://www.geeksforgeeks.org/deep-learning/what-is-fully-connected-layer-in-deep-learning/ 
10. Squeeze and Excitation Network -> https://medium.com/@tahasamavati/squeeze-and-excitation-explained-387b5981f249 
11. Explanation of Variational Quantum Algorithms -> https://pennylane.ai/codebook/variational-quantum-algorithms/parametrized-quantum-circuits
12. Pennylane pytorch integration -> https://www.codegenes.net/blog/pennylane-pytorch/
--- 
*Sources Used*
3,6,7,9,10,11,12
