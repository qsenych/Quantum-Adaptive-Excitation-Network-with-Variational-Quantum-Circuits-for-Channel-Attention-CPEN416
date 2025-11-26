## Description of VQC

### Justification of using pooling and ReLU in between layers Without Pooling
1. Input: 28×28
2. Conv1 (5×5 kernel, valid padding): 28−5+1=24×24
3. Conv2 (5×5 kernel, valid padding): 24−5+1=20×20
4. Flatten: 16 channels×20×20=6400 inputs.

the linear layer expects 256 inputs, cannot throw 6400 inputs in.

With pooling:    
	Input: 28×28
    Conv1: 24×24
    Pool1 (2×2): 24/2=12×12
    Conv2: 12−5+1=8×8
    Pool2 (2×2): 8/2=4×4
    Flatten: 16 channels×4×4=256 inputs.

## Code overview

### Dependencies
- set up virtual environment
- pip3 install pennylane
- set up pytorch according to this link: https://pytorch.org/get-started/locally/
	Personally my command on ubuntu 24 is this:
	pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
- pip3 install tensorboard

### Some justifications:
the device must be ```"default.qubit"``` not ```"lightning.gpu"``` as defined in the paper. the former is many times faster than the latter.

Even though the paper doesn't mention it, I believe the .relu is necessary because without it 2 linear layers in a row just collapse down to 1


## Resources:
https://www.codegenes.net/blog/pennylane-pytorch/
