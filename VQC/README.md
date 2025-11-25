## Description of VQC

### Dependencies
- set up virtual environment
- pip3 install pennylane
- set up pytorch according to this link: https://pytorch.org/get-started/locally/
	Personally my command on ubuntu 24 is this:
	pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
- pip3 install tensorboard


Depending on your platform some changes may need to be made to the qubit device
Either "lightning.gpu" if you have a gpu or "default.qubit" for cpu only
