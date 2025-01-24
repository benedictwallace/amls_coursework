
This file collection contains everything necessary to run and train models for the BloodMNIST and BreastMNIST datasets.

My Git repo was not allowing me to upload some .pth files, I panicked and acidently deleted the whole repo and created a new one. 
If there is a way to find the old commits and comments please let me know, I'm not that familiar with git but I had loadds of comits before accidently deleting them
I was stupid to follow ChatGPTs solutions to my git problems.


The layout is as follows:

/A/ 
    /augmented_dataset/
        This contains all images with class labels that have been augmented. It's removed and created again when running create_augmented_dataset() from augment_dataset.py
    /hypertuning.py 
        This script has all functions to train models for hyper tuning.
    /model_A.py
        This contains the code for the model and all relevant functions for that model.
    /visualise.py
        This contains scripts to allow the user to visualise various convolution layers feature maps.
    /model_A_best.pth
        This is the pre-trainied model that has performed best out of a number of attempts.
    /figures/
        This folder contains all relevant figures and charts about the models training.
    
    There may be other files or folders but these can be ignored.

/B/ 
    /hypertuning.py 
        This script has all functions to train models for hyper tuning.
    /model_B.py
        This contains the code for the model and all relevant functions for that model.
    /visualise.py
        This contains scripts to allow the user to visualise various convolution layers feature maps.
    /model_B_best.pth
        This is the pre-trainied model that has performed best out of a number of attempts.
    /figures/
        This folder contains all relevant figures and charts about the models training.

    There may be other files or folders but these can be ignored.

/Datasets/
    Blank folder to add relevant datasets, not that my models use the data aquired from the MedMNIST modules so paths have to be changed.

/final_accuracies.py   
    This script runs the best models and calculates the validation and test accuracies. Also breaks down accuracy by class.

/main.py    
    Runs pre-trained models or trains new ones and prints accuracies. mode = 0 for loading models, mode = 1 for training new.

There may be other files/folders but these can be ignored.

Required Modules:

torch
torch.nn
medmnist
sys
os
torchvision
torch.optim
matplotlib.pyplot
numpy
torchvision.datasets
shutil
pathlib