import sys
import os
from medmnist import BreastMNIST
from medmnist import BloodMNIST
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'A')))
from A.model_A import CNN_A

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'B')))
from B.model_B import CNN_B

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LOAD MODEL A
transform_A = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to mean 0 and std 1
])

test_dataset_A = BreastMNIST(split="test", transform=transform_A, download=True)
val_dataset_A = BreastMNIST(split="val", transform=transform_A, download=True)
test_loader_A = torch.utils.data.DataLoader(dataset=test_dataset_A, batch_size=32, shuffle=False)
val_loader_A = torch.utils.data.DataLoader(dataset=val_dataset_A, batch_size=32, shuffle=False)


model_A = CNN_A().to(device)
model_A.load_state_dict(torch.load('AMLS_24-25_SN24249071/A/model_A_best.pth', weights_only=True))
model_A.eval()

# LOAD MODEL B
transform_B = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to mean 0 and std 1
])

val_dataset_B = BloodMNIST(split="val", transform=transform_B, download=True)
val_loader_B = torch.utils.data.DataLoader(dataset=val_dataset_B, batch_size=64, shuffle=False)
test_dataset_B = BloodMNIST(split="test", transform=transform_B, download=True)
test_loader_B = torch.utils.data.DataLoader(dataset=test_dataset_B, batch_size=64, shuffle=False)

model_B = CNN_B().to(device)
model_B.load_state_dict(torch.load('AMLS_24-25_SN24249071/B/model_B_best.pth', weights_only=True))
model_B.eval()



def run_models_test():
    print("A Validation accuracy: ", model_A.evaluate_accuracy(val_loader_A, device))
    print("A Test accuracy: ", model_A.evaluate_accuracy(test_loader_A, device))


    print("B Validation accuracy: ", model_B.evaluate_accuracy(val_loader_B, device))
    print("B Test accuracy: ", model_B.evaluate_accuracy(test_loader_B, device))


def get_class_subset_loader(dataset, target_label, batch_size=64):
    """
    Returns a DataLoader for all samples in 'dataset' that match 'target_label'.
    """
    # Collect all indices where the sample's label matches 'target_label'
    indices = [i for i, (_, label) in enumerate(dataset) if label == target_label]
    
    # Create a Subset of just those samples
    subset = torch.utils.data.Subset(dataset, indices)

    # Return a DataLoader from the subset
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)


def per_label_accuracy(model, dataset, num_classes):
    print(model.__class__.__name__)
    for class_label in range(num_classes):
        # Get DataLoader containing only samples from the current class
        subset_loader = get_class_subset_loader(dataset, class_label, batch_size=64)
        
        print(f"Class {class_label}:", model.evaluate_accuracy(subset_loader, device))


if __name__=="__main__":


    run_models_test()

    per_label_accuracy(model_A, test_dataset_A, 2)

    per_label_accuracy(model_B, test_dataset_B, 8)