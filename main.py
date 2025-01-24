import sys
import os
from medmnist import BreastMNIST
from medmnist import BloodMNIST
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'A')))
from A.model_A import CNN_A, EarlyStopping_A
from A.augment_dataset import create_augmented_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'B')))
from B.model_B import CNN_B, EarlyStopping_B

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# mode 0 to use pre-trained models, 1 to train new model.
mode = 1
create_augmented_dataset()

transform_A = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to mean 0 and std 1
])

train_dataset_A = ImageFolder(root="/AMLS_24-25_SN24249071/A/augmented_dataset/breast", transform=transform_A, target_transform=lambda y: torch.tensor([y], dtype=torch.long))
train_loader_A = torch.utils.data.DataLoader(dataset=train_dataset_A, batch_size=32, shuffle=True)

val_dataset_A = BreastMNIST(split="val", transform=transform_A, download=True)
val_loader_A = torch.utils.data.DataLoader(dataset=val_dataset_A, batch_size=32, shuffle=False)

test_dataset_A = BreastMNIST(split="test", transform=transform_A, download=True)
test_loader_A = torch.utils.data.DataLoader(dataset=test_dataset_A, batch_size=32, shuffle=False)

transform_B = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to mean 0 and std 1
])

train_dataset_B = BloodMNIST(split="train", transform=transform_B, download=True)
train_loader_B = torch.utils.data.DataLoader(dataset=train_dataset_B, batch_size=64, shuffle=False)

val_dataset_B = BloodMNIST(split="val", transform=transform_B, download=True)
val_loader_B = torch.utils.data.DataLoader(dataset=val_dataset_B, batch_size=64, shuffle=False)

test_dataset_B = BloodMNIST(split="test", transform=transform_B, download=True)
test_loader_B = torch.utils.data.DataLoader(dataset=test_dataset_B, batch_size=64, shuffle=False)



# If using pre-saved 
if mode == 0:
    epochs = 15

    # MODEL A PRETRAINED
    criterion_A = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping_A(patience=10)
    model_A = CNN_A(early_stopping_on=True).to(device)
    model_A.load_state_dict(torch.load('AMLS_24-25_SN24249071/A/model_A_best.pth', weights_only=True))


    # MODEL B PRE TRAINED
    criterion_B = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping_B(patience=10)
    model_B = CNN_B(early_stopping_on=True).to(device)
    model_B.load_state_dict(torch.load('AMLS_24-25_SN24249071/B/model_B_best.pth', weights_only=True))





# Create new models
if mode == 1:
    epochs = 15

    criterion_A = nn.BCEWithLogitsLoss()
    early_stopping_A = EarlyStopping_A(patience=10)
    model_A = CNN_A(early_stopping_on=True).to(device)
    train_loss_A, val_loss_A = model_A.train_model(train_loader_A, val_loader_A, criterion_A, epochs, device, early_stopping_A, show_progress=False)
    
    
    criterion_B = nn.CrossEntropyLoss()
    early_stopping_B = EarlyStopping_B(patience=10)
    model_B = CNN_B(early_stopping_on=True).to(device)
    train_loss_B, val_loss_B = model_B.train_model(train_loader_B, val_loader_B, criterion_B, epochs, device, early_stopping_B, show_progress=False)
    

print("Model A, Breast:")
print("train accuracy: ", model_A.evaluate_accuracy(train_loader_A, device))
print("val accuracy: ", model_A.evaluate_accuracy(val_loader_A, device))
print("test accuracy: ", model_A.evaluate_accuracy(test_loader_A, device))


print("Model B, Blood:")
print("train accuracy: ", model_B.evaluate_accuracy(train_loader_B, device))
print("val accuracy: ", model_B.evaluate_accuracy(val_loader_B, device))
print("test accuracy: ", model_B.evaluate_accuracy(test_loader_B, device))