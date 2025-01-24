from medmnist import BloodMNIST
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
import os
from torchvision.datasets import ImageFolder


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'B')))
from model_B import CNN_B

def plot_data(loss_records, title, x_label, y_label):
    # create the figure
    fig, ax = plt.subplots(1, figsize=(17, 7))

    # add the data to the plot
    for df in loss_records:
        x = range(len(df[0]))
        y = df[0]

        ax.plot(x, y, label=df[1])

    # remove whitespace before and after
    ax.margins(x=0)

    # format the axes
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    return fig, ax


if __name__=="__main__":
    transform = transforms.Compose([
        #transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to mean 0 and std 1
    ])

    train_dataset = BloodMNIST(split="train", transform=transform, download=True)
    validate_dataset = BloodMNIST(split="val", transform=transform, download=True)
    test_dataset = BloodMNIST(split="test", transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()

    # CHOOSE WHICH TEST TO DO
    test = "learn_rate"

    if test == "learn_rate":
        # Test learning rate
        epochs = 20
        train_loss_records = []
        val_loss_records = []
        learning_rates = [0.1, 0.01, 0.001, 0.0001]
        for lr in learning_rates:
            # Three models to average the losses.
            model1 = CNN_B(lr=lr).to(device)
            train_loss1, val_loss1 = model1.train_model(train_loader, validate_loader, criterion, epochs, device)

            model2 = CNN_B(lr=lr).to(device)
            train_loss2, val_loss2 = model2.train_model(train_loader, validate_loader, criterion, epochs, device)

            model3 = CNN_B(lr=lr).to(device)
            train_loss3, val_loss3 = model3.train_model(train_loader, validate_loader, criterion, epochs, device)

            averaged_train_loss = [(train_loss1[i] + train_loss2[i] + train_loss3[i]) / 3 for i in range(len(train_loss1))]
            averaged_val_loss = [(val_loss1[i] + val_loss2[i] + val_loss3[i]) / 3 for i in range(len(val_loss1))]


            train_loss_records.append((averaged_train_loss, f'LR={lr}'))
            val_loss_records.append((averaged_val_loss, f'LR={lr}'))
        
        plot_data(train_loss_records, "Loss change", "Epochs", "loss")
        plt.legend()
        #plt.ylim(0,1)
        plt.tight_layout() 
        plt.savefig('AMLS_24-25_SN24249071/B/figures/learning_rates_train.png', dpi=300, bbox_inches='tight')

        plot_data(val_loss_records, "Loss change", "Epochs", "loss")
        plt.legend()
        #plt.ylim(0,1)
        plt.tight_layout() 
        plt.savefig('AMLS_24-25_SN24249071/B/figures/learning_rates_val.png', dpi=300, bbox_inches='tight')

    if test=="kernel":
        kernel_size = [2,3,6]
        train_loss_records = []
        val_loss_records = []
        for k in kernel_size:
            epochs = 25

            # Three models to average the losses.
            model1 = CNN_B(kernel_size=k).to(device)
            train_loss1, val_loss1 = model1.train_model(train_loader, validate_loader, criterion, epochs, device)

            model2 = CNN_B(kernel_size=k).to(device)
            train_loss2, val_loss2 = model2.train_model(train_loader, validate_loader, criterion, epochs, device)

            model3 = CNN_B(kernel_size=k).to(device)
            train_loss3, val_loss3 = model3.train_model(train_loader, validate_loader, criterion, epochs, device)

            averaged_train_loss = [(train_loss1[i] + train_loss2[i] + train_loss3[i]) / 3 for i in range(len(train_loss1))]
            averaged_val_loss = [(val_loss1[i] + val_loss2[i] + val_loss3[i]) / 3 for i in range(len(val_loss1))]


            train_loss_records.append([averaged_train_loss, f'kernel={k}'])
            val_loss_records.append((averaged_val_loss, f'kernel={k}'))
        #print(train_loss_records)
        plot_data(train_loss_records, "Loss change", "Epochs", "loss")
        plt.legend()
        #plt.ylim(0,1)
        plt.tight_layout() 
        plt.savefig('AMLS_24-25_SN24249071/B/figures/kernel_size_train.png', dpi=300, bbox_inches='tight')

        plot_data(val_loss_records, "Loss change", "Epochs", "loss")
        plt.legend()
        #plt.ylim(0,1)
        plt.tight_layout() 
        plt.savefig('AMLS_24-25_SN24249071/B/figures/kernel_size_val.png', dpi=300, bbox_inches='tight')

    if test=="fc_nodes":
        # Test learning rate
        epochs = 25
        train_loss_records = []
        val_loss_records = []
        fc_nodes = [64, 128, 256, 4096]
        for n in fc_nodes:
            # Three models to average the losses.
            model1 = CNN_B(fc1_out=n, fc2_out=n).to(device)
            train_loss1, val_loss1 = model1.train_model(train_loader, validate_loader, criterion, epochs, device)

            model2 = CNN_B(fc1_out=n, fc2_out=n).to(device)
            train_loss2, val_loss2 = model2.train_model(train_loader, validate_loader, criterion, epochs, device)

            model3 = CNN_B(fc1_out=n, fc2_out=n).to(device)
            train_loss3, val_loss3 = model3.train_model(train_loader, validate_loader, criterion, epochs, device)

            averaged_train_loss = [(train_loss1[i] + train_loss2[i] + train_loss3[i]) / 3 for i in range(len(train_loss1))]
            averaged_val_loss = [(val_loss1[i] + val_loss2[i] + val_loss3[i]) / 3 for i in range(len(val_loss1))]


            train_loss_records.append((averaged_train_loss, f'FC nodes={n}'))
            val_loss_records.append((averaged_val_loss, f'FC nodes={n}'))
        
        plot_data(train_loss_records, "Loss change", "Epochs", "loss")
        plt.legend()
        #plt.ylim(0,1)
        plt.tight_layout() 
        plt.savefig('AMLS_24-25_SN24249071/B/figures/fc_nodes_train.png', dpi=300, bbox_inches='tight')

        plot_data(val_loss_records, "Loss change", "Epochs", "loss")
        plt.legend()
        #plt.ylim(0,1)
        plt.tight_layout() 
        plt.savefig('AMLS_24-25_SN24249071/B/figures/fc_nodes_val.png', dpi=300, bbox_inches='tight')

    if test=="batch":
        # Test learning rate
        epochs = 25
        train_loss_records = []
        val_loss_records = []
        batch_size = [32, 64, 128, 256]
        for size in batch_size:

            train_dataset = BloodMNIST(split="train", transform=transform, download=True)
            validate_dataset = BloodMNIST(split="val", transform=transform, download=True)
            test_dataset = BloodMNIST(split="test", transform=transform, download=True)

            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=size, shuffle=True)
            validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=size, shuffle=False)

            # Three models to average the losses.
            model1 = CNN_B().to(device)
            train_loss1, val_loss1 = model1.train_model(train_loader, validate_loader, criterion, epochs, device)

            model2 = CNN_B().to(device)
            train_loss2, val_loss2 = model2.train_model(train_loader, validate_loader, criterion, epochs, device)

            model3 = CNN_B().to(device)
            train_loss3, val_loss3 = model3.train_model(train_loader, validate_loader, criterion, epochs, device)

            averaged_train_loss = [(train_loss1[i] + train_loss2[i] + train_loss3[i]) / 3 for i in range(len(train_loss1))]
            averaged_val_loss = [(val_loss1[i] + val_loss2[i] + val_loss3[i]) / 3 for i in range(len(val_loss1))]


            train_loss_records.append((averaged_train_loss, f'batch={size}'))
            val_loss_records.append((averaged_val_loss, f'batch={size}'))
        
        plot_data(train_loss_records, "Loss change", "Epochs", "loss")
        plt.legend()
        #plt.ylim(0,1)
        plt.tight_layout() 
        plt.savefig('AMLS_24-25_SN24249071/B/figures/batch_size_train.png', dpi=300, bbox_inches='tight')

        plot_data(val_loss_records, "Loss change", "Epochs", "loss")
        plt.legend()
        #plt.ylim(0,1)
        plt.tight_layout() 
        plt.savefig('AMLS_24-25_SN24249071/B/figures/batch_size_val.png', dpi=300, bbox_inches='tight')