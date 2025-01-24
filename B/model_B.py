from medmnist import BloodMNIST
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class CNN_B(nn.Module):
    def __init__(self, input_channels=3, filters1=32, filters2=64, filters3=32, filters4=32, kernel_size=3, image_size=28, fc1_out=128, fc2_out=128, fc3_out=8, lr=0.001, early_stopping_on=False):
        """
        Args:
            input_channels (int): Number of channels, 1 if using grayscale, 3 if using rgb e.t.c.
            filters1 (int): Number of filters for the first conv layer.
            filters2 (int): Number of filters for the second conv layer.
            kernel_size (int): Size of the kernel used (nxn).
            image_size (int): Height and width of image size, assuming square (nxn) images.
            fc1_out (int): Number of output nodes for first layer.
            fc2_out (int): Number of output nodes for second layer.
            lr (int): Learning rate, used for optimiser.
        """
        super(CNN_B, self).__init__()
        # Choose padding 
        padding = kernel_size//2
        # Define Convolution layers
        self.conv1_1 = nn.Conv2d(input_channels, filters1, kernel_size, padding=padding)
        self.conv1_2 = nn.Conv2d(filters1, filters1, kernel_size, padding=padding)
        self.conv2_1 = nn.Conv2d(filters1, filters2, kernel_size, padding=padding)
        self.conv2_2 = nn.Conv2d(filters2, filters2, kernel_size, padding=padding)
        self.conv3_1 = nn.Conv2d(filters2, filters3, kernel_size, padding=padding)
        self.conv3_2 = nn.Conv2d(filters3, filters3, kernel_size, padding=padding)
        self.conv4_1 = nn.Conv2d(filters3, filters4, kernel_size, padding=padding)
        self.conv4_2 = nn.Conv2d(filters4, filters4, kernel_size, padding=padding)

        # Dummy run through to calculate the flattened size
        dummy = torch.zeros(1, input_channels, image_size, image_size)
        with torch.no_grad():
            # Pass through convolution + pool layers
            out = self.forward_features(dummy) 
        flat_size = out.view(1, -1).size(1)

        # Fully connected layers
        self.fc1 = nn.Linear(flat_size, fc1_out) 
        self.fc2 = nn.Linear(fc1_out, fc2_out)  
        self.fc3 = nn.Linear(fc2_out, fc3_out)

        # Define optimiser as adam optimiser with weight decay
        self.optimiser = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)

        # Dropout for regularisation
        self.dropout = nn.Dropout(0.5)
        self.e_stop_on = early_stopping_on

    def forward_features(self, x):
        # Convolutional and pooling layers (called in forward())
        x = torch.relu(self.conv1_1(x))
        x = torch.relu(self.conv1_2(x))
        x = torch.max_pool2d(x, 2)
        
        x = torch.relu(self.conv2_1(x))
        x = torch.relu(self.conv2_2(x))
        x = torch.max_pool2d(x, 2)

        # x = torch.relu(self.conv3_1(x))
        # x = torch.relu(self.conv3_2(x))
        # x = torch.max_pool2d(x, 2)

        # x = torch.relu(self.conv4_1(x))
        # x = torch.relu(self.conv4_2(x))
        # x = torch.max_pool2d(x, 2)
        return x
    
    def forward(self, x):
        # Whole forward step.
        # Conv layers
        x = self.forward_features(x)

        # Reshape to correct size
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout to improve generalisation.
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def train_model(self, data_loader, val_loader, criterion, epochs, device, early_stopping=False, show_progress=False):
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device, dtype=torch.long).squeeze()
                
                # Zero gradients
                self.optimiser.zero_grad()
                
                # Forward pass
                outputs = self(images)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimiser.step()
                
                running_loss += loss.item()
            
            v_loss = self.evaluate_loss(val_loader, criterion, device)
            val_losses.append(v_loss)
            train_losses.append(running_loss/len(data_loader))
            if show_progress==True:
                print(f"Epoch {epoch+1}, Training loss: {running_loss/len(data_loader)}, Validation loss: {v_loss}")
                print(self.evaluate_accuracy(val_loader, device))
            if self.e_stop_on:
                early_stopping(v_loss, self)

                if early_stopping.early_stop:
                    print("Early stopping triggered!")
                    break
        # Load early model if stopped early.
        if self.e_stop_on:
            self.load_state_dict(torch.load('AMLS_24-25_SN24249071/B/checkpoint.pth', weights_only=True))

        return train_losses, val_losses

    def evaluate_loss(self, data_loader, criterion, device):
        self.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device, dtype=torch.long).squeeze()
                #labels = labels.view(-1, 1)  # Reshape to (batch_size, 1)

                # Forward pass
                outputs = self(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        return running_loss / len(data_loader)
    
    def evaluate_accuracy(self, data_loader, device):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device, dtype=torch.long).squeeze()
                
                # Get model predictions
                outputs = self(images)
                
                # Count correct predictions
                _, predicted = torch.max(outputs, 1)  # softmax
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        return accuracy
        
class EarlyStopping_B:
    def __init__(self, patience=5, delta=0, path='AMLS_24-25_SN24249071/B/checkpoint.pth'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            delta (float): Minimum change in to qualify as an improvement.
            path (str): Filepath to save model.
    
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss  # lower value better

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
        else:
        
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decreases.
        """
        print(f"Validation loss decreased ({self.val_loss_min:.3f} to {val_loss:.3f})")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

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

    # Hyperparameters
    learn_rate = 0.001
    epochs = 30


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN_B(early_stopping_on=True).to(device)

    early_stopping = EarlyStopping_B(patience=10)

    criterion = nn.CrossEntropyLoss()

    loss, val_loss = model.train_model(train_loader, validate_loader, criterion, epochs, device, early_stopping)


    losses = [[loss, "train"], [val_loss, "validate"]]
    plot_data(losses, "Loss v Validation loss", "epochs", "loss")
    plt.tight_layout() 

    plt.savefig('AMLS_24-25_SN24249071/B/figures/current_model.png', dpi=300, bbox_inches='tight')
    
    torch.save(model.state_dict(), 'AMLS_24-25_SN24249071/B/model_B.pth')

    v = model.evaluate_accuracy(validate_loader, device)
    print(f"Validation Accuracy: {v:.2f}%")