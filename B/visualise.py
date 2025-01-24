import torch
import matplotlib.pyplot as plt
import sys
import os

from torchvision import transforms
from medmnist import BloodMNIST

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'B')))
from model_B import CNN_B

# Hook to store the outputs of a specific layer
activation = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = CNN().to(device)
model = CNN_B().to(device)
model.load_state_dict(torch.load('AMLS_24-25_SN24249071/B/model_B_best.pth', weights_only=True))
model.eval()

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to mean 0 and std 1
])

train_dataset = BloodMNIST(split="train", transform=transform, download=True)

image, label = train_dataset[0]

image = image.unsqueeze(0)



def plot_feature_maps(feature_maps, num_cols=8):
    num_filters = feature_maps.shape[1]
    num_rows = (num_filters + num_cols - 1) // num_cols  # Compute rows needed for grid
    
    plt.figure(figsize=(15, 15))
    for i in range(num_filters):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(feature_maps[0, i].cpu().numpy(), cmap='viridis')  # Show filter i for batch index 0
        plt.axis('off')
    #plt.show()


layers_to_visualize = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2']
for layer in layers_to_visualize:
    model._modules[layer].register_forward_hook(get_activation(layer))

# Pass an input through the model
output = model(image)

# Plot the feature maps for each layer
for layer in layers_to_visualize:
    print(f"Visualizing {layer}")
    plot_feature_maps(activation[layer])
    plt.savefig(f'AMLS_24-25_SN24249071/B/figures/Activation_layer_{layer}.png', dpi=300, bbox_inches='tight')

