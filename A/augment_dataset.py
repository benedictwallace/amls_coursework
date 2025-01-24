import torch
from torchvision import transforms
from medmnist import BreastMNIST
from pathlib import Path
import shutil
import os
import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'A')))

def create_augmented_dataset():
    # Define the augmentation pipeline
    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # Rotate within ±15 degrees
        transforms.RandomCrop(28, padding=4)
    ])

    # Directory to save augmented data
    output_dir = Path("AMLS_24-25_SN24249071/A/augmented_dataset/breast")
    if output_dir.exists():
        #shutil.rmtree(output_dir)
        print("Folder exists, if changes wanted, delete and re run script.")
        pass
    else:

        output_dir.mkdir(parents=True, exist_ok=True)

        # Load the original training dataset
        train_dataset = BreastMNIST(split="train", download=True)

        # Number of augmentations per image
        num_augmentations = 10

        label_count = 0
        for i, (image, label) in enumerate(train_dataset):
            label_count += label
            # Create a subfolder for each class
            class_dir = output_dir / f"class_{label}"
            class_dir.mkdir(parents=True, exist_ok=True)
            # Save the original image
            image.save(class_dir / f"image_{i}_original.png")
            
            # Create and save augmented images
            for j in range(num_augmentations):
                augmented_image = augmentation_transform(image)
                augmented_image.save(class_dir / f"image_{i}_augmented_{j}.png")


create_augmented_dataset()

