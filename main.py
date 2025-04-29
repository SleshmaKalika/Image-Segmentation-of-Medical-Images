import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom Dataset class for loading medical images and masks
class MedicalImageDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  # Convert mask to grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Example of dataset paths (replace with actual file paths)
image_paths = ["path/to/image1.png", "path/to/image2.png", ...]
mask_paths = ["path/to/mask1.png", "path/to/mask2.png", ...]

# Define transformations (resize and normalization)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Split the dataset into training and testing sets
train_images, test_images, train_masks, test_masks = train_test_split(image_paths, mask_paths, test_size=0.2)

# Create DataLoader for training and testing
train_dataset = MedicalImageDataset(train_images, train_masks, transform=transform)
test_dataset = MedicalImageDataset(test_images, test_masks, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.bottleneck(x1)
        x3 = self.decoder(x2)
        return x3


# Initialize the model, loss function, and optimizer
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for segmentation
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # Loss calculation
        loss = criterion(outputs.squeeze(1), labels.squeeze(1))
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")


# Evaluation on test dataset
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        
        # Convert outputs and labels to binary values (for segmentation)
        predicted = outputs > 0.5
        correct += (predicted == labels).sum().item()
        total += labels.numel()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


def visualize_results(inputs, labels, outputs):
    inputs = inputs.cpu().numpy()
    labels = labels.cpu().numpy()
    outputs = outputs.cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(inputs[0].transpose(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(labels[0], cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(outputs[0], cmap='gray')

    plt.show()

# Example of visualizing results
inputs, labels = next(iter(test_loader))
inputs, labels = inputs.to(device), labels.to(device)
outputs = model(inputs)

visualize_results(inputs, labels, outputs)

