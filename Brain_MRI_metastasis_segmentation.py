import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ------------------- Data Preprocessing -------------------

# CLAHE Preprocessing
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

# Custom Dataset Class
class BrainMRIDataset(Dataset):
    def _init_(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def _len_(self):
        return len(self.images)

    def _getitem_(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        image = cv2.imread(img_path, 0)  # Load as grayscale
        mask = cv2.imread(mask_path, 0)  # Load mask as grayscale
        
        # Apply CLAHE
        image = apply_clahe(image)
        
        # Normalize
        image = image / 255.0
        mask = mask / 255.0
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# ------------------- Model Implementation -------------------

# Nested U-Net Model (U-Net++)
class NestedUNet(nn.Module):
    def _init_(self, num_classes=1):
        super(NestedUNet, self)._init_()
        self.down1 = self.conv_block(1, 64)
        self.down2 = self.conv_block(64, 128)
        self.down3 = self.conv_block(128, 256)
        self.down4 = self.conv_block(256, 512)
        
        self.middle = self.conv_block(512, 1024)
        
        self.up1 = self.up_block(1024, 512)
        self.up2 = self.up_block(512, 256)
        self.up3 = self.up_block(256, 128)
        self.up4 = self.up_block(128, 64)
        
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        middle = self.middle(d4)
        
        up1 = self.up1(middle)
        up2 = self.up2(up1 + d4)
        up3 = self.up3(up2 + d3)
        up4 = self.up4(up3 + d2)
        
        return torch.sigmoid(self.final(up4 + d1))

# ------------------- Model Training and Evaluation -------------------

# Dice Loss Function
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    
    return 1 - dice.mean()

# Training Function
def train_model(model, dataloader, optimizer, criterion, num_epochs=25, device='cuda'):
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, masks in dataloader:
            inputs = inputs.unsqueeze(1).to(device)  # Add channel dimension
            masks = masks.unsqueeze(1).to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

# Evaluation Function
def evaluate_model(model, dataloader, device='cuda'):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for inputs, masks in dataloader:
            inputs = inputs.unsqueeze(1).to(device)
            masks = masks.unsqueeze(1).to(device)
            
            outputs = model(inputs)
            dice = 1 - dice_loss(outputs, masks)
            dice_scores.append(dice.item())
    
    avg_dice = np.mean(dice_scores)
    print(f"Average DICE Score: {avg_dice:.4f}")
    return avg_dice

# ------------------- Data Loading and Training Setup -------------------

# Directories for images and masks
train_images_dir = 'data/train/images'
train_masks_dir = 'data/train/masks'
test_images_dir = 'data/test/images'
test_masks_dir = 'data/test/masks'

# Data Transforms (optional)
data_transforms = transforms.Compose([
    transforms.ToTensor(),
])

# Create Datasets and Dataloaders
train_dataset = BrainMRIDataset(train_images_dir, train_masks_dir, transform=data_transforms)
test_dataset = BrainMRIDataset(test_images_dir, test_masks_dir, transform=data_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Model Initialization
model_nested_unet = NestedUNet(num_classes=1)

# Optimizer and Loss Function
optimizer = optim.Adam(model_nested_unet.parameters(), lr=1e-4)
criterion = dice_loss

# Train the Model
train_model(model_nested_unet, train_dataloader, optimizer, criterion, num_epochs=25)

# Evaluate the Model
evaluate_model(model_nested_unet, test_dataloader)