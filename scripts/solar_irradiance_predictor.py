import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import gradio as gr

class SolarIrradianceDataset(Dataset):
    def __init__(self, image_dir, irradiance_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Load irradiance values
        with open(irradiance_file, 'r') as f:
            self.irradiance_values = [float(line.strip()) for line in f]
            
        # Get list of image files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
            
        irradiance = torch.tensor(self.irradiance_values[idx], dtype=torch.float32)
        return image, irradiance

class SolarIrradianceCNN(nn.Module):
    def __init__(self):
        super(SolarIrradianceCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, irradiance in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), irradiance)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

def predict_irradiance(image, model):
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Convert to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)
    
    return float(prediction.item())

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create dataset and dataloader
    dataset = SolarIrradianceDataset(
        image_dir='path/to/images',  # Update with your image directory
        irradiance_file='path/to/irradiance.txt',  # Update with your irradiance file
        transform=transform
    )
    
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    model = SolarIrradianceCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer)
    
    # Save the trained model
    torch.save(model.state_dict(), 'baseline_cnn_model.pth')
    
    # Create Gradio interface
    def gradio_predict(image):
        return predict_irradiance(image, model)
    
    interface = gr.Interface(
        fn=gradio_predict,
        inputs=gr.Image(type="pil"),
        outputs=gr.Number(label="Predicted Solar Irradiance"),
        title="Solar Irradiance Predictor",
        description="Upload a grayscale infrared sky image to predict solar irradiance."
    )
    
    interface.launch()

if __name__ == "__main__":
    main() 