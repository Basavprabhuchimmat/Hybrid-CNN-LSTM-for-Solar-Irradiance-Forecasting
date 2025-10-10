import os
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from torchvision import models, transforms



class MultiDayDataset(Dataset):
    def __init__(self, image_paths, irradiance_values, target_size=(224,224)):
        self.image_paths = image_paths
        self.irradiance_values = irradiance_values
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img)
        irradiance = torch.tensor(self.irradiance_values[idx], dtype=torch.float32)
        return img_tensor, irradiance

def get_multi_day_dataset(image_dirs, irradiance_files):
    image_paths = []
    irradiance_values = []
    for img_dir, irr_file in zip(image_dirs, irradiance_files):
        if irr_file.endswith('.csv'):
            df = pd.read_csv(irr_file)
            values = df.iloc[:, 1].values.astype(np.float32)
        else:
            values = np.loadtxt(irr_file, delimiter=',')[:, 1].astype(np.float32)

        files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        files = files[:len(values)]
        image_paths.extend([os.path.join(img_dir, f) for f in files])
        irradiance_values.extend(values[:len(files)])
    return MultiDayDataset(image_paths, irradiance_values)


class EfficientNetRegression(nn.Module):
    def __init__(self, num_classes=1, model_name='efficientnet_b0', pretrained=True):
        super(EfficientNetRegression, self).__init__()
        # Use weights argument instead of pretrained
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None
        backbone = getattr(models, model_name)(weights=weights)
        self.features = backbone.features
        self.pooling = nn.AdaptiveAvgPool2d(1)
        in_features = backbone.classifier[1].in_features
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = self.regressor(x)
        return x
    


class CNNTrainer:
    def __init__(self, config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.config = {
            'learning_rate': 1e-4,
            'batch_size': 64,
            'num_epochs': 20,
            'weight_decay': 1e-4,
            'scheduler_patience': 5,
            'early_stopping_patience': 10,
            'save_dir': 'models',
            'log_dir': 'logs'
        }
        if config:
            self.config.update(config)
        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        self.model = EfficientNetRegression().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.config['learning_rate'],
                                    weight_decay=self.config['weight_decay'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                             mode='min',
                                                             patience=self.config['scheduler_patience'],
                                                             factor=0.5)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        with tqdm(train_loader, desc="Training") as pbar:
            for images, targets in pbar:
                images, targets = images.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs.squeeze(), targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        return total_loss / len(train_loader)

    def validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        predictions, targets_all = [], []
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation") as pbar:
                for images, targets in pbar:
                    images, targets = images.to(self.device), targets.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs.squeeze(), targets)
                    total_loss += loss.item()
                    predictions.extend(outputs.squeeze().cpu().numpy())
                    targets_all.extend(targets.cpu().numpy())
                    pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
        avg_loss = total_loss / len(val_loader)
        predictions = np.array(predictions)
        targets_all = np.array(targets_all)
        rmse = np.sqrt(mean_squared_error(targets_all, predictions))
        mae = mean_absolute_error(targets_all, predictions)
        return avg_loss, rmse, mae

    def train(self, train_loader, val_loader):
        print(f"Starting training for {self.config['num_epochs']} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            val_loss, rmse, mae = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            print(f"Val Loss: {val_loss:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
            self.scheduler.step(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, os.path.join(self.config['save_dir'], 'best_efficientnet_model.pth'))
                print(f"New best model saved!")
            else:
                self.patience_counter += 1
            if self.patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        plt.figure(figsize=(10,6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('EfficientNet Training Progress')
        plt.legend()
        plt.grid(True)
        plt.show()
        

# def train_efficientnet_nowcasting():
#     config = {
#         'learning_rate': 1e-4,
#         'batch_size': 64,
#         'num_epochs': 25,
#         'image_dirs': [
#             '/content/drive/MyDrive/data/processed/2019_01_15',
#             '/content/drive/MyDrive/data/processed/2019_01_16',
#             '/content/drive/MyDrive/data/processed/2019_01_17',
#             '/content/drive/MyDrive/data/processed/2019_01_18',
#             '/content/drive/MyDrive/data/processed/2019_01_19',
#             '/content/drive/MyDrive/data/processed/2019_01_20'

#         ],
#         'irradiance_files': [
#             '/content/drive/MyDrive/GIRASOL_DATASET/2019_01_15/pyranometer/2019_01_15.csv',
#             '/content/drive/MyDrive/GIRASOL_DATASET/2019_01_16/pyranometer/2019_01_16.csv',
#             '/content/drive/MyDrive/GIRASOL_DATASET/2019_01_17/pyranometer/2019_01_17.csv',
#             '/content/drive/MyDrive/GIRASOL_DATASET/2019_01_18/pyranometer/2019_01_18.csv',
#             '/content/drive/MyDrive/GIRASOL_DATASET/2019_01_19/pyranometer/2019_01_19.csv',
#             '/content/drive/MyDrive/GIRASOL_DATASET/2019_01_20/pyranometer/2019_01_20.csv',

#         ]
#     }
#     print("Loading multi-day dataset...")
#     dataset = get_multi_day_dataset(config['image_dirs'], config['irradiance_files'])
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
#     train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
#     val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
#     print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
#     trainer = CNNTrainer(config=config)
#     trainer.train(train_loader, val_loader)
#     print("EfficientNet nowcasting training completed!")

# # ============================
# # 6. Run Training
# # ============================
# train_efficientnet_nowcasting()