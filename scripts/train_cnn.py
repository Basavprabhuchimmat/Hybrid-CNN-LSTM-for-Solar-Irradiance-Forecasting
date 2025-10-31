import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import cv2  
from preprocess import IRImageProcessor
from cnn_model import SolarCNNRegression, SolarCNNWithFeatureExtraction
class MultiDayDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, irradiance_values):
        self.image_paths = image_paths
        self.irradiance_values = irradiance_values
        self.image_processor = IRImageProcessor(target_size=(240, 320))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
            img = self.image_processor.process_single_image(img_path)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0
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

class CNNTrainer:
    """
    Trainer class for CNN nowcasting model 
    """

    def __init__(self, model_type='standard', config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.config = {
            'learning_rate': 1e-4,
            'batch_size': 32,
            'num_epochs': 50,
            'weight_decay': 1e-4,
            'scheduler_patience': 10,
            'early_stopping_patience': 15,
            'save_dir': 'models',
            'log_dir': 'logs'
        }

        if config:
            self.config.update(config)

        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)

        if model_type == 'with_features':
            self.model = SolarCNNWithFeatureExtraction().to(self.device)
        else:
            self.model = SolarCNNRegression().to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config['scheduler_patience'],
            factor=0.5,
        )

        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        with tqdm(train_loader, desc="Training") as pbar:
            for batch_idx, (images, targets) in enumerate(pbar):
                images, targets = images.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs.squeeze(), targets)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []

        with torch.no_grad():
            with tqdm(val_loader, desc="Validation") as pbar:
                for images, batch_targets in pbar:
                    images, batch_targets = images.to(self.device), batch_targets.to(self.device)

                    outputs = self.model(images)
                    loss = self.criterion(outputs.squeeze(), batch_targets)

                    total_loss += loss.item()

                    predictions.extend(outputs.squeeze().cpu().numpy())
                    targets.extend(batch_targets.cpu().numpy())

                    pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(val_loader)

        predictions = np.array(predictions)
        targets = np.array(targets)

        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)

        return avg_loss, rmse, mae, predictions, targets

    def train(self, train_loader, val_loader=None):
        """Main training loop"""
        print(f"Starting training for {self.config['num_epochs']} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")

            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            print(f"Train Loss: {train_loss:.4f}")

            if val_loader is not None:
                val_loss, rmse, mae, predictions, targets = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)

                print(f"Val Loss: {val_loss:.4f}, RMSE: {rmse:.2f} W/m², MAE: {mae:.2f} W/m²")

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
                    }, os.path.join(self.config['save_dir'], 'best_cnn_model.pth'))

                    print(f"New best model saved! RMSE: {rmse:.2f} W/m²")
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config['early_stopping_patience']:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, os.path.join(self.config['save_dir'], 'final_cnn_model.pth'))

        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }

        with open(os.path.join(self.config['log_dir'], 'cnn_training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        print("Training completed!")
        return self.train_losses, self.val_losses

    def load_model(self, checkpoint_path):
        """Load a saved model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for images, batch_targets in tqdm(test_loader, desc="Evaluating"):
                images, batch_targets = images.to(self.device), batch_targets.to(self.device)

                outputs = self.model(images)
                predictions.extend(outputs.squeeze().cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())

        predictions = np.array(predictions)
        targets = np.array(targets)

        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)

        print(f"Test Results - RMSE: {rmse:.2f} W/m², MAE: {mae:.2f} W/m²")

        return rmse, mae, predictions, targets

def train_cnn_nowcasting():
    """Main function to train CNN nowcasting model"""

    config = {
        'learning_rate': 1e-4,
        'batch_size': 32,
        'num_epochs': 50,
        'image_dirs': [
            '/content/drive/MyDrive/data/processed/2019_01_15',
            '/content/drive/MyDrive/data/processed/2019_01_16',
            '/content/drive/MyDrive/data/processed/2019_01_17',
            '/content/drive/MyDrive/data/processed/2019_01_18',
            '/content/drive/MyDrive/data/processed/2019_01_19',
            '/content/drive/MyDrive/data/processed/2019_01_20'
        ],
        'irradiance_files': [
            '/content/drive/MyDrive/GIRASOL_DATASET/2019_01_15/pyranometer/2019_01_15.csv',
            '/content/drive/MyDrive/GIRASOL_DATASET/2019_01_16/pyranometer/2019_01_16.csv',
            '/content/drive/MyDrive/GIRASOL_DATASET/2019_01_17/pyranometer/2019_01_17.csv',
            '/content/drive/MyDrive/GIRASOL_DATASET/2019_01_18/pyranometer/2019_01_18.csv',
            '/content/drive/MyDrive/GIRASOL_DATASET/2019_01_19/pyranometer/2019_01_19.csv',
            '/content/drive/MyDrive/GIRASOL_DATASET/2019_01_20/pyranometer/2019_01_20.csv'
        ]
    }

    print("Loading multi-day dataset...")
    dataset = get_multi_day_dataset(config['image_dirs'], config['irradiance_files'])

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2 
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2 
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    trainer = CNNTrainer(model_type='standard', config=config)

    train_losses, val_losses = trainer.train(train_loader, val_loader)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('CNN Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/cnn_training_curves.png')
    plt.show()

    print("CNN nowcasting training completed!")

if __name__ == '__main__':
    train_cnn_nowcasting()