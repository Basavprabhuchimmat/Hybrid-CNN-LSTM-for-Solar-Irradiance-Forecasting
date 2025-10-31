import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from cnn_model import SolarCNNWithFeatureExtraction
from lstm_model import HybridCNNLSTM
from solar_datasets import SolarSequenceDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

class HybridTrainer:
    """
    Trainer class for the hybrid CNN-LSTM model for solar irradiance forecasting.
    """

    def __init__(self, pretrained_cnn_path=None, config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.config = {
            'sequence_length': 20,
            'forecast_horizon': 4,
            'learning_rate': 1e-4,
            'batch_size': 16,  
            'num_epochs': 30,
            'lstm_hidden_size': 128,
            'feature_dim': 512,
            'weight_decay': 1e-4,
            'scheduler_patience': 8,
            'early_stopping_patience': 12,
            'freeze_cnn': True,  
            'save_dir': 'models',
            'log_dir': 'logs'
        }

        if config:
            self.config.update(config)

        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)

        cnn_model = None
        if pretrained_cnn_path:
            print(f"Loading pretrained CNN from {pretrained_cnn_path}")
            cnn_model = SolarCNNWithFeatureExtraction()
            checkpoint = torch.load(pretrained_cnn_path, map_location=self.device)
            cnn_model.load_state_dict(checkpoint['model_state_dict'])

        self.model = HybridCNNLSTM(
            cnn_model=cnn_model,
            feature_dim=self.config['feature_dim'],
            sequence_length=self.config['sequence_length'],
            lstm_hidden_size=self.config['lstm_hidden_size'],
            forecast_horizon=self.config['forecast_horizon']
        ).to(self.device)

        self.model.set_cnn_trainable(not self.config['freeze_cnn'])

        if self.config['freeze_cnn']:
            lstm_params = list(self.model.lstm.parameters())
            print(f"Training LSTM only ({sum(p.numel() for p in lstm_params):,} parameters)")
        else:
            lstm_params = self.model.parameters()
            print(f"Training full hybrid model ({sum(p.numel() for p in self.model.parameters()):,} parameters)")

        self.optimizer = optim.Adam(
            lstm_params, 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=self.config['scheduler_patience'],
            factor=0.5,
            verbose=True
        )

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_nowcast_loss = 0.0
        total_forecast_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        with tqdm(train_loader, desc="Training Hybrid") as pbar:
            for batch_idx, (image_sequences, historical_irradiance, future_irradiance) in enumerate(pbar):
                image_sequences = image_sequences.to(self.device)
                historical_irradiance = historical_irradiance.to(self.device)
                future_irradiance = future_irradiance.to(self.device)

                self.optimizer.zero_grad()

                nowcasts, forecasts = self.model(image_sequences)

                nowcast_loss = self.mse_loss(nowcasts.squeeze(-1), historical_irradiance.squeeze(-1))
                forecast_loss = self.mse_loss(forecasts, future_irradiance)

                total_batch_loss = 0.3 * nowcast_loss + 0.7 * forecast_loss

                total_batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_nowcast_loss += nowcast_loss.item()
                total_forecast_loss += forecast_loss.item()
                total_loss += total_batch_loss.item()
                num_batches += 1

                pbar.set_postfix({
                    'Nowcast': f'{nowcast_loss.item():.4f}',
                    'Forecast': f'{forecast_loss.item():.4f}',
                    'Total': f'{total_batch_loss.item():.4f}'
                })

        return (total_nowcast_loss / num_batches, 
                total_forecast_loss / num_batches, 
                total_loss / num_batches)

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_nowcast_loss = 0.0
        total_forecast_loss = 0.0
        all_nowcast_pred = []
        all_nowcast_target = []
        all_forecast_pred = []
        all_forecast_target = []

        with torch.no_grad():
            with tqdm(val_loader, desc="Validation") as pbar:
                for image_sequences, historical_irradiance, future_irradiance in pbar:
                    image_sequences = image_sequences.to(self.device)
                    historical_irradiance = historical_irradiance.to(self.device)
                    future_irradiance = future_irradiance.to(self.device)

                    nowcasts, forecasts = self.model(image_sequences)

                    nowcast_loss = self.mse_loss(nowcasts.squeeze(-1), historical_irradiance.squeeze(-1))
                    forecast_loss = self.mse_loss(forecasts, future_irradiance)

                    total_nowcast_loss += nowcast_loss.item()
                    total_forecast_loss += forecast_loss.item()

                    all_nowcast_pred.append(nowcasts.squeeze(-1).cpu().numpy())
                    all_nowcast_target.append(historical_irradiance.squeeze(-1).cpu().numpy())
                    all_forecast_pred.append(forecasts.cpu().numpy())
                    all_forecast_target.append(future_irradiance.cpu().numpy())

                    pbar.set_postfix({
                        'Nowcast': f'{nowcast_loss.item():.4f}',
                        'Forecast': f'{forecast_loss.item():.4f}'
                    })

        avg_nowcast_loss = total_nowcast_loss / len(val_loader)
        avg_forecast_loss = total_forecast_loss / len(val_loader)

        nowcast_pred = np.concatenate(all_nowcast_pred, axis=0)
        nowcast_target = np.concatenate(all_nowcast_target, axis=0)
        forecast_pred = np.concatenate(all_forecast_pred, axis=0)
        forecast_target = np.concatenate(all_forecast_target, axis=0)

        nowcast_rmse = np.sqrt(mean_squared_error(
            nowcast_target.flatten(), nowcast_pred.flatten()
        ))

        forecast_rmse = np.sqrt(mean_squared_error(
            forecast_target.flatten(), forecast_pred.flatten()
        ))

        forecast_rmse_per_step = []
        for step in range(self.config['forecast_horizon']):
            step_rmse = np.sqrt(mean_squared_error(
                forecast_target[:, step], forecast_pred[:, step]
            ))
            forecast_rmse_per_step.append(step_rmse)

        return (avg_nowcast_loss, avg_forecast_loss, 
                nowcast_rmse, forecast_rmse, forecast_rmse_per_step)

    def train(self, train_loader, val_loader=None):
        """Main training loop"""
        print(f"Starting hybrid CNN-LSTM training for {self.config['num_epochs']} epochs...")
        print(f"CNN frozen: {self.config['freeze_cnn']}")

        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")

            train_nowcast_loss, train_forecast_loss, train_total_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_total_loss)

            print(f"Train - Nowcast: {train_nowcast_loss:.4f}, "
                  f"Forecast: {train_forecast_loss:.4f}, Total: {train_total_loss:.4f}")

            if val_loader is not None:
                (val_nowcast_loss, val_forecast_loss, 
                 nowcast_rmse, forecast_rmse, forecast_rmse_per_step) = self.validate_epoch(val_loader)

                val_total_loss = 0.3 * val_nowcast_loss + 0.7 * val_forecast_loss
                self.val_losses.append(val_total_loss)

                print(f"Val - Nowcast RMSE: {nowcast_rmse:.2f} W/m², "
                      f"Forecast RMSE: {forecast_rmse:.2f} W/m²")
                print(f"Forecast RMSE per step: {[f'{r:.2f}' for r in forecast_rmse_per_step]}")

                self.scheduler.step(val_total_loss)

                if val_total_loss < self.best_val_loss:
                    self.best_val_loss = val_total_loss
                    self.patience_counter = 0

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_total_loss,
                        'nowcast_rmse': nowcast_rmse,
                        'forecast_rmse': forecast_rmse,
                        'config': self.config
                    }, os.path.join(self.config['save_dir'], 'best_hybrid_model.pth'))

                    print(f"New best model saved! Forecast RMSE: {forecast_rmse:.2f} W/m²")
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
        }, os.path.join(self.config['save_dir'], 'final_hybrid_model.pth'))

        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }

        with open(os.path.join(self.config['log_dir'], 'hybrid_training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        print("Hybrid CNN-LSTM training completed!")
        return self.train_losses, self.val_losses


def train_hybrid_model():
    """Main function to train hybrid CNN-LSTM model"""

    config = {
        'sequence_length': 20,
        'forecast_horizon': 4,
        'learning_rate': 1e-4,
        'batch_size': 16,
        'num_epochs': 30,
        'freeze_cnn': True,
        'image_dir': 'data/processed/2017_12_18',
        'irradiance_file': 'GIRASOL_DATASET/2017_12_18/pyranometer/2017_12_18.csv',
        'pretrained_cnn_path': 'models/best_cnn_model.pth'
    }

    print("Loading sequence dataset...")
    dataset = SolarSequenceDataset(
        image_dir=config['image_dir'],
        irradiance_file=config['irradiance_file'],
        sequence_length=config['sequence_length'],
        forecast_horizon=config['forecast_horizon']
    )

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

    trainer = HybridTrainer(
        pretrained_cnn_path=config.get('pretrained_cnn_path'),
        config=config
    )

    train_losses, val_losses = trainer.train(train_loader, val_loader)

    print("Hybrid CNN-LSTM training completed!")


if __name__ == '__main__':
    train_hybrid_model()
