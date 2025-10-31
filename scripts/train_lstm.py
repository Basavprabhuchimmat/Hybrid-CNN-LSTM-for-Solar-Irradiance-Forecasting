import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from lstm_model import SolarLSTMForecasting
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

def load_multi_day_irradiance(files):
    """Load and concatenate irradiance data from multiple CSV files"""
    all_values = []
    for f in files:
        df = pd.read_csv(f)
        values = df.iloc[:, 1].values.astype(np.float32)
        all_values.extend(values)
    return np.array(all_values)

class MultiDayTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, irradiance_files, sequence_length, forecast_horizon):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.irradiance = load_multi_day_irradiance(irradiance_files)
        self.length = len(self.irradiance) - sequence_length - forecast_horizon + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        seq = self.irradiance[idx:idx+self.sequence_length]
        target = self.irradiance[idx+self.sequence_length:idx+self.sequence_length+self.forecast_horizon]
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(-1)
        target = torch.tensor(target, dtype=torch.float32)
        return seq, target

class LSTMTrainer:
    """
    Trainer class for LSTM forecasting model 
    """

    def __init__(self, config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.config = {
            'sequence_length': 20,
            'forecast_horizon': 4,
            'learning_rate': 1e-3,
            'batch_size': 32,
            'num_epochs': 5,
            'lstm_hidden_size': 128,
            'lstm_num_layers': 2,
            'dropout': 0.2,
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

        self.model = SolarLSTMForecasting(
            input_size=1,
            hidden_size=self.config['lstm_hidden_size'],
            num_layers=self.config['lstm_num_layers'],
            output_size=self.config['forecast_horizon'],
            dropout=self.config['dropout']
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=self.config['scheduler_patience'],
            factor=0.5
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

        with tqdm(train_loader, desc="Training LSTM") as pbar:
            for batch_idx, (sequences, targets) in enumerate(pbar):
                sequences, targets = sequences.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            with tqdm(val_loader, desc="Validation") as pbar:
                for sequences, targets in pbar:
                    sequences, targets = sequences.to(self.device), targets.to(self.device)

                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, targets)

                    total_loss += loss.item()

                    all_predictions.append(outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

                    pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(val_loader)

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        rmse_per_step = []
        mae_per_step = []

        for step in range(self.config['forecast_horizon']):
            step_pred = predictions[:, step]
            step_target = targets[:, step]
            mse = mean_squared_error(step_target, step_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(step_target, step_pred)
            rmse_per_step.append(rmse)
            mae_per_step.append(mae)

        overall_rmse = np.sqrt(mean_squared_error(targets.flatten(), predictions.flatten()))
        overall_mae = mean_absolute_error(targets.flatten(), predictions.flatten())

        return avg_loss, overall_rmse, overall_mae, rmse_per_step, mae_per_step

    def train(self, train_loader, val_loader=None):
        """Main training loop"""
        print(f"Starting LSTM training for {self.config['num_epochs']} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Sequence length: {self.config['sequence_length']}, Forecast horizon: {self.config['forecast_horizon']}")

        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")

            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")

            if val_loader is not None:
                val_loss, overall_rmse, overall_mae, rmse_per_step, mae_per_step = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)

                print(f"Val Loss: {val_loss:.4f}, Overall RMSE: {overall_rmse:.2f} W/m²")
                print(f"RMSE per step: {[f'{r:.2f}' for r in rmse_per_step]}")

                self.scheduler.step(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'rmse': overall_rmse,
                        'config': self.config
                    }, os.path.join(self.config['save_dir'], 'best_lstm_model.pth'))

                    print(f"New best model saved! RMSE: {overall_rmse:.2f} W/m²")
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
        }, os.path.join(self.config['save_dir'], 'final_lstm_model.pth'))

        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }

        with open(os.path.join(self.config['log_dir'], 'lstm_training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        print("LSTM training completed!")
        return self.train_losses, self.val_losses

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

    def evaluate(self, test_loader):
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for sequences, targets in tqdm(test_loader, desc="Evaluating LSTM"):
                sequences, targets = sequences.to(self.device), targets.to(self.device)
                outputs = self.model(sequences)
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        overall_rmse = np.sqrt(mean_squared_error(targets.flatten(), predictions.flatten()))
        overall_mae = mean_absolute_error(targets.flatten(), predictions.flatten())

        rmse_per_step = []
        for step in range(self.config['forecast_horizon']):
            step_rmse = np.sqrt(mean_squared_error(targets[:, step], predictions[:, step]))
            rmse_per_step.append(step_rmse)

        print(f"Test Results - Overall RMSE: {overall_rmse:.2f} W/m²")
        print(f"RMSE per forecast step: {[f'{r:.2f}' for r in rmse_per_step]}")

        return overall_rmse, overall_mae, rmse_per_step, predictions, targets

def train_lstm_forecasting():
    """Main function to train LSTM forecasting model"""

    config = {
        'sequence_length': 20,
        'forecast_horizon': 4,
        'learning_rate': 1e-3,
        'batch_size': 32,
        'num_epochs': 5,
        'irradiance_files': [
            'GIRASOL_DATASET/2019_01_18/pyranometer/2019_01_18.csv',
            'GIRASOL_DATASET/2019_01_19/pyranometer/2019_01_19.csv'
        ]
    }
    print("Loading multi-day time series dataset...")
    dataset = MultiDayTimeSeriesDataset(
        irradiance_files=config['irradiance_files'],
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
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    trainer = LSTMTrainer(config=config)
    train_losses, val_losses = trainer.train(train_loader, val_loader)

    print("LSTM forecasting training completed!")

if __name__ == '__main__':
    train_lstm_forecasting()