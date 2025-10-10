import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd

class SolarIrradianceDataset(Dataset):
    """
    Dataset class for loading IR sky images and corresponding solar irradiance values
    """

    def __init__(self, image_dir, irradiance_file, transform=None, target_size=(240, 320)):
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size

        # Load irradiance data
        if irradiance_file.endswith('.csv'):
            df = pd.read_csv(irradiance_file)
            # Assuming the irradiance values are in the second column
            self.irradiance_values = df.iloc[:, 1].values.astype(np.float32)
        else:
            self.irradiance_values = np.loadtxt(irradiance_file, delimiter=',')[:, 1].astype(np.float32)

        # Get image files
        self.image_files = sorted([f for f in os.listdir(image_dir)
                                  if f.endswith(('.png', '.jpg', '.jpeg'))])

        # Ensure we have matching number of images and irradiance values
        min_length = min(len(self.image_files), len(self.irradiance_values))
        self.image_files = self.image_files[:min_length]
        self.irradiance_values = self.irradiance_values[:min_length]

        # Initialize image processor
        self.image_processor = IRImageProcessor(target_size=target_size)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load and process image
        img_path = os.path.join(self.image_dir, self.image_files[idx])

        # Check if image is already processed (RGB) or raw IR
        img = cv2.imread(img_path)
        if img is None:
            # Try loading as 16-bit IR image
            img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
            if img is not None:
                # Process raw IR image
                img = self.image_processor.process_single_image(img_path)
            else:
                raise ValueError(f"Could not load image: {img_path}")
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to tensor
        img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0

        # Apply transforms if provided
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Get corresponding irradiance value
        irradiance = torch.tensor(self.irradiance_values[idx], dtype=torch.float32)

        return img_tensor, irradiance


class SolarSequenceDataset(Dataset):
    """
    Dataset class for sequence-based training (for LSTM and hybrid model)
    """

    def __init__(self, image_dir, irradiance_file, sequence_length=20,
                 forecast_horizon=4, transform=None, target_size=(240, 320)):
        self.image_dir = image_dir
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.transform = transform
        self.target_size = target_size

        # Load irradiance data
        if irradiance_file.endswith('.csv'):
            df = pd.read_csv(irradiance_file)
            self.irradiance_values = df.iloc[:, 1].values.astype(np.float32)
        else:
            self.irradiance_values = np.loadtxt(irradiance_file, delimiter=',')[:, 1].astype(np.float32)

        # Get image files
        self.image_files = sorted([f for f in os.listdir(image_dir)
                                  if f.endswith(('.png', '.jpg', '.jpeg'))])

        # Ensure we have enough data for sequences
        min_length = min(len(self.image_files), len(self.irradiance_values))
        self.image_files = self.image_files[:min_length]
        self.irradiance_values = self.irradiance_values[:min_length]

        # Calculate valid sequence indices
        self.valid_indices = list(range(len(self.image_files) - sequence_length - forecast_horizon + 1))

        # Initialize image processor
        self.image_processor = IRImageProcessor(target_size=target_size)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]

        # Get sequence of images
        image_sequence = []
        for i in range(start_idx, start_idx + self.sequence_length):
            img_path = os.path.join(self.image_dir, self.image_files[i])

            # Load and process image
            img = cv2.imread(img_path)
            if img is None:
                img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
                if img is not None:
                    img = self.image_processor.process_single_image(img_path)
                else:
                    raise ValueError(f"Could not load image: {img_path}")
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0

            if self.transform:
                img_tensor = self.transform(img_tensor)

            image_sequence.append(img_tensor)

        # Stack images into sequence tensor
        image_sequence = torch.stack(image_sequence)  # (seq_len, channels, height, width)

        # Get historical irradiance values (for LSTM input)
        historical_irradiance = torch.tensor(
            self.irradiance_values[start_idx:start_idx + self.sequence_length],
            dtype=torch.float32
        ).unsqueeze(-1)  # (seq_len, 1)

        # Get future irradiance values (targets)
        future_irradiance = torch.tensor(
            self.irradiance_values[start_idx + self.sequence_length:
                                 start_idx + self.sequence_length + self.forecast_horizon],
            dtype=torch.float32
        )  # (forecast_horizon,)

        return image_sequence, historical_irradiance, future_irradiance


class SolarTimeSeriesDataset(Dataset):
    """
    Simple time series dataset for LSTM-only training
    """

    def __init__(self, irradiance_file, sequence_length=20, forecast_horizon=4):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

        # Load irradiance data
        if irradiance_file.endswith('.csv'):
            df = pd.read_csv(irradiance_file)
            self.data = df.iloc[:, 1].values.astype(np.float32)
        else:
            self.data = np.loadtxt(irradiance_file, delimiter=',')[:, 1].astype(np.float32)

        # Calculate valid indices
        self.valid_indices = list(range(len(self.data) - sequence_length - forecast_horizon + 1))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]

        # Input sequence
        x = torch.tensor(
            self.data[start_idx:start_idx + self.sequence_length],
            dtype=torch.float32
        ).unsqueeze(-1)  # (seq_len, 1)

        # Target sequence
        y = torch.tensor(
            self.data[start_idx + self.sequence_length:
                     start_idx + self.sequence_length + self.forecast_horizon],
            dtype=torch.float32
        )  # (forecast_horizon,)

        return x, y


# Legacy dataset for backward compatibility
class GSIDataset(SolarIrradianceDataset):
    """Legacy dataset class - kept for compatibility"""

    def __init__(self, image_dir, gsi_file):
        super().__init__(image_dir, gsi_file)

    def __getitem__(self, idx):
        img_tensor, irradiance = super().__getitem__(idx)
        return img_tensor, irradiance


class GSITimeSeriesDataset(SolarTimeSeriesDataset):
    """Legacy time series dataset - kept for compatibility"""

    def __init__(self, gsi_values, sequence_length=10):
        self.sequence_length = sequence_length
        self.data = gsi_values.astype(np.float32)
        self.valid_indices = list(range(len(self.data) - sequence_length))

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        x = torch.tensor(
            self.data[start_idx:start_idx + self.sequence_length],
            dtype=torch.float32
        ).unsqueeze(-1)
        y = torch.tensor(self.data[start_idx + self.sequence_length], dtype=torch.float32)
        return x, y