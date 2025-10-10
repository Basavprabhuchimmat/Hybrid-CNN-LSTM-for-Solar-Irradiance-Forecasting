import cv2
import numpy as np
import os
from tqdm import tqdm
import torch
from torchvision import transforms

class IRImageProcessor:
    """
    Image processor for infrared sky images following the paper methodology:
    1. Normalization
    2. Bicubic interpolation for upscaling
    3. OpenCV colormap (JET) application
    """

    def __init__(self, target_size=(240, 320), colormap=cv2.COLORMAP_JET):
        self.target_size = target_size
        self.colormap = colormap

    def process_single_image(self, ir_image_path):
        """Process a single IR image following paper methodology"""
        # Read IR image (16-bit depth preserved)
        ir_image = cv2.imread(ir_image_path, cv2.IMREAD_ANYDEPTH)

        # Step 1: Normalization to 0-255 range
        img_normalized = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX)

        # Step 2: Bicubic interpolation for upscaling
        img_upscaled = cv2.resize(
            img_normalized, 
            self.target_size, 
            interpolation=cv2.INTER_CUBIC
        )

        # Step 3: Apply OpenCV colormap (convert to RGB)
        img_colored = cv2.applyColorMap(
            img_upscaled.astype(np.uint8), 
            self.colormap
        )

        # Convert BGR to RGB for proper processing
        img_rgb = cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB)

        return img_rgb

    def process_batch_images(self, input_dir, output_dir):
        """Process all images in a directory"""
        os.makedirs(output_dir, exist_ok=True)

        image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in tqdm(image_files, desc="Processing IR images"):
            img_path = os.path.join(input_dir, img_file)
            processed_img = self.process_single_image(img_path)

            output_path = os.path.join(output_dir, img_file)
            # Save as RGB
            cv2.imwrite(output_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))

    def get_tensor_transform(self):
        """Get PyTorch transform for processed images"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def preprocess_for_training(input_dir, output_dir, target_size=(240, 320)):
    """Main preprocessing function for training data"""
    processor = IRImageProcessor(target_size)
    processor.process_batch_images(input_dir, output_dir)
    print(f"Preprocessing complete. Processed images saved to {output_dir}")

def preprocess_multiple_days(input_dirs, output_base_dir, target_size=(240, 320)):
    """Preprocess IR images from multiple days"""
    for input_dir in input_dirs:
        date_folder = os.path.basename(os.path.dirname(input_dir.rstrip('/\\')))
        output_dir = os.path.join(output_base_dir, date_folder)
        preprocess_for_training(input_dir, output_dir, target_size)

if __name__ == '__main__':
    # Example usage for multiple days
    input_dirs = [
        r'D:\Projects\Hybrid CNN-LSTM for Solar Irradiance Forecasting\GIRASOL_DATASET\2019_01_15\infrared',
        r'D:\Projects\Hybrid CNN-LSTM for Solar Irradiance Forecasting\GIRASOL_DATASET\2019_01_16\infrared',
        r'D:\Projects\Hybrid CNN-LSTM for Solar Irradiance Forecasting\GIRASOL_DATASET\2019_01_17\infrared',
        r'D:\Projects\Hybrid CNN-LSTM for Solar Irradiance Forecasting\GIRASOL_DATASET\2019_01_18\infrared',
        r'D:\Projects\Hybrid CNN-LSTM for Solar Irradiance Forecasting\GIRASOL_DATASET\2019_01_19\infrared',
        r'D:\Projects\Hybrid CNN-LSTM for Solar Irradiance Forecasting\GIRASOL_DATASET\2019_01_20\infrared'        
    ]
    preprocess_multiple_days(
        input_dirs=input_dirs,
        output_base_dir='data/processed'
    )
