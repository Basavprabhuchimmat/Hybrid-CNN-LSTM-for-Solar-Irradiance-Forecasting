import torch
import torch.nn as nn
import torch.nn.functional as F

class SolarCNNRegression(nn.Module):


    def __init__(self, input_channels=3, num_classes=1):
        super(SolarCNNRegression, self).__init__()

        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  

            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 120x160 -> 60x80

            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 60x80 -> 30x40

            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 30x40 -> 15x20

            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )

        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)  
        )

        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        features = self.features(x)

        
        irradiance = self.regressor(features)

        return irradiance


    def get_features(self, x):

        with torch.no_grad():
            features = self.features(x)
            return features.flatten(1)  


class SolarCNNWithFeatureExtraction(SolarCNNRegression):


    def __init__(self, input_channels=3, feature_dim=512):
        super().__init__(input_channels, 1)
        self.feature_dim = feature_dim

        
        self.feature_projector = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x, return_features=False):
        
        conv_features = self.features(x)
        flattened_features = conv_features.flatten(1)

        
        irradiance = self.regressor[-3:](
            self.regressor[:-3](flattened_features)
        )

        if return_features:
            
            projected_features = self.feature_projector(flattened_features)
            return irradiance, projected_features

        return irradiance



class SolarCNN(SolarCNNRegression):

    pass