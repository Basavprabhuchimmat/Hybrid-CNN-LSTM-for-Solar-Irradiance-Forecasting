import torch
import torch.nn as nn

class SolarLSTMForecasting(nn.Module):
    """
    LSTM model for solar irradiance forecasting .
    Uses bidirectional LSTM with 2 layers and 128 hidden units per direction.
    """

    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=4, dropout=0.2, bidirectional=True, use_legacy_head=False):
        super(SolarLSTMForecasting, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.use_legacy_head = use_legacy_head

        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)

        if self.use_legacy_head:
            self.fc = nn.Linear(lstm_output_size, output_size)
        else:
            self.fc_layers = nn.Sequential(
                nn.Linear(lstm_output_size, 128),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(128, 64),
                nn.ReLU(), 
                nn.Dropout(dropout),

                nn.Linear(64, output_size)
            )

        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)

        if self.use_legacy_head:
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.constant_(self.fc.bias, 0)
        else:
            for m in self.fc_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        lstm_out, (hidden, cell) = self.lstm(x)

        last_output = lstm_out[:, -1, :]  

        if self.use_legacy_head:
            forecast = self.fc(last_output)
        else:
            forecast = self.fc_layers(last_output)

        return forecast


class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model as described in the paper.
    CNN extracts features from images, LSTM performs temporal forecasting.
    """

    def __init__(self, cnn_model=None, feature_dim=512, sequence_length=20, 
                 lstm_hidden_size=128, forecast_horizon=4):
        super(HybridCNNLSTM, self).__init__()

        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

        if cnn_model is None:
            from scripts.cnn_model import SolarCNNWithFeatureExtraction
            self.cnn = SolarCNNWithFeatureExtraction(feature_dim=feature_dim)
        else:
            self.cnn = cnn_model

        self.lstm = SolarLSTMForecasting(
            input_size=1,  
            hidden_size=lstm_hidden_size,
            output_size=forecast_horizon
        )

        self.freeze_cnn = False

    def set_cnn_trainable(self, trainable=True):
        """Control whether CNN parameters are trainable"""
        for param in self.cnn.parameters():
            param.requires_grad = trainable
        self.freeze_cnn = not trainable

    def forward(self, image_sequence):
        """
        Forward pass for the hybrid model

        Args:
            image_sequence: Tensor of shape (batch_size, sequence_length, channels, height, width)

        Returns:
            nowcasts: Current irradiance predictions for each image
            forecasts: Future irradiance predictions
        """
        batch_size, seq_len, channels, height, width = image_sequence.shape

        images_flat = image_sequence.view(-1, channels, height, width)

        nowcasts = self.cnn(images_flat)  
        nowcasts = nowcasts.view(batch_size, seq_len, 1)  

        forecasts = self.lstm(nowcasts)  

        return nowcasts, forecasts

    def predict_from_sequence(self, image_sequence):
        """Prediction method for inference"""
        self.eval()
        with torch.no_grad():
            nowcasts, forecasts = self.forward(image_sequence)
        return nowcasts, forecasts


class SolarLSTM(nn.Module):
    """Legacy LSTM model - kept for compatibility"""

    def __init__(self, input_size=1, hidden_size=128, output_size=1):
        super(SolarLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output