# revised_model.py

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class DeepfakeDetector(nn.Module):
    """
    A hybrid deepfake detection model combining a pre-trained EfficientNet-B0
    for spatial feature extraction and a GRU for temporal analysis.
    """
    def __init__(self, pretrained=True, rnn_hidden_size=128, rnn_num_layers=1, dropout_prob=0.5):
        """
        Args:
            pretrained (bool): Whether to use pre-trained weights for EfficientNet.
            rnn_hidden_size (int): The number of features in the hidden state of the GRU.
            rnn_num_layers (int): Number of recurrent layers.
            dropout_prob (float): Dropout probability for the final classifier.
        """
        super(DeepfakeDetector, self).__init__()

        # 1. Load pre-trained EfficientNet-B0 as the feature extractor
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.cnn = efficientnet_b0(weights=weights)
        else:
            self.cnn = efficientnet_b0(weights=None)

        # The feature dimension from EfficientNet-B0's adaptive_avgpool2d is 1280
        cnn_feature_dim = self.cnn.classifier[1].in_features
        self.cnn.classifier = nn.Identity()  # Remove the original classifier

        # 2. GRU for temporal feature learning
        self.rnn = nn.GRU(
            input_size=cnn_feature_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,  # Input/output tensors are (batch, seq, feature)
            bidirectional=False # Unidirectional is often sufficient and more memory efficient
        )

        # 3. Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, 1) # Output a single logit for binary classification
        )

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W)
                              B: Batch size, T: Sequence length (num frames)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, 1)
        """
        batch_size, seq_length, c, h, w = x.shape
        
        # Reshape for CNN processing: (B, T, C, H, W) -> (B * T, C, H, W)
        x_reshaped = x.view(batch_size * seq_length, c, h, w)
        
        # Get spatial features from the CNN
        cnn_features = self.cnn(x_reshaped)
        
        # Reshape back for RNN processing: (B * T, feature_dim) -> (B, T, feature_dim)
        rnn_input = cnn_features.view(batch_size, seq_length, -1)
        
        # Get temporal features from the RNN
        # We only need the last hidden state (or the output of the last time step)
        # rnn_output shape: (B, T, hidden_size)
        # hidden_state shape: (num_layers, B, hidden_size)
        self.rnn.flatten_parameters() # Improves performance with DataParallel/DDP
        rnn_output, hidden_state = self.rnn(rnn_input)
        
        # Use the output of the last time step for classification
        last_time_step_output = rnn_output[:, -1, :]
        
        # Pass through the final classifier
        output = self.classifier(last_time_step_output)
        
        return output