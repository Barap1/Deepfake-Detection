import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class DeepfakeDetector(nn.Module):
    """
    A hybrid deepfake detection model combining a pre-trained EfficientNet-B0
    for spatial feature extraction and a GRU for temporal analysis.
    """
    def __init__(self, pretrained=True, rnn_hidden_size=128, rnn_num_layers=1, dropout_prob=0.5):
 
        super(DeepfakeDetector, self).__init__()

        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.cnn = efficientnet_b0(weights=weights)
        else:
            self.cnn = efficientnet_b0(weights=None)

        cnn_feature_dim = self.cnn.classifier[1].in_features
        self.cnn.classifier = nn.Identity() 

        self.rnn = nn.GRU(
            input_size=cnn_feature_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True, 
            bidirectional=False 
        )

        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        batch_size, seq_length, c, h, w = x.shape
        
        x_reshaped = x.view(batch_size * seq_length, c, h, w)
        
        cnn_features = self.cnn(x_reshaped)
        
        rnn_input = cnn_features.view(batch_size, seq_length, -1)

        self.rnn.flatten_parameters()  # Improves performance with DataParallel/DDP
        rnn_output, hidden_state = self.rnn(rnn_input)
        
        last_time_step_output = rnn_output[:, -1, :]
        
        # Pass through the final classifier
        output = self.classifier(last_time_step_output)
        
        return output