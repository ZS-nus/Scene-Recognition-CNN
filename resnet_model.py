import torch
import torch.nn as nn
import torchvision.models as models
from model import get_device

class ResNet50Transfer(nn.Module):
    def __init__(self, num_classes=15, pretrained=True):
        """
        Initialize ResNet50 model with transfer learning.
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
        """
        super(ResNet50Transfer, self).__init__()
        
        # Load pretrained ResNet-50 model
        self.resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Replace the final fully connected layer to match our number of classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),  # Add dropout for regularization
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)
        
    def freeze_backbone(self, freeze=True):
        """
        Freeze or unfreeze the backbone ResNet layers
        
        Args:
            freeze (bool): Whether to freeze layers (True) or make them trainable (False)
        """
        # Freeze/unfreeze all parameters except final fully connected layer
        for name, param in self.resnet.named_parameters():
            if "fc" not in name:  # Exclude the final FC layer
                param.requires_grad = not freeze
                
    def unfreeze_layers_from(self, layer_name):
        """
        Unfreeze ResNet layers starting from a specified layer
        
        Args:
            layer_name (str): Layer to start unfreezing from (e.g., 'layer4')
        """
        # First freeze everything
        self.freeze_backbone(True)
        
        # Then unfreeze from the specified layer
        unfreezing = False
        for name, param in self.resnet.named_parameters():
            if layer_name in name:
                unfreezing = True
            if unfreezing:
                param.requires_grad = True