import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models  # Added for ResNet50Transfer

print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

def get_device():
    """
    Check and return the best available device for PyTorch.
    Will use CUDA if available, then MPS (Metal Performance Shaders for Mac), 
    otherwise CPU.
    
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print(f"Using CPU \n")
    
    return device

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
        
        # Replace the final fully connected layer with a more complex classifier
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.4),  # Increased dropout
            nn.Linear(in_features, 1024),  # Larger intermediate layer
            nn.BatchNorm1d(1024),  # Added batch normalization
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),  # Added one more layer
            nn.BatchNorm1d(512),
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
                
    def progressive_unfreeze(self, current_epoch, stage_epochs=[5, 10, 15]):
        """
        Progressively unfreeze more layers as training progresses
        
        Args:
            current_epoch (int): Current training epoch
            stage_epochs (list): Epochs at which to unfreeze more layers
        """
        if current_epoch < stage_epochs[0]:
            # Only FC layer is trainable
            self.freeze_backbone(True)
        elif current_epoch < stage_epochs[1]:
            # Unfreeze layer4
            self.unfreeze_layers_from("layer4")
        elif current_epoch < stage_epochs[2]:
            # Unfreeze layer3 too
            self.unfreeze_layers_from("layer3")
        else:
            # Unfreeze layer2 as well
            self.unfreeze_layers_from("layer2")