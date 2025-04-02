import torch
import torch.nn as nn
import torchvision.models as models

class CNNLayer(nn.Module):
    def __init__(self, output_dim=512, pretrained=True):
        """
        Args:
            output_dim (int): Dimension of the output feature vector.
            pretrained (bool): Whether to use pretrained weights.
        """
        super(CNNLayer, self).__init__()
        # Load a pre-trained ResNet-18 model
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final classification layer (fc)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])  # Output: (batch, 512, 1, 1)
        
        # New fully connected layer to get desired output dimension
        self.fc = nn.Linear(512, output_dim)
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input image tensor of shape (batch_size, 3, H, W)
        Returns:
            Tensor: Image feature embeddings of shape (batch_size, output_dim)
        """
        x = self.features(x)        # Shape: (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)     # Flatten to shape: (batch, 512)
        x = self.fc(x)              # Shape: (batch, output_dim)
        return x

if __name__ == "__main__":
    # Example usage:
    model = CNNLayer(output_dim=256)
    dummy_input = torch.randn(8, 3, 224, 224)
    output = model(dummy_input)
    print("CNNLayer output shape:", output.shape)  # Expected: (8, 256)
