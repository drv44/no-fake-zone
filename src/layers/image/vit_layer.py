import torch
import torch.nn as nn
import timm

class ViTLayer(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", output_dim=768, pretrained=True):
        """
        Args:
            model_name (str): Name of the ViT model architecture.
            output_dim (int): Desired output dimension of the features.
            pretrained (bool): Whether to load pretrained weights.
        """
        super(ViTLayer, self).__init__()
        # Load a pre-trained ViT model from timm
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        
        # Remove the classification head. Note: timm models usually have a 'head' attribute.
        if hasattr(self.vit, 'head'):
            in_features = self.vit.head.in_features
            self.vit.head = nn.Identity()
        else:
            raise ValueError("The model does not have a 'head' attribute to remove.")
        
        # Optional: a linear projection to desired output dimension
        self.fc = nn.Linear(in_features, output_dim) if in_features != output_dim else nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input image tensor of shape (batch_size, 3, H, W)
        Returns:
            Tensor: Image feature embeddings of shape (batch_size, output_dim)
        """
        x = self.vit(x)  # Extract features using ViT (output shape depends on model configuration)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Example usage:
    model = ViTLayer(model_name="vit_base_patch16_224", output_dim=512)
    dummy_input = torch.randn(8, 3, 224, 224)
    output = model(dummy_input)
    print("ViTLayer output shape:", output.shape)  # Expected: (8, 512)
