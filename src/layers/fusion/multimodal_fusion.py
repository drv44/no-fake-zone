import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    def __init__(self, text_dim, image_dim, fusion_dim):
        """
        Args:
            text_dim (int): Dimensionality of the text feature vector.
            image_dim (int): Dimensionality of the image feature vector.
            fusion_dim (int): Dimensionality of the common fused representation.
        """
        super(MultimodalFusion, self).__init__()
        # Project text and image features to the same dimension
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        
        # Fusion layer: concatenate the projected features and pass them through FC
        self.fusion_fc = nn.Linear(2 * fusion_dim, fusion_dim)
        self.activation = nn.ReLU()

    def forward(self, text_features, image_features):
        """
        Args:
            text_features (Tensor): Tensor of shape (batch_size, text_dim)
            image_features (Tensor): Tensor of shape (batch_size, image_dim)
        Returns:
            Tensor: Fused feature representation of shape (batch_size, fusion_dim)
        """
        # Project features into the common space
        text_proj = self.text_proj(text_features)
        image_proj = self.image_proj(image_features)
        
        # Concatenate along the feature dimension
        fused = torch.cat((text_proj, image_proj), dim=1)
        fused = self.activation(self.fusion_fc(fused))
        return fused

if __name__ == "__main__":
    # Example usage:
    batch_size = 8
    text_features = torch.randn(batch_size, 768)
    image_features = torch.randn(batch_size, 512)
    fusion_model = MultimodalFusion(text_dim=768, image_dim=512, fusion_dim=256)
    fused_features = fusion_model(text_features, image_features)
    print("Fused features shape:", fused_features.shape)  # Expected: (8, 256)
