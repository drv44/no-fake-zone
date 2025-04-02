
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        """
        Args:
            input_dim (int): Dimension of the input feature vector.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Number of classes (e.g., 2 for binary classification).
            dropout_rate (float): Dropout probability.
        """
        super(FCClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input features of shape (batch_size, input_dim)
        Returns:
            Tensor: Logits for each class of shape (batch_size, output_dim)
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Example usage:
    batch_size = 8
    input_dim = 256
    hidden_dim = 128
    output_dim = 2  # e.g., fake or real
    model = FCClassifier(input_dim, hidden_dim, output_dim)
    dummy_input = torch.randn(batch_size, input_dim)
    logits = model(dummy_input)
    print("Logits shape:", logits.shape)  # Expected: (8, 2)
