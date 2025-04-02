import sys
import os

# Add 'src' directory to sys.path
sys.path.append(os.path.abspath("src"))

# Now import the classifier model
from layers.classification.fc_layers import FCClassifier

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Training Parameters
input_dim, hidden_dim, output_dim = 256, 128, 2
learning_rate, num_epochs = 0.001, 10

# Initialize Model
model = FCClassifier(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Dummy Dataset (Replace with real data)
X_train = torch.randn(100, input_dim)
y_train = torch.randint(0, 2, (100,))
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8)

# Training Loop
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), f"models/fake_news_model_epoch_{epoch+1}.pth")
    print(f"Checkpoint saved: models/fake_news_model_epoch_{epoch+1}.pth")

torch.save(model.state_dict(), "models/fake_news_model.pth")
print("Final model saved: models/fake_news_model.pth")
