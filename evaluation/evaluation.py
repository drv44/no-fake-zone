import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add 'src' directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Import the model
from layers.classification.fc_layers import FCClassifier

# Load the trained model
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "fake_news_model.pth"))

# Ensure the model parameters match training
input_dim, hidden_dim, output_dim = 256, 128, 2
model = FCClassifier(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Dummy test dataset (Replace with real test data)
X_test = torch.randn(50, input_dim)
y_test = torch.randint(0, 2, (50,))

# Make predictions
with torch.no_grad():
    logits = model(X_test)
    predictions = torch.argmax(logits, dim=1)

# Evaluate performance
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='binary')
recall = recall_score(y_test, predictions, average='binary')
f1 = f1_score(y_test, predictions, average='binary')

print(f"Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
