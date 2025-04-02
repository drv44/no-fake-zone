import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class ModelEvaluator:
    def __init__(self, model, dataloader, device=None):
        self.model = model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dataloader = dataloader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self):
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                predictions = torch.argmax(outputs, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
        conf_matrix = confusion_matrix(y_true, y_pred)

        return {"accuracy": acc, "precision": precision, "recall": recall, "f1_score": f1, "confusion_matrix": conf_matrix}

# Example Usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from fake_news_model import FakeNewsClassifier  # Assume you have this model
    from dataset_loader import FakeNewsDataset     # Assume you have a dataset loader

    test_loader = DataLoader(FakeNewsDataset("test.csv"), batch_size=32, shuffle=False)
    model = FakeNewsClassifier()

    evaluator = ModelEvaluator(model, test_loader)
    results = evaluator.evaluate()
    print("Evaluation Metrics:", results)
