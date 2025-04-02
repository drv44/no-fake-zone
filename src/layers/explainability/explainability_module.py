import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from captum.attr import IntegratedGradients

def get_pretrained_model():
    model = models.resnet18(pretrained=True)  # Load a pretrained model
    model.fc = nn.Linear(512, 2)  # Adjust for binary classification
    return model

class ExplainabilityModule:
    def __init__(self, model):
        self.model = model
        self.model.train()  # Enable gradients
        for param in self.model.parameters():
            param.requires_grad = True
        self.ig = IntegratedGradients(self.model)

    def get_attributions(self, inputs, target=0, n_steps=50):
        inputs.requires_grad = True  # Ensure input requires gradients
        self.model.zero_grad()  # Zero out gradients

        outputs = self.model(inputs)
        if not outputs.requires_grad:
            outputs.retain_grad()  # Ensure output retains gradients

        attributions, delta = self.ig.attribute(inputs, target=target, n_steps=n_steps, return_convergence_delta=True)
        return attributions, delta

if __name__ == "__main__":
    model = get_pretrained_model()
    explainer = ExplainabilityModule(model)

    # Create a dummy input
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)  # Enable gradients
    target_class = 0  # Assuming binary classification

    attributions, delta = explainer.get_attributions(dummy_input, target=target_class)
    print("Attributions:", attributions)
    print("Delta (convergence measure):", delta)
