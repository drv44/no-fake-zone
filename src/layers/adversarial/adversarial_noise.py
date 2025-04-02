# import torch

# def fgsm_attack(model, loss_fn, inputs, labels, epsilon=0.01):
#     """
#     Generates adversarial examples using the FGSM method.
#     Args:
#         model (nn.Module): The model under attack.
#         loss_fn (callable): Loss function used to compute gradients.
#         inputs (Tensor): Input data (e.g., images or embeddings).
#         labels (Tensor): True labels corresponding to inputs.
#         epsilon (float): Magnitude of the perturbation.
#     Returns:
#         Tensor: Adversarially perturbed inputs.
#     """
#     # Enable gradients on inputs
#     inputs.requires_grad = True
    
#     # Forward pass
#     outputs = model(inputs)
#     loss = loss_fn(outputs, labels)
    
#     # Backward pass: compute gradients with respect to inputs
#     model.zero_grad()
#     loss.backward()
#     data_grad = inputs.grad.data
    
#     # Generate perturbed input by adjusting each pixel by epsilon in the direction of the gradient sign
#     perturbed_inputs = inputs + epsilon * data_grad.sign()
    
#     # Clamp the perturbed images to be within valid range, e.g. [0, 1] for normalized images
#     perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
#     return perturbed_inputs

# if __name__ == "__main__":
#     # Example usage with a dummy model
#     import torch.nn as nn

#     class DummyModel(nn.Module):
#         def forward(self, x):
#             # Simple dummy forward: compute mean over features
#             return x.mean(dim=1)
    
#     model = DummyModel()
#     loss_fn = nn.MSELoss()
#     # Dummy inputs (e.g., images): batch of 4 images with 3 channels and size 224x224
#     dummy_inputs = torch.rand(4, 3, 224, 224)
#     # Dummy labels: a tensor with the same batch size
#     dummy_labels = torch.rand(4)
    
#     perturbed = fgsm_attack(model, loss_fn, dummy_inputs, dummy_labels, epsilon=0.05)
#     print("Perturbed inputs shape:", perturbed.shape)

import torch
import torch.nn as nn

def fgsm_attack(model, loss_fn, inputs, labels, epsilon=0.01):
    """
    Generates adversarial examples using the FGSM method.
    Args:
        model (nn.Module): The model under attack.
        loss_fn (callable): Loss function used to compute gradients.
        inputs (Tensor): Input data (e.g., images or embeddings).
        labels (Tensor): True labels corresponding to inputs.
        epsilon (float): Magnitude of the perturbation.
    Returns:
        Tensor: Adversarially perturbed inputs.
    """
    # Enable gradients on inputs
    inputs.requires_grad = True
    
    # Forward pass
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    
    # Backward pass: compute gradients with respect to inputs
    model.zero_grad()
    loss.backward()
    data_grad = inputs.grad.data
    
    # Generate perturbed input by adjusting each pixel by epsilon in the direction of the gradient sign
    perturbed_inputs = inputs + epsilon * data_grad.sign()
    
    # Clamp the perturbed images to be within valid range, e.g., [0, 1] for normalized images
    perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
    return perturbed_inputs

if __name__ == "__main__":
    # Updated dummy model that outputs a scalar per sample.
    class DummyModel(nn.Module):
        def forward(self, x):
            # Flatten and average over all dimensions except the batch dimension.
            return x.view(x.size(0), -1).mean(dim=1)
    
    model = DummyModel()
    loss_fn = nn.MSELoss()
    
    # Dummy inputs (e.g., images): batch of 4 images with 3 channels and size 224x224
    dummy_inputs = torch.rand(4, 3, 224, 224)
    # Dummy labels: a scalar for each sample (matching the model output shape)
    dummy_labels = torch.rand(4)
    
    perturbed = fgsm_attack(model, loss_fn, dummy_inputs, dummy_labels, epsilon=0.05)
    print("Perturbed inputs shape:", perturbed.shape)
