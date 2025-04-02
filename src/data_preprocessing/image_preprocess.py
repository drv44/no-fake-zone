import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class NewsImageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and preprocessing images.
    Expects a directory structure like:
      data/images/train/real/
      data/images/train/fake/
      data/images/test/real/
      data/images/test/fake/
      data/images/valid/real/
      data/images/valid/fake/
    """
    def __init__(self, image_dir: str, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing subfolders (e.g., 'real', 'fake').
            transform (callable, optional): Transformations to be applied to the images.
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Find all jpg and png files recursively within the image_dir
        self.image_paths = glob.glob(os.path.join(image_dir, '**', '*.jpg'), recursive=True)
        self.image_paths += glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True)
        
        # Set labels based on the folder name: 'real' -> 1, 'fake' -> 0.
        self.labels = []
        for path in self.image_paths:
            if 'real' in os.path.normpath(path).lower():
                self.labels.append(1)
            else:
                self.labels.append(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open image and convert to RGB
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_transforms(image_size: int = 224):
    """
    Creates a set of transforms for image preprocessing.
    Args:
        image_size (int): Target image size (width and height).
    Returns:
        transforms.Compose: Composed transformations.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def load_image_dataset(image_dir: str, batch_size: int = 32, image_size: int = 224, shuffle: bool = True):
    """
    Loads the image dataset from a given directory and returns a PyTorch DataLoader.
    Args:
        image_dir (str): Path to the directory containing the image data.
        batch_size (int): Batch size.
        image_size (int): Size to resize the images.
        shuffle (bool): Whether to shuffle the data.
    Returns:
        DataLoader: DataLoader for the dataset.
    """
    transform = create_transforms(image_size)
    dataset = NewsImageDataset(image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def main():
    # Example: Load training images from data/images/train/
    train_dir = 'data/image_data/train'
    train_loader = load_image_dataset(train_dir, batch_size=8, image_size=224, shuffle=True)
    
    # Iterate through a single batch and print shapes
    for images, labels in train_loader:
        print("Batch of images shape:", images.shape)  # Expected: [batch_size, 3, 224, 224]
        print("Batch of labels:", labels)
        break

if __name__ == "__main__":
    main()
