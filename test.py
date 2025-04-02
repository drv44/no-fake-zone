# # import os
# # import torch
# # import torch.nn.functional as F
# # import torchvision.transforms as transforms
# # from PIL import Image
# # from src.layers.classification.fc_layers import FCClassifier  # Import your classifier

# # # Define model parameters (Ensure they match training)
# # input_dim_text = 256   # Feature size for text
# # input_dim_image = 512  # Feature size for image
# # hidden_dim = 128       # Hidden layer size
# # output_dim = 2         # Binary classification (FAKE/REAL)

# # # Load trained model
# # MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "fake_news_model.pth")
# # model = FCClassifier(input_dim_text, hidden_dim, output_dim)  # Adjust input_dim if testing image
# # model.load_state_dict(torch.load(MODEL_PATH))
# # model.eval()  # Set model to evaluation mode

# # # Define image preprocessing
# # transform = transforms.Compose([
# #     transforms.Resize((224, 224)),  # Resize image for CNN input
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
# # ])

# # # Function to convert text into a feature vector
# # def text_to_vector(text):
# #     # Replace with actual text processing (TF-IDF, BERT, etc.)
# #     return torch.randn(1, input_dim_text)  # Simulated text features

# # # Function to convert image into a feature vector
# # def image_to_vector(image_path):
# #     image = Image.open(image_path).convert("RGB")  # Load image
# #     image = transform(image).unsqueeze(0)  # Preprocess and add batch dimension
# #     return torch.randn(1, input_dim_text)  # Simulated image features

# # # Function to classify news
# # def classify_news(input_data, is_text=True):
# #     if is_text:
# #         input_vector = text_to_vector(input_data)  # Convert text to vector
# #     else:
# #         input_vector = image_to_vector(input_data)  # Convert image to vector

# #     with torch.no_grad():
# #         logits = model(input_vector)  # Forward pass
# #         probs = F.softmax(logits, dim=1)  # Convert to probabilities
# #         prediction = torch.argmax(probs, dim=1).item()  # Get class label

# #     return "FAKE" if prediction == 0 else "REAL"

# # # Example Testing
# # input_text = "Donald Trump became president of india"
# # print(f"Predicted News Category (Text): {classify_news(input_text, is_text=True)}")

# # input_image_path = "data//image_data//test//real//290269670_10161560661591756_4441974433210634991_n_png.rf.25142f031e778a417a9ab21d1d126651.jpg"
# # print(f"Predicted News Category (Image): {classify_news(input_image_path, is_text=False)}")



# # from utils.fact_check import fact_check_news

# # def classify_news(input_data, is_text=True):
# #     """
# #     Classifies input as fake or real news and fact-checks the claim.
    
# #     Args:
# #         input_data (str or image path): The input text or image path.
# #         is_text (bool): Whether the input is text (True) or image (False).

# #     Returns:
# #         str: Prediction result with explanation and fact-checking info.
# #     """
# #     # Preprocess input based on type
# #     if is_text:
# #         input_vector = text_preprocessor(input_data)  # Convert text to vector
# #     else:
# #         input_vector = image_preprocessor(input_data)  # Convert image to vector
    
# #     input_vector = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# #     # Get prediction
# #     logits = model(input_vector)  
# #     prediction = torch.argmax(logits, dim=1).item()  # Get class index (0 = Fake, 1 = Real)

# #     # Define explanations
# #     explanations = {
# #         0: "FAKE NEWS: The claim is not supported by factual evidence.",
# #         1: "REAL NEWS: The claim is supported by reliable sources."
# #     }

# #     # Google Fact Check API Call
# #     fact_check_result = fact_check_news(input_data)

# #     return f"{explanations[prediction]}\nFact-Check Result: {fact_check_result['rating']} (Source: {fact_check_result['source']})"

# # # Example Test
# # input_text = "Donald Trump became president of India"
# # print(f"Model Output: {classify_news(input_text, is_text=True)}")




# import torch
# from utils.fact_check import fact_check_news
# from src.data_preprocessing.text_preprocess import main
# from src.data_preprocessing.image_preprocess import main
# from src.layers.classification.fc_layers import FCClassifier

# # Load the trained model (Ensure the correct model path is used)
# MODEL_PATH = "models/fake_news_model.pth"
# input_dim = 256  # Adjust according to the model
# hidden_dim = 128  # Adjust according to the model
# output_dim = 2  # Binary classification (Fake/Real)

# model = FCClassifier(input_dim, hidden_dim, output_dim)
# model.load_state_dict(torch.load(MODEL_PATH))
# model.eval()

# def classify_news(input_data, is_text=True):
#     """
#     Classifies input as fake or real news and fact-checks the claim.
    
#     Args:
#         input_data (str or image path): The input text or image path.
#         is_text (bool): Whether the input is text (True) or image (False).
    
#     Returns:
#         str: Prediction result with explanation and fact-checking info.
#     """
#     # Preprocess input based on type
#     if is_text:
#         input_vector = main(input_data)  # Convert text to vector
#     else:
#         input_vector = main(input_data)  # Convert image to vector
    
#     input_vector = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

#     # Get prediction
#     logits = model(input_vector)  
#     prediction = torch.argmax(logits, dim=1).item()  # Get class index (0 = Fake, 1 = Real)

#     # Define explanations
#     explanations = {
#         0: "FAKE NEWS: The claim is not supported by factual evidence.",
#         1: "REAL NEWS: The claim is supported by reliable sources."
#     }

#     # Google Fact Check API Call
#     fact_check_result = fact_check_news(input_data if is_text else "")

#     return f"{explanations[prediction]}\nFact-Check Result: {fact_check_result['rating']} (Source: {fact_check_result['source']})"

# # Example Test
# if __name__ == "__main__":
#     input_text = "Donald Trump became president of India"
#     print(f"Model Output (Text): {classify_news(input_text, is_text=True)}")
    
#     input_image_path = "data/test_images/news_sample.jpg"
#     print(f"Model Output (Image): {classify_news(input_image_path, is_text=False)}")


import torch
import torchvision.transforms as transforms
from PIL import Image
from src.data_preprocessing.text_preprocess import text_preprocess
# from models.fake_news_model import FakeNewsClassifier
from src.layers.classification.fc_layers import FCClassifier
import numpy as np

# Load the trained model
# Define input, hidden, and output dimensions based on your dataset
INPUT_DIM = 100  # Example: Adjust based on your actual input feature size
HIDDEN_DIM = 50  # Example: Adjust based on model architecture
OUTPUT_DIM = 2   # Assuming binary classification (FAKE vs REAL)

# Initialize model correctly
model = FCClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

MODEL_PATH = "models/fake_news_model.pth"
# model = FCClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

def classify_news(input_data, is_text=True):
    """
    Classifies input as real or fake news.
    
    Args:
        input_data (str): The text or image path.
        is_text (bool): True for text, False for image.
    
    Returns:
        str: 'REAL' or 'FAKE'
    """
    if is_text:
        # Preprocess text
        tokens = text_preprocess(input_data)
        input_vector = np.array([len(tokens)])  # Simple feature: number of words
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
    else:
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        image = Image.open(input_data).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

    # Forward pass
    logits = model(input_tensor)
    prediction = torch.argmax(logits, dim=1).item()
    return "REAL" if prediction == 1 else "FAKE"

if __name__ == "__main__":
    # Test with text
    input_text = "Donald Trump became president of India."
    print(f"Model Output (Text): {classify_news(input_text, is_text=True)}")

    # Test with an image
    input_image_path = "data/images/sample_news.jpg"
    print(f"Model Output (Image): {classify_news(input_image_path, is_text=False)}")
