import argparse
import torch
import logging
from src.models.fake_news_model import FakeNewsClassifier
from src.utils.logger import setup_logging
from src.utils.config import load_config
from src.data.preprocessing import preprocess_text, preprocess_image
from src.layers.text.transformer_layer import TransformerLayer
from src.layers.image.vit_layer import ViTLayer
from src.fusion.multimodal_fusion import MultimodalFusion
from src.explainability.explainability_module import ExplainabilityModule

# Setup logging
setup_logging()
logger = logging.getLogger("FakeNewsDetection")

def load_model(model_path):
    model = FakeNewsClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, text, image):
    text_embedding = TransformerLayer().forward(preprocess_text(text))
    image_embedding = ViTLayer().forward(preprocess_image(image))
    fused_features = MultimodalFusion().forward(text_embedding, image_embedding)
    prediction = model(fused_features)
    return prediction

def main(args):
    config = load_config("config.json")
    model = load_model(config["model_path"])
    
    logger.info("Processing input...")
    text = args.text
    image = args.image
    prediction = predict(model, text, image)
    
    if config["explainability"]:
        explainer = ExplainabilityModule(model)
        explanation = explainer.get_explanation(text)
        logger.info(f"Prediction Explanation: {explanation}")
    
    logger.info(f"Prediction: {'FAKE' if prediction > 0.5 else 'REAL'}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake News Detection System")
    parser.add_argument("--text", type=str, required=True, help="Input text for news article")
    parser.add_argument("--image", type=str, required=False, help="Optional image associated with the news")
    args = parser.parse_args()
    main(args)