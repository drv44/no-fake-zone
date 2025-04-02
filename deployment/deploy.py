import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification

# Load Model
model_path = "model_checkpoint.pth"
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Initialize FastAPI
app = FastAPI()

class NewsRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: NewsRequest):
    inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return {"prediction": "Fake News" if prediction == 1 else "Real News"}

# Run with: `uvicorn deploy:app --host 0.0.0.0 --port 8000`
