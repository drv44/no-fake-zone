import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn.functional as F

class CounterargumentGenerator:
    def __init__(self, model_name="t5-small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def generate_counterargument(self, claim, max_length=128):
        input_text = f"counterargument: {claim}"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        
        output_ids = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    generator = CounterargumentGenerator()
    claim = "Artificial intelligence will replace most jobs in the future."
    print("Counterargument:", generator.generate_counterargument(claim))
