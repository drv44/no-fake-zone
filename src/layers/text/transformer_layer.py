import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class TransformerLayer(nn.Module):
    def __init__(self, model_name='bert-base-uncased', output_dim=768):
        super(TransformerLayer, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.output_dim = output_dim

    def forward(self, input_texts):
        """
        Args:
            input_texts (List[str]): A list of input sentences.
        Returns:
            Tensor: Contextualized embeddings of shape (batch_size, output_dim)
        """
        encoded_inputs = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        outputs = self.bert(**encoded_inputs)
        # Pooling: take the mean of the last hidden state tokens
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

if __name__ == "__main__":
    # Example usage:
    layer = TransformerLayer()
    sample_texts = ["This is a test sentence.", "Deep learning models are amazing."]
    embeddings = layer(sample_texts)
    print("Embeddings shape:", embeddings.shape)  # Expected: (batch_size, 768)
