import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class SBERTSiamese(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        super(SBERTSiamese, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def encode(self, input_texts):
        """
        Encodes a list of texts into embeddings.
        Args:
            input_texts (List[str]): List of sentences.
        Returns:
            Tensor: Embeddings of shape (batch_size, embedding_dim)
        """
        encoded = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        outputs = self.encoder(**encoded)
        # Mean pooling over the token dimension
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    def forward(self, text1, text2):
        """
        Computes cosine similarity between two sets of texts.
        Args:
            text1 (List[str]): First list of sentences.
            text2 (List[str]): Second list of sentences.
        Returns:
            Tensor: Cosine similarity scores for each pair.
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        cos_sim = F.cosine_similarity(emb1, emb2)
        return cos_sim

if __name__ == "__main__":
    model = SBERTSiamese()
    text1 = ["This is a fake news article."]
    text2 = ["This news article is likely fabricated."]
    similarity = model(text1, text2)
    print("Cosine Similarity:", similarity.item())
