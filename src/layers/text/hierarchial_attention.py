import torch
import torch.nn as nn
import torch.nn.functional as F

class WordAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(WordAttention, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.context_vector = nn.Parameter(torch.randn(hidden_dim * 2))

    def forward(self, word_embeddings):
        """
        Args:
            word_embeddings: Tensor of shape (batch_size, max_words, input_dim)
        Returns:
            Tensor: Sentence vector of shape (batch_size, hidden_dim*2)
        """
        gru_output, _ = self.gru(word_embeddings)  # (batch_size, max_words, hidden_dim*2)
        u = torch.tanh(self.fc(gru_output))          # (batch_size, max_words, hidden_dim*2)
        attn_scores = torch.matmul(u, self.context_vector)  # (batch_size, max_words)
        attn_weights = F.softmax(attn_scores, dim=1)         # (batch_size, max_words)
        sentence_vector = torch.sum(gru_output * attn_weights.unsqueeze(-1), dim=1)
        return sentence_vector

class SentenceAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SentenceAttention, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.context_vector = nn.Parameter(torch.randn(hidden_dim * 2))

    def forward(self, sentence_vectors):
        """
        Args:
            sentence_vectors: Tensor of shape (batch_size, max_sentences, input_dim)
        Returns:
            Tensor: Document vector of shape (batch_size, hidden_dim*2)
        """
        gru_output, _ = self.gru(sentence_vectors)  # (batch_size, max_sentences, hidden_dim*2)
        u = torch.tanh(self.fc(gru_output))           # (batch_size, max_sentences, hidden_dim*2)
        attn_scores = torch.matmul(u, self.context_vector)  # (batch_size, max_sentences)
        attn_weights = F.softmax(attn_scores, dim=1)         # (batch_size, max_sentences)
        document_vector = torch.sum(gru_output * attn_weights.unsqueeze(-1), dim=1)
        return document_vector

class HierarchicalAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, word_hidden_dim, sentence_hidden_dim):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimensionality of word embeddings.
            word_hidden_dim (int): Hidden dimension for word-level GRU.
            sentence_hidden_dim (int): Hidden dimension for sentence-level GRU.
        """
        super(HierarchicalAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.word_attention = WordAttention(embedding_dim, word_hidden_dim)
        self.sentence_attention = SentenceAttention(word_hidden_dim * 2, sentence_hidden_dim)

    def forward(self, documents):
        """
        Args:
            documents: Tensor of shape (batch_size, max_sentences, max_words)
                       containing word indices.
        Returns:
            Tensor: Document embeddings of shape (batch_size, sentence_hidden_dim*2)
        """
        batch_size, max_sentences, max_words = documents.size()
        # Reshape documents to process each sentence individually
        documents = documents.view(batch_size * max_sentences, max_words)
        word_embeds = self.embedding(documents)  # (batch_size*max_sentences, max_words, embedding_dim)
        sentence_vectors = self.word_attention(word_embeds)  # (batch_size*max_sentences, word_hidden_dim*2)
        # Reshape back to batch and sentence dimensions
        sentence_vectors = sentence_vectors.view(batch_size, max_sentences, -1)
        document_vector = self.sentence_attention(sentence_vectors)  # (batch_size, sentence_hidden_dim*2)
        return document_vector

if __name__ == "__main__":
    # Example usage:
    vocab_size = 5000
    embedding_dim = 100
    word_hidden_dim = 50
    sentence_hidden_dim = 50
    
    # Create the hierarchical attention model
    model = HierarchicalAttention(vocab_size, embedding_dim, word_hidden_dim, sentence_hidden_dim)
    
    # Example input: batch of 2 documents, each with 3 sentences and 4 words per sentence
    sample_input = torch.randint(0, vocab_size, (2, 3, 4))
    output = model(sample_input)
    print("Output document embeddings shape:", output.shape)  # Expected shape: (2, sentence_hidden_dim*2)
