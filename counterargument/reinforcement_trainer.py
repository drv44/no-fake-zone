import torch
import torch.optim as optim
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.distributions import Categorical

class ReinforcementTrainer:
    def __init__(self, model_name="t5-small", learning_rate=1e-5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

    def reward_function(self, generated_text, claim):
        # Define a reward based on factual accuracy, diversity, or similarity to ground truth
        return len(set(generated_text.split())) / max(1, len(generated_text.split()))

    def train_step(self, claim):
        self.model.train()
        
        input_text = f"counterargument: {claim}"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        
        outputs = self.model(input_ids, labels=input_ids)
        loss = outputs.loss
        
        # RL reward computation
        generated_text = self.tokenizer.decode(self.model.generate(input_ids)[0], skip_special_tokens=True)
        reward = self.reward_function(generated_text, claim)

        # Policy gradient step
        self.optimizer.zero_grad()
        (-loss * reward).backward()
        self.optimizer.step()

        return loss.item(), reward

# Example usage
if __name__ == "__main__":
    trainer = ReinforcementTrainer()
    claim = "Climate change is a hoax."
    loss, reward = trainer.train_step(claim)
    print(f"Training Loss: {loss}, Reward: {reward}")
