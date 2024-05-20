import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(760)


class BigramModel3D(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets == None:
            loss = None
        else:
            B, T, W, C = logits.shape
            loss = 0
            for w in range(W):
                w_logits = logits[:, :, w, :].view(B * T, C)
                w_targets = targets[:, :, w].view(B * T)
                loss += F.cross_entropy(w_logits, w_targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            B, T, W, C = logits.shape
            idx_next = torch.zeros((1, 1, W), dtype=torch.long)
            for w in range(W):
                w_logits = logits[:, -1, w, :]
                probs = F.softmax(w_logits, dim=-1)
                w_next = torch.multinomial(probs, num_samples=1)
                idx_next[0, 0, w] = w_next
            idx = torch.cat((idx, idx_next), dim=2)
        return idx
