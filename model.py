import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import numpy as np

from data import ModelData

torch.manual_seed(760)


class BigramLanguageModel(nn.Module):

    def __init__(self, params: dict):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            params["vocab_size"], params["vocab_size"]
        )

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices
        for _ in range(max_new_tokens):
            # Get predictions.
            logits, loss = self(idx)
            # Only check the last time step. (reshape to B, C)
            logits = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the dist. (1 for each batch)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class AttentionHead(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.key = nn.Linear(params["n_embd"], params["head_size"], bias=False)
        self.query = nn.Linear(params["n_embd"], params["head_size"], bias=False)
        self.value = nn.Linear(params["n_embd"], params["head_size"], bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(params["block_size"], params["block_size"]))
        )
        self.dropout = nn.Dropout(params["dropout"])

    def forward(self, x, y=None):
        B, T, C = x.shape
        k = self.key(x)  # B, T, C
        q = self.query(x)  # B, T, C
        # Compute attention scores.
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # Weighted aggregation of values.
        v = self.value(x)  # B, T, C
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(params) for _ in range(params["num_heads"])]
        )
        self.proj = nn.Linear(params["n_embd"], params["n_embd"])
        self.dropout = nn.Dropout(params["dropout"])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Polynomial3(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.x1 = torch.nn.Parameter(torch.randn(()))
        self.x2 = torch.nn.Parameter(torch.randn(()))
        self.x3 = torch.nn.Parameter(torch.randn(()))
        self.x4 = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.x1 + self.x2 * x + self.x3 * x**2 + self.x4 * x**3


class FeedForward(nn.Module):
    """Linear layer followed by non-linear transformation."""

    def __init__(self, params: dict):
        super().__init__()
        # The inner layer of the feed-forward network should be 4x the size of the embeddings, according to AIAYN.
        self.net = nn.Sequential(
            nn.Linear(params["n_embd"], 4 * params["n_embd"]),
            Polynomial3(),
            nn.Linear(4 * params["n_embd"], params["n_embd"]),
            nn.Dropout(params["dropout"]),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Communcation followed by computation."""

    def __init__(self, params: dict):
        super().__init__()
        self.sa_heads = MultiHeadAttention(params)
        self.ln_sa = nn.LayerNorm(params["n_embd"])
        self.ffwd = FeedForward(params)
        self.ln_ff = nn.LayerNorm(params["n_embd"])

    def forward(self, x):
        # "x +" makes it residual.
        x = x + self.sa_heads(self.ln_sa(x))
        x = x + self.ffwd(self.ln_ff(x))
        return x


class SimpleLinear(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.block_size = model_params["block_size"]
        self.batch_size = model_params["batch_size"]
        self.device = model_params["device"]
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(
            model_params["vocab_size"], model_params["n_embd"]
        )
        self.position_embedding_table = nn.Embedding(
            self.block_size, model_params["n_embd"]
        )
        self.blocks = nn.Sequential(
            *[TransformerBlock(model_params) for _ in range(model_params["n_layer"])],
            nn.LayerNorm(model_params["n_embd"]),
        )
        # Each token reads off the logits for the next token in a lookup
        self.ln_f = nn.LayerNorm(model_params["n_embd"])  # final layer norm
        self.lm_head = nn.Linear(model_params["n_embd"], model_params["vocab_size"])

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx and targets are (B, T) tensors of ints
        B, T = idx.shape
        # Final Size is (Batch Size, Time [block size], Channels [vocab size])
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        # idx is (B, T) array of indices in the current context.
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx[:, -self.block_size :]
            # Get the predictions.
            logits, loss = self(idx_cond)
            # Only use the last time step.
            logits = logits[:, -1, :] / temperature  # (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class Model:
    def __init__(self, data: ModelData, model_type, model_params: dict):
        self.train_data = data.train
        self.val_data = data.val
        self.decode = data.decode
        self.model = model_type(model_params)
        self._unpack_model_params(model_params)
        self.model = self.model.to(self.device)

    def _unpack_model_params(self, params):
        self.block_size = params["block_size"]
        self.batch_size = params["batch_size"]
        self.lr = params["lr"]
        self.eval_iters = params["eval_iters"]
        self.device = params["device"]
        self.temperature = params["temperature"]

    def get_batch(self, split="train"):
        """
        Generate a small batch of data with inputs and targets
        """
        if split == "train":
            data = self.train_data
            ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        elif split == "val":
            data = self.val_data
            ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        # Set model to evaluation mode.
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        # Set the model back to training mode.
        self.model.train()
        return out

    def train(self, text, iters, max_tokens):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.95)
        )
        standard_train = self.train_data.detach().clone()
        for iter in range(iters):
            # Once in a while, evaluate the loss.
            if iter % self.eval_iters == 0:
                losses = self.estimate_loss()
                print(
                    f"step {iter}: train loss {(losses['train']):.4f}, val loss {losses['val']:.4f}"
                )
                print(self.decode(self.inference(max_tokens=max_tokens).tolist()))
            # Sample from data.
            xb, yb = self.get_batch("train")
            # Evaluate the loss
            logits, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # Reset the training data.
            self.train_data = standard_train.detach().clone()
        # Print the final results.
        losses = self.estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        print(self.decode(self.inference(max_tokens=max_tokens).tolist()))

    def inference(self, max_tokens, iter=None):
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        next_tokens = self.model.generate(
            context, max_new_tokens=max_tokens, temperature=self.temperature
        )
        return next_tokens[0]

    def context_target_demo(self):
        xb, yb = self.get_batch("train")
        print(xb.shape)
        print(yb.shape)
        # Batch dimension.
        for batch in range(self.batch_size):
            # Time dimension.
            for time in range(self.block_size):
                context = xb[batch, : time + 1]
                target = yb[batch, time]
                print(f"When input is {context} the target is {target}")
