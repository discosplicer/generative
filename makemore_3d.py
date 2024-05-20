from tokenizer_3d import Tokenizer3D
import torch
import os
from model_3d import BigramModel3D

if not os.path.exists("token3d_nietzsche.pt"):
    with open("nietzsche.txt", "r", encoding="Latin-1") as f:
        text = f.read()
    tokenizer = Tokenizer3D(text, num_merges=500)
    torch.save(tokenizer, "token3d_nietzsche.pt")
else:
    tokenizer = torch.load("token3d_nietzsche.pt")
encode = tokenizer.encode
decode = tokenizer.decode
data = torch.tensor(tokenizer.tokens, dtype=torch.long)
vocab_size = tokenizer.vocab_size
word_len = tokenizer.max_word_len

# GPT Video starts here
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(760)
iters = 3000
eval_iters = 100
learning_rate = 6e-3
block_size = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


model = BigramModel3D(vocab_size)
model = model.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
batch_size = 32

# Training loop.
for iter in range(iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(
    tokenizer.decode(
        model.generate(
            torch.zeros((1, 1, word_len), dtype=torch.long, device=device),
            max_new_tokens=100,
            device=device,
        )[0].tolist()
    )
)
