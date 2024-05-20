from tokenizer_3d import Tokenizer3D
import torch
import os
from model_3d import BigramModel3D

if not os.path.exists("token3d_nietzsche.pt"):
    with open("nietzsche.txt", "r", encoding="Latin-1") as f:
        text = f.read()
    tokenizer = Tokenizer3D(text, num_merges=10)
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
batch_size = 4
block_size = 8


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch("train")

m = BigramModel3D(vocab_size)
out, loss = m(xb, yb)
print(vocab_size)
print(out.shape)
print(loss)
print(tokenizer.int_to_str)
print(
    tokenizer.decode(
        m.generate(torch.zeros((1, 1, word_len), dtype=torch.long), max_new_tokens=10)[
            0
        ].tolist()
    )
)
