import torch
import os
from tokenizer import Tokenizer


class ModelData:
    def __init__(self, file, min_pairs, num_heads, train_split_pct=0.9):
        encoding = "UTF-8"
        with open(file, "r", encoding=encoding) as f:
            self.text = f.read()
        self.tokenizer = Tokenizer(
            self.text,
            min_pairs=min_pairs / train_split_pct,
            num_heads=num_heads,
            encoding=encoding,
        )
        self.encode = self.tokenizer.encode
        self.decode = self.tokenizer.decode
        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        self.train_test_split(train_split_pct)
        self.vocab_size = len(self.tokenizer.vocab)

    def train_test_split(self, train_split_pct):
        train_n = int(train_split_pct * len(self.data))
        self.train = self.data[:train_n]
        self.val = self.data[train_n:]


if __name__ == "__main__":
    x = ModelData("nietzsche.txt", 0.9)
    print(x.data.shape)
    print(x.data.dtype)
    print(x.data[:1000])
