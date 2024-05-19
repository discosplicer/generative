import torch
import os
from tokenizer import Tokenizer


class ModelData:
    def __init__(self, file, train_split_pct):
        with open(file, "r", encoding="Latin-1") as f:
            self.text = f.read()
        self.tokenizer = Tokenizer(self.text, num_merges=10)
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
