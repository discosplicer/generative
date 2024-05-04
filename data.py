import torch


class ModelData:
    def __init__(self, file, train_split_pct):
        with open(file, "r", encoding="utf-8") as f:
            self.text = f.read()
        self.unique_chars()
        self.encoder_decoder()
        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        self.train_test_split(train_split_pct)

    def unique_chars(self):
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

    def encoder_decoder(self):
        str_to_int = {ch: i for i, ch in enumerate(self.chars)}
        int_to_str = {i: ch for i, ch in enumerate(self.chars)}
        # encoder takes a string and returns a list of ints
        self.encode = lambda s: [str_to_int[c] for c in s]
        # decoder takes a list of ints and returns a string
        self.decode = lambda l: "".join([int_to_str[i] for i in l])

    def train_test_split(self, train_split_pct):
        train_n = int(train_split_pct * len(self.data))
        self.train = self.data[:train_n]
        self.val = self.data[train_n:]


if __name__ == "__main__":
    x = ModelData("nietzsche.txt", 0.9)
    print(x.data.shape)
    print(x.data.dtype)
    print(x.data[:1000])
