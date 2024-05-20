class Tokenizer:
    def __init__(self, text, num_merges=40):
        self.text = text
        self.tokens = self.text.encode("Latin-1")
        self.tokens = list(map(int, self.tokens))
        ids = list(self.tokens)
        self.merges = {}
        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"merging {pair} into new token {idx}")
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
            print(self.vocab[idx])

    def get_stats(self, ids):
        counts = {}
        # Consecutive elements.
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        """
        In the list of ints (ids), replace all consecutive occurrences of pair with the new token idx.
        """
        newids = []
        i = 0
        while i < len(ids):
            # If we are not at the last position and the pair matches, replace it.
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("Latin-1")
        return text

    def encode(self, text):
        tokens = list(text.encode("Latin-1"))
        while True:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                # Nothing to merge.
                break
            else:
                idx = self.merges[pair]
                tokens = self.merge(tokens, pair, idx)
        return tokens
