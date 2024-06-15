class Tokenizer:
    def __init__(self, text, min_pairs, num_heads, encoding):
        self.text = text
        self.encoding = encoding
        self.tokens = self.text.encode(encoding)
        self.tokens = list(map(int, self.tokens))
        ids = list(self.tokens)
        self.gate_tokens = list(map(int, ".?!".encode(encoding)))
        self.word_token = list(map(int, " \n".encode(encoding)))
        stats = self.get_stats(ids)
        occurrences = max(stats.values())
        self.merges = {}
        idx = 256
        i = 0
        # Go until you have no occurences with min required pairs
        # and the number of tokens evenly divides number of heads.
        while occurrences > min_pairs or idx % num_heads != 0:
            pair = max(stats, key=stats.get)
            print(f"merging {pair} occurring {occurrences} times into new token {idx}")
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
            stats = self.get_stats(ids)
            occurrences = max(stats.values())
            i += 1
            idx += 1

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def get_stats(self, ids):
        counts = {}
        # Consecutive elements.
        for pair in zip(ids, ids[1:]):
            if pair[0] not in [*self.gate_tokens, *self.word_token] and pair[1] not in [
                *self.gate_tokens,
                *self.word_token,
            ]:
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
        text = tokens.decode(self.encoding, errors="replace")
        return text

    def encode(self, text):
        tokens = list(text.encode(self.encoding))
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
