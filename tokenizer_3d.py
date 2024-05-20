class Tokenizer3D:
    def __init__(self, text, num_merges=40):
        self.text = text
        self.chars = sorted(list(set(self.text)))

        self.str_to_int = {ch: (i + 1) for i, ch in enumerate(self.chars)}
        self.int_to_str = {(i + 1): ch for i, ch in enumerate(self.chars)}

        self.str_to_int[""] = 0
        self.int_to_str[0] = ""
        self.merges = {}
        tokens = self.encode(text)
        self.vocab = self.int_to_str
        for i in range(num_merges):
            stats = self.get_stats(tokens)
            print(max(stats, key=stats.get))
            top_pair = self.multi_encode(max(stats, key=stats.get))
            top_pair_token = [char for token in top_pair for char in token]
            tokens = self.merge(tokens, top_pair, top_pair_token)
            self.vocab[len(self.chars) + i] = self.decode(top_pair)
        self.tokens, self.max_word_len = self.pad_tokens(tokens)
        self.vocab_size = len(self.chars) + num_merges

    def encode(self, str_to_encode):
        return [[self.str_to_int[c]] for c in str_to_encode]

    def multi_encode(self, strs_to_encode):
        return [[self.str_to_int[c] for c in s] for s in strs_to_encode]

    def decode(self, ints_to_decode):
        return "".join(
            [
                "".join([self.vocab[i] for i in list_of_ints])
                for list_of_ints in ints_to_decode
            ]
        )

    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            hashable_pair = tuple([self.decode([word]) for word in pair])
            counts[hashable_pair] = counts.get(hashable_pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            # if not at last position and pair matches, replace it.
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def pad_tokens(self, tokens):
        max_len = max([len(token) for token in tokens])
        for token in tokens:
            token += [0] * (max_len - len(token))
        return tokens, max_len
