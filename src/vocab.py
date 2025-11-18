import json
from collections import Counter, defaultdict


class Vocab(object):
    def __init__(self, special_tokens=["<pad>", "<unk>", "<sos>", "<eos>"]):
        self.special_tokens = special_tokens
        self.itos = list(special_tokens)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

        # BPE
        self.bpe_merges = []
        self.bpe_ranks = {}  # For O(1) lookup: pair -> rank
        self.use_bpe = False
        self.bpe_cache = {}  # Cache BPE splits for common words

    def train_bpe(self, tokenized_sentences, num_merges):
        word_freqs = Counter()
        for sent in tokenized_sentences:
            for word in sent:
                word_freqs[tuple(list(word) + ["</w>"])] += 1

        for merge_idx in range(num_merges):
            if merge_idx % 100 == 0:
                print(f"  BPE merge {merge_idx}/{num_merges}...")

            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=lambda x: pairs[x])
            # Merge this pair in all words
            new_word_freqs = {}
            bigram = "".join(best_pair)

            for word, freq in word_freqs.items():
                new_word = []
                i = 0
                while i < len(word):
                    # Check if we can merge at position i
                    if (
                        i < len(word) - 1
                        and word[i] == best_pair[0]
                        and word[i + 1] == best_pair[1]
                    ):
                        new_word.append(bigram)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1

                new_word_freqs[tuple(new_word)] = freq

            word_freqs = new_word_freqs
            self.bpe_merges.append(best_pair)

        # Build rank mapping for efficient lookup
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.bpe_merges)}
        print(f"  BPE training complete. Learned {len(self.bpe_merges)} merges.")
        self.use_bpe = True

    def _apply_bpe(self, word):
        if not self.use_bpe or not word:
            return [word]

        # Check cache first
        if word in self.bpe_cache:
            return self.bpe_cache[word]

        word_tokens = tuple(list(word) + ["</w>"])

        # If already a single token, return immediately
        if len(word_tokens) == 1:
            result = [word_tokens[0]]
            self.bpe_cache[word] = result
            return result

        # Iteratively merge the pair with lowest rank until no more merges possible
        while True:
            # Find all pairs and their ranks
            pairs = []
            for i in range(len(word_tokens) - 1):
                pair = (word_tokens[i], word_tokens[i + 1])
                if pair in self.bpe_ranks:
                    pairs.append((self.bpe_ranks[pair], i, pair))

            # If no valid pairs, we're done
            if not pairs:
                break

            # Get the pair with the lowest rank (learned earliest = most frequent)
            _, merge_pos, best_pair = min(pairs)

            # Merge this pair
            new_tokens = []
            i = 0
            while i < len(word_tokens):
                if i == merge_pos:
                    new_tokens.append("".join(best_pair))
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1

            word_tokens = tuple(new_tokens)

            # If reduced to single token, we're done
            if len(word_tokens) == 1:
                break

        result = list(word_tokens)
        self.bpe_cache[word] = result
        return result

    def _merge_bpe(self, tokens):
        if not self.use_bpe:
            return tokens

        words = []
        current_word = ""

        for token in tokens:
            if token.endswith("</w>"):
                current_word += token.replace("</w>", "")
                words.append(current_word)
                current_word = ""
            else:
                current_word += token

        if current_word:
            words.append(current_word)
        return words

    def build_vocab(self, tokenized_sentences, min_freq=1):
        if self.use_bpe:
            print("  Building vocab from BPE tokens...")
            # Use Counter directly for efficiency
            counter = Counter()
            total_sents = len(tokenized_sentences)

            for i, sent in enumerate(tokenized_sentences):
                if i % 5000 == 0 and i > 0:
                    print(
                        f"    Processed {i}/{total_sents} sentences... (vocab size so far: {len(counter)})"
                    )
                for word in sent:
                    subwords = self._apply_bpe(word)
                    for subword in subwords:
                        counter[subword] += 1

            print(f"    Total unique BPE tokens found: {len(counter)}")
        else:
            counter = Counter(tok for sent in tokenized_sentences for tok in sent)

        # Add tokens to vocabulary
        added = 0
        for tok, freq in counter.most_common():
            if freq < min_freq:
                continue
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
                added += 1

        print(
            f"  Vocabulary built with {len(self.itos)} tokens ({added} new tokens added)."
        )

    def encode(self, tokens):
        if self.use_bpe:
            subwords = []
            for token in tokens:
                subwords.extend(self._apply_bpe(token))
            return [self.stoi.get(tok, self.stoi["<unk>"]) for tok in subwords]
        else:
            return [self.stoi.get(tok, self.stoi["<unk>"]) for tok in tokens]

    def decode(self, idxs, skip_special_tokens=True):
        tokens = [self.itos[idx] for idx in idxs]
        if skip_special_tokens:
            tokens = [tok for tok in tokens if tok not in self.special_tokens]

        if self.use_bpe:
            tokens = self._merge_bpe(tokens)
        return tokens

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "itos": self.itos,
                    "bpe_merges": self.bpe_merges,
                    "use_bpe": self.use_bpe,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls()
        vocab.itos = data["itos"]
        vocab.stoi = {tok: idx for idx, tok in enumerate(vocab.itos)}
        vocab.bpe_merges = [tuple(pair) for pair in data.get("bpe_merges", [])]
        vocab.use_bpe = data.get("use_bpe", False)
        # Rebuild bpe_ranks for efficient lookup
        vocab.bpe_ranks = {pair: i for i, pair in enumerate(vocab.bpe_merges)}
        vocab.bpe_cache = {}
        return vocab
