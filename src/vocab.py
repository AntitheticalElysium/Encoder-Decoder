import json
from collections import Counter

class Vocab(object):
    def __init__(self, special_tokens=["<pad>", "<unk>", "<sos>", "<eos>"]):
        self.special_tokens = special_tokens 
        self.itos = list(special_tokens)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def build_vocab(self, tokenized_sentences, min_freq=1):
        counter = Counter(tok for sent in tokenized_sentences for tok in sent)
        for tok, freq in counter.most_common():
            if freq < min_freq:
                continue
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
        
    def encode(self, tokens):
        return [self.stoi.get(tok, self.stoi["<unk>"]) for tok in tokens]

    def decode(self, idxs, skip_special_tokens=True):
        tokens = [self.itos[idx] for idx in idxs]
        if skip_special_tokens:
            tokens = [tok for tok in tokens if tok not in self.special_tokens]
        return tokens

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'itos': self.itos}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vocab = cls()
        vocab.itos = data['itos']
        vocab.stoi = {tok: idx for idx, tok in enumerate(vocab.itos)}
        return vocab
