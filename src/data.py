import os
import zipfile
import pickle
from d2l import torch as d2l
from vocab import Vocab

def download_and_save_raw_data(root='../data', filename='raw.txt'):
    os.makedirs(root, exist_ok=True)
    
    zip_path = d2l.download('fra-eng', root)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root)
    
    fra_txt_path = os.path.join(root, 'fra-eng', 'fra.txt')
    raw_path = os.path.join(root, filename)
    
    with open(fra_txt_path, 'r', encoding='utf-8') as f_in, \
         open(raw_path, 'w', encoding='utf-8') as f_out:
        f_out.write(f_in.read())
    
    print(f'Raw data saved to {raw_path}')
    print('\nFirst 5 lines of the raw data:')
    with open(raw_path, 'r', encoding='utf-8') as f:
        for _ in range(5):
            print(f.readline().strip())

def normalize_corpus(raw_path='../data/raw.txt', norm_path='../data/norm.txt'):
    with open(raw_path, 'r', encoding='utf-8') as f_in, \
         open(norm_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            eng_raw, fra_raw = line.strip().split('\t')
            # convert to lowercase
            eng, fra = eng_raw.lower(), fra_raw.lower()
            # add spaces before punctuation
            for symbol in '.,!?':
                eng = eng.replace(symbol, f' {symbol} ')
                fra = fra.replace(symbol, f' {symbol} ')
            # remove non printable characters
            eng = ''.join(char for char in eng if char.isprintable())
            fra = ''.join(char for char in fra if char.isprintable())
            # remove extra spaces
            eng = ' '.join(eng.split())
            fra = ' '.join(fra.split())

            f_out.write(f'{eng}\t{fra}\n')
    
    print(f'Normalized data saved to {norm_path}')
    print('\nFirst 5 lines of the normalized data:')
    with open(norm_path, 'r', encoding='utf-8') as f:
        for _ in range(5):
            print(f.readline().strip())

def tokenize_corpus(norm_path='../data/norm.txt'):
    with open(norm_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    tokenized_pairs = []
    for line in lines:
        try:
            eng, fra = line.strip().split('\t')
        except ValueError:
            continue
        eng_tok = [t for t in eng.split(' ') if t]
        fra_tok = [t for t in fra.split(' ') if t]
        tokenized_pairs.append((eng_tok, fra_tok))

    return tokenized_pairs

def filter_pairs(tokenized_pairs, max_len=15):
    filtered_pairs = []
    for eng_tok, fra_tok in tokenized_pairs:
        if len(eng_tok) <= max_len and len(fra_tok) <= max_len:
            filtered_pairs.append((eng_tok, fra_tok))

    return filtered_pairs

def build_vocabulary(filtered_pairs, min_freq=1):
    eng_sentences = [eng for eng, fra in filtered_pairs]
    fra_sentences = [fra for eng, fra in filtered_pairs]

    vocab_src = Vocab() # English
    vocab_tgt = Vocab() # French

    vocab_src.build_vocab(eng_sentences, min_freq)
    vocab_tgt.build_vocab(fra_sentences, min_freq)

    print(f'English vocabulary size: {len(vocab_src)}')
    print(f'French vocabulary size: {len(vocab_tgt)}')

    print('\nSample English tokens and their indices:')
    for token in list(vocab_src.stoi.keys())[:10]:
        print(f'{token}: {vocab_src.stoi[token]}')
    print('\nSample French tokens and their indices:')
    for token in list(vocab_tgt.stoi.keys())[:10]:
        print(f'{token}: {vocab_tgt.stoi[token]}')

    print('\nSaving vocabularies to ../data/vocab_src.json and ../data/vocab_tgt.json')
    vocab_src.save('../data/vocab_src.json')
    vocab_tgt.save('../data/vocab_tgt.json')
    return vocab_src, vocab_tgt

def convert_to_ids(filtered_pairs, vocab_src, vocab_tgt):
    src_ids_list = []
    tgt_input_ids_list = []
    tgt_output_ids_list = []

    for eng_tok, fra_tok in filtered_pairs:
        # encoder input: source IDs + <eos>
        src_ids = vocab_src.encode(eng_tok) + [vocab_src.stoi['<eos>']]
        src_ids_list.append(src_ids)
        # decoder input: <sos> + target IDs
        tgt_input_ids = [vocab_tgt.stoi['<sos>']] + vocab_tgt.encode(fra_tok)
        tgt_input_ids_list.append(tgt_input_ids)
        # decoder output: target IDs + <eos>
        tgt_output_ids = vocab_tgt.encode(fra_tok) + [vocab_tgt.stoi['<eos>']]
        tgt_output_ids_list.append(tgt_output_ids)

    return src_ids_list, tgt_input_ids_list, tgt_output_ids_list

def save_preprocessed_data(src_ids_list, tgt_input_ids_list, tgt_output_ids_list,
                           vocab_src, vocab_tgt, save_path='../data/preprocessed_data.pkl'):
    data = {
        'src_ids': src_ids_list,
        'tgt_input_ids': tgt_input_ids_list,
        'tgt_output_ids': tgt_output_ids_list,
        'vocab_src': vocab_src,
        'vocab_tgt': vocab_tgt
    }

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f'Preprocessed data saved to {save_path}')

def explore_data():
    pass

if __name__ == '__main__':
    #download_and_save_raw_data()
    #normalize_corpus()
    tokenized_pairs = tokenize_corpus()
    filtered_pairs = filter_pairs(tokenized_pairs)
    #build_vocabulary(filtered_pairs)
    vocab_src = Vocab.load('../data/vocab_src.json')
    vocab_tgt = Vocab.load('../data/vocab_tgt.json')
    src_ids_list, tgt_input_ids_list, tgt_output_ids_list = convert_to_ids(filtered_pairs, vocab_src, vocab_tgt)
    save_preprocessed_data(src_ids_list, tgt_input_ids_list, tgt_output_ids_list, vocab_src, vocab_tgt)
