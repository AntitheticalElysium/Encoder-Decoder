import cupy as cp
import pickle
from src.utils import model_to_gpu
from src.vocab import Vocab


if __name__ == '__main__':
    print("Loading model and vocabularies")
    
    with open('models/seq2seq.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Loaded Seq2Seq model from models/seq2seq.pkl")
        
    # Move model to GPU
    all_learnable_layers = model.get_all_params()
    model_to_gpu(all_learnable_layers)

    vocab_src = Vocab.load('data/vocab_src.json')
    vocab_tgt = Vocab.load('data/vocab_tgt.json')

    test_phrases = [
        "the cat is blue .",
        "i am a student .",
        "this is a big house .",
        "we are driving a car .",
        "she is very smart .",
        "the car is red and gold .",
        "i really like eating cucumbers .",
        "i played with my friends yesterday ."
    ]

    print("\nStarting Translation")
    print("=" * 50)
    for phrase in test_phrases:
        translation = model.predict(phrase, vocab_src, vocab_tgt)
        print(f"EN: {phrase}")
        print(f"FR: {translation}\n")
