import cupy as np
import pickle
from src.train import model_to_gpu
from src.vocab import Vocab

def translate_sentence(sentence, encoder, decoder, vocab_src, vocab_tgt, max_len=10):
    sentence = sentence.lower()
    for symbol in '.,!?':
        sentence = sentence.replace(symbol, f' {symbol} ')
    
    tokens = [token for token in sentence.split(' ') if token]
    
    token_ids = vocab_src.encode(tokens)
    token_ids.append(vocab_src.stoi['<eos>'])
    
    src_tensor = np.array(token_ids).reshape(1, -1)

    encoder_hidden_states = encoder.forward(src_tensor)
    context_vector = encoder_hidden_states[:, -1, :]

    translated_token_ids = []
    decoder_input = np.array([vocab_tgt.stoi['<sos>']]).reshape(1, 1)
    decoder_hidden = context_vector

    for _ in range(max_len):
        embedded = decoder.embedding.forward(decoder_input)
        decoder_gru_outputs = decoder.gru.forward(embedded, decoder_hidden)
        logits = decoder.fc.forward(decoder_gru_outputs)
        
        predicted_id = np.argmax(logits, axis=-1).item()

        decoder_hidden = decoder_gru_outputs[:, -1, :]

        if predicted_id == vocab_tgt.stoi['<eos>']:
            break
            
        translated_token_ids.append(predicted_id)

        decoder_input = np.array([predicted_id]).reshape(1, 1)

    translated_sentence = " ".join(vocab_tgt.decode(translated_token_ids, skip_special_tokens=True))
    return translated_sentence


if __name__ == '__main__':
    print("--- Loading models and vocabularies ---")
    with open('models/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('models/decoder.pkl', 'rb') as f:
        decoder = pickle.load(f)
        
    all_learnable_layers = [
        encoder.embedding, 
        encoder.gru.forward_gru.gru_cell,
        encoder.gru.backward_gru.gru_cell,
        decoder.embedding, 
        decoder.gru.gru_cell, 
        decoder.fc
    ]
    model_to_gpu(all_learnable_layers)

    vocab_src = Vocab.load('data/vocab_src.json')
    vocab_tgt = Vocab.load('data/vocab_tgt.json')

    test_phrases = [
        "the cat is blue .",
        "i am a student .",
        "this is a big house .",
        "we are driving a car .",
        "she is very smart ."
    ]

    print("\n--- Starting Translation ---")
    for phrase in test_phrases:
        translation = translate_sentence(phrase, encoder, decoder, vocab_src, vocab_tgt)
        print(f"English Input:  {phrase}")
        print(f"French Output:  {translation}\n")
