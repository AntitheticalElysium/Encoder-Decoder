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
    
    # Extract context vector from bidirectional encoder (same as training)
    hidden_dim = encoder.layers[-1].hidden_dim
    last_fwd = encoder_hidden_states[:, -1, :hidden_dim]
    first_bwd = encoder_hidden_states[:, 0, hidden_dim:]
    context_vector = np.concatenate([last_fwd, first_bwd], axis=-1)

    translated_token_ids = []
    # Start with <sos> token
    generated_ids = [vocab_tgt.stoi['<sos>']]

    for _ in range(max_len):
        # Pass the full sequence generated so far
        decoder_input = np.array(generated_ids).reshape(1, -1)
        logits = decoder.forward(decoder_input, context_vector)
        
        # Get the last time step's logits
        predicted_id = np.argmax(logits[:, -1, :], axis=-1).item()

        if predicted_id == vocab_tgt.stoi['<eos>']:
            break
            
        translated_token_ids.append(predicted_id)
        generated_ids.append(predicted_id)

    translated_sentence = " ".join(vocab_tgt.decode(translated_token_ids, skip_special_tokens=True))
    return translated_sentence


if __name__ == '__main__':
    print("Loading models and vocabularies")
    with open('models/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('models/decoder.pkl', 'rb') as f:
        decoder = pickle.load(f)
        
    all_learnable_layers = [
        encoder.embedding,
        decoder.embedding,
        decoder.fc
    ]
    for layer in encoder.layers:
        all_learnable_layers.append(layer.forward_gru.gru_cell)
        all_learnable_layers.append(layer.backward_gru.gru_cell)
    for layer in decoder.layers:
        all_learnable_layers.append(layer.gru_cell)
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

    print("\nStarting Translation")
    for phrase in test_phrases:
        translation = translate_sentence(phrase, encoder, decoder, vocab_src, vocab_tgt)
        print(f"English Input:  {phrase}")
        print(f"French Output:  {translation}\n")
