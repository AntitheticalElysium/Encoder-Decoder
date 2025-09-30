import numpy as np
import pickle
from src.vocab import Vocab
from src.utils import sigmoid, softmax, CrossEntropyLoss
from src.models.rnn_seq2seq import Encoder, Decoder, Embedding, GRUCell, GRULayer, LinearLayer

class SGD(object):
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate

    def step(self):
        for layer in self.layers:
            for key in layer.params:
                param, grad = layer.params[key]
                param -= self.learning_rate * grad

    def zero_grad(self):
        for layer in self.layers:
            for key in layer.params:
                param, grad = layer.params[key]
                # Reset the gradient to zero
                grad.fill(0)

def get_batch(src_ids_list, tgt_input_ids_list, tgt_output_ids_list, batch_size, pad_idx):
    num_samples = len(src_ids_list)
    indices = np.random.choice(num_samples, batch_size, replace=False)

    batch_src = [src_ids_list[i] for i in indices]
    batch_tgt_in = [tgt_input_ids_list[i] for i in indices]
    batch_tgt_out = [tgt_output_ids_list[i] for i in indices]

    max_len_src = max(len(s) for s in batch_src)
    max_len_tgt = max(len(t) for t in batch_tgt_in) # in and out have same length

    # Pad all sequences in the batch to the max length
    padded_src = np.full((batch_size, max_len_src), pad_idx, dtype=np.int32)
    padded_tgt_in = np.full((batch_size, max_len_tgt), pad_idx, dtype=np.int32)
    padded_tgt_out = np.full((batch_size, max_len_tgt), pad_idx, dtype=np.int32)

    for i in range(batch_size):
        padded_src[i, :len(batch_src[i])] = batch_src[i]
        padded_tgt_in[i, :len(batch_tgt_in[i])] = batch_tgt_in[i]
        padded_tgt_out[i, :len(batch_tgt_out[i])] = batch_tgt_out[i]

    return padded_src, padded_tgt_in, padded_tgt_out

if __name__ == '__main__':
    with open('data/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # Smaller dataset for quicker debugging
    src_ids_list = data['src_ids'][:1000]
    tgt_input_ids_list = data['tgt_input_ids'][:1000]
    tgt_output_ids_list = data['tgt_output_ids'][:1000]
    vocab_src = data['vocab_src']
    vocab_tgt = data['vocab_tgt']

    print(f'Loaded {len(src_ids_list)} training samples.')

    pad_idx = vocab_tgt.stoi['<pad>']
    vocab_size_src = len(vocab_src)
    vocab_size_tgt = len(vocab_tgt)

    # Hyperparams
    embed_dim = 64
    hidden_dim = 128
    batch_size = 32 
    num_epochs = 1000
    learning_rate = 0.01

    encoder = Encoder(vocab_size_src, embed_dim, hidden_dim)
    decoder = Decoder(vocab_size_tgt, embed_dim, hidden_dim)
    criterion = CrossEntropyLoss()

    learnable_layers = [encoder.embedding, encoder.gru.gru_cell, decoder.embedding, decoder.gru.gru_cell, decoder.fc]
    optimizer = SGD(layers=learnable_layers, learning_rate=learning_rate)
    
    print('Starting training...')
    for i in range(num_epochs):
        input_seq, decoder_src, decoder_tgt = get_batch(src_ids_list, tgt_input_ids_list, tgt_output_ids_list, batch_size, pad_idx)
        # Grads to zero
        optimizer.zero_grad()

        # Forwards
        encoder_hidden = encoder.forward(input_seq)
        context_vect = encoder_hidden[:, -1, :]  # Use the last hidden state as context 
        
        logits = decoder.forward(decoder_src, context_vect)
        loss = criterion.forward(logits, decoder_tgt, pad_idx)

        if (i % 50 == 0):
            print(f'Epoch {i}, Loss: {loss:.4f}')

        # Backwards
        d_logits = criterion.backward()
        d_context_vect = decoder.backward(d_logits)
        encoder.backward(d_context_vect)

        # Update params
        optimizer.step()

    print('Training complete.')

    with open('models/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    with open('models/decoder.pkl', 'wb') as f:
        pickle.dump(decoder, f)
    print('Models saved to models/encoder.pkl and models/decoder.pkl')
