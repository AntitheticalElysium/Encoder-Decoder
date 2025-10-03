import cupy as np
import numpy as host_np
import os
import random 
import pickle
from src.vocab import Vocab
from src.utils import sigmoid, softmax, CrossEntropyLoss 
from src.models.rnn_seq2seq import Encoder, Decoder, Embedding, GRUCell, GRULayer, LinearLayer
from src.optimizer import SGD, Adam
from src.utils import get_batch, model_to_cpu, model_to_gpu

if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    host_np.random.seed(seed)
    np.random.seed(seed)

    with open('data/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    all_indices = list(range(len(data['src_ids'])))
    random.shuffle(all_indices)
    sample_indices = all_indices[:50000]

    src_ids_list = [data['src_ids'][i] for i in sample_indices]
    tgt_input_ids_list = [data['tgt_input_ids'][i] for i in sample_indices]
    tgt_output_ids_list = [data['tgt_output_ids'][i] for i in sample_indices]
    vocab_src = data['vocab_src']
    vocab_tgt = data['vocab_tgt']

    print(f'Loaded {len(src_ids_list)} training samples.')

    pad_idx = vocab_tgt.stoi['<pad>']
    vocab_size_src = len(vocab_src)
    vocab_size_tgt = len(vocab_tgt)

    embed_dim = 256
    hidden_dim = 256
    batch_size = 64
    num_iterations = 3000
    learning_rate = 0.001
    clip_value = 1.0

    encoder = Encoder(vocab_size_src, embed_dim, hidden_dim, 2)
    decoder = Decoder(vocab_size_tgt, embed_dim, hidden_dim, 2)
    criterion = CrossEntropyLoss()

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
    #optimizer = SGD(layers=all_learnable_layers, learning_rate=learning_rate, clip_value=clip_value)
    optimizer = Adam(layers=all_learnable_layers, learning_rate=learning_rate, clip_value=clip_value)

    best_loss = float('inf')
    print('Starting training on GPU...')
    for i in range(num_iterations):
        input_cpu, dec_in_cpu, dec_tgt_cpu = get_batch(src_ids_list, tgt_input_ids_list, tgt_output_ids_list, batch_size, pad_idx)
        
        input_seq = np.asarray(input_cpu)
        decoder_src = np.asarray(dec_in_cpu)
        decoder_tgt = np.asarray(dec_tgt_cpu)
        
        optimizer.zero_grad()

        encoder_hidden = encoder.forward(input_seq)

        # Use two states from bidirectional GRU as context vector
        last_fwd = encoder_hidden[:, -1, :hidden_dim]
        first_bwd = encoder_hidden[:, 0, hidden_dim:]
        context_vect = np.concatenate([last_fwd, first_bwd], axis=-1)
        
        logits = decoder.forward(decoder_src, context_vect)
        loss_gpu = criterion.forward(logits, decoder_tgt, pad_idx)

        loss_cpu = loss_gpu.get()

        if i % 100 == 0:
            print(f'Iteration {i}/{num_iterations}, Loss: {loss_cpu:.4f}')
            if loss_cpu < best_loss:
                best_loss = loss_cpu
                print(f'  -> New best loss. Saving model.')
                
                model_to_cpu(all_learnable_layers)
                
                os.makedirs('models', exist_ok=True)
                with open('models/encoder.pkl', 'wb') as f:
                    pickle.dump(encoder, f)
                with open('models/decoder.pkl', 'wb') as f:
                    pickle.dump(decoder, f)
                
                model_to_gpu(all_learnable_layers)

        d_logits = criterion.backward()
        d_context_vect = decoder.backward(d_logits)
        encoder.backward(d_context_vect)

        optimizer.step()

    print('Training complete.')

    print('Saving final model...')
    model_to_cpu(all_learnable_layers)

    os.makedirs('models', exist_ok=True)
    with open('models/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    with open('models/decoder.pkl', 'wb') as f:
        pickle.dump(decoder, f)
    print('Models saved to models/encoder.pkl and models/decoder.pkl')
