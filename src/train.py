import cupy as np
import numpy as host_np
import os
import random 
import pickle
from src.vocab import Vocab
from src.utils import sigmoid, softmax, CrossEntropyLoss 
from src.models.rnn_seq2seq import Encoder, Decoder, Embedding, GRUCell, GRULayer, LinearLayer

class SGD(object):
    def __init__(self, layers, learning_rate=0.01, clip_value=5.0):
        self.layers = layers
        self.learning_rate = learning_rate
        self.clip_value = clip_value

    def _get_params_and_grads(self):
        for layer in self.layers:
            if hasattr(layer, 'params'):
                for key in layer.params:
                    param = getattr(layer, key)
                    grad = getattr(layer, 'd_' + key)
                    yield param, grad

    def step(self):
        total_norm = 0
        for _, grad in self._get_params_and_grads():
            total_norm += np.sum(grad**2)
        total_norm = np.sqrt(total_norm)
        
        clip_coef = self.clip_value / (total_norm + 1e-6)

        for param, grad in self._get_params_and_grads():
            if clip_coef < 1:
                grad *= clip_coef
            param -= self.learning_rate * grad

    def zero_grad(self):
        for _, grad in self._get_params_and_grads():
            grad.fill(0)

def get_batch(src_ids_list, tgt_input_ids_list, tgt_output_ids_list, batch_size, pad_idx):
    num_samples = len(src_ids_list)
    indices = host_np.random.choice(num_samples, batch_size, replace=False)

    batch_src = [src_ids_list[i] for i in indices]
    batch_tgt_in = [tgt_input_ids_list[i] for i in indices]
    batch_tgt_out = [tgt_output_ids_list[i] for i in indices]

    max_len_src = max(len(s) for s in batch_src)
    max_len_tgt = max(len(t) for t in batch_tgt_in)

    padded_src = host_np.full((batch_size, max_len_src), pad_idx, dtype=host_np.int32)
    padded_tgt_in = host_np.full((batch_size, max_len_tgt), pad_idx, dtype=host_np.int32)
    padded_tgt_out = host_np.full((batch_size, max_len_tgt), pad_idx, dtype=host_np.int32)

    for i in range(batch_size):
        padded_src[i, :len(batch_src[i])] = batch_src[i]
        padded_tgt_in[i, :len(batch_tgt_in[i])] = batch_tgt_in[i]
        padded_tgt_out[i, :len(batch_tgt_out[i])] = batch_tgt_out[i]

    return padded_src, padded_tgt_in, padded_tgt_out

def model_to_cpu(model_layers):
    for layer in model_layers:
        if hasattr(layer, 'params'):
            for param_name in layer.params:
                gpu_param = getattr(layer, param_name)
                setattr(layer, param_name, gpu_param.get())
                
                grad_name = 'd_' + param_name
                gpu_grad = getattr(layer, grad_name)
                setattr(layer, grad_name, gpu_grad.get())


def model_to_gpu(model_layers):
    for layer in model_layers:
        if hasattr(layer, 'params'):
            for param_name in layer.params:
                cpu_param = getattr(layer, param_name)
                setattr(layer, param_name, np.asarray(cpu_param))

                grad_name = 'd_' + param_name
                cpu_grad = getattr(layer, grad_name)
                setattr(layer, grad_name, np.asarray(cpu_grad))


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
    learning_rate = 0.005
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
    optimizer = SGD(layers=all_learnable_layers, learning_rate=learning_rate, clip_value=clip_value)

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
