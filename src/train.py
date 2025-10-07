import cupy as cp
import numpy as np
import os
import random 
import pickle
from src.vocab import Vocab
from src.utils import CrossEntropyLoss, get_batch, model_to_cpu, model_to_gpu
from src.models import Seq2Seq
from src.optimizer import SGD


if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)

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

    # Create Seq2Seq model
    model = Seq2Seq.create(vocab_size_src, vocab_size_tgt, embed_dim, hidden_dim, num_layers=2)
    criterion = CrossEntropyLoss()

    # Get all learnable parameters
    all_learnable_layers = model.get_all_params()
    optimizer = SGD(layers=all_learnable_layers, learning_rate=learning_rate, clip_value=clip_value)

    best_loss = float('inf')
    print('Starting training on GPU...')
    for i in range(num_iterations):
        input_cpu, dec_in_cpu, dec_tgt_cpu = get_batch(src_ids_list, tgt_input_ids_list, tgt_output_ids_list, batch_size, pad_idx)
        
        input_seq = np.asarray(input_cpu)
        decoder_src = np.asarray(dec_in_cpu)
        decoder_tgt = np.asarray(dec_tgt_cpu)
        
        optimizer.zero_grad()

        # Forward pass through the model
        logits = model.forward(input_seq, decoder_src)
        loss_gpu = criterion.forward(logits, decoder_tgt, pad_idx)

        loss_cpu = loss_gpu.get()

        if i % 100 == 0:
            print(f'Iteration {i}/{num_iterations}, Loss: {loss_cpu:.4f}')
            if loss_cpu < best_loss:
                best_loss = loss_cpu
                print(f'  -> New best loss. Saving model.')
                
                model_to_cpu(all_learnable_layers)
                
                os.makedirs('models', exist_ok=True)
                with open('models/seq2seq.pkl', 'wb') as f:
                    pickle.dump(model, f)
                
                model_to_gpu(all_learnable_layers)

        # Backward pass
        d_logits = criterion.backward()
        model.backward(d_logits)

        optimizer.step()

    print('Training complete.')

    print('Saving final model...')
    model_to_cpu(all_learnable_layers)

    os.makedirs('models', exist_ok=True)
    with open('models/seq2seq.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('Model saved to models/seq2seq.pkl')
