import cupy as cp
import numpy as np
import os
import random 
import pickle
from src.vocab import Vocab
from src.utils import CrossEntropyLoss, get_batch, model_to_cpu, model_to_gpu
from src.models import Seq2Seq
from src.optimizer import SGD,Adam


if __name__ == '__main__':
    # Clear GPU memory... 16G is rough....
    cp.get_default_memory_pool().free_all_blocks()
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)

    with open('data/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    all_indices = list(range(len(data['src_ids'])))
    random.shuffle(all_indices)
    sample_indices = all_indices[:]

    src_ids_list = [data['src_ids'][i] for i in sample_indices]
    tgt_input_ids_list = [data['tgt_input_ids'][i] for i in sample_indices]
    tgt_output_ids_list = [data['tgt_output_ids'][i] for i in sample_indices]
    vocab_src = data['vocab_src']
    vocab_tgt = data['vocab_tgt']

    print(f'Loaded {len(src_ids_list)} training samples.')

    pad_idx = vocab_tgt.stoi['<pad>']
    vocab_size_src = len(vocab_src)
    vocab_size_tgt = len(vocab_tgt)

    embed_dim = 512
    hidden_dim = 512
    attention_dim = 512
    num_layers = 2
    batch_size = 32
    num_iterations = 5000
    learning_rate = 0.001
    clip_value = 5.0

    # Create Seq2Seq model
    model = Seq2Seq.create(vocab_size_src, vocab_size_tgt, embed_dim, hidden_dim, num_layers, attention_dim)
    criterion = CrossEntropyLoss()

    # Get all learnable parameters
    all_learnable_layers = model.get_all_params()
    optimizer = Adam(layers=all_learnable_layers, learning_rate=learning_rate, clip_value=clip_value)

    best_loss = float('inf')
    print('Starting training on GPU...')
    print(f'Config: embed_dim={embed_dim}, hidden_dim={hidden_dim}, attention_dim={attention_dim}, batch_size={batch_size}')
    print(f'        optimizer=Adam, lr={learning_rate}, clip={clip_value}, iterations={num_iterations}')
    print(f'        vocab_src={vocab_size_src}, vocab_tgt={vocab_size_tgt}')
    for i in range(num_iterations):
        # Simple learning rate decay
        if i > 1000:  # Warm-up period
            current_lr = learning_rate * (0.95 ** ((i - 1000) // 100))
        else:
            current_lr = learning_rate

        if optimizer.learning_rate != current_lr:
            optimizer.learning_rate = current_lr
            print(f'  -> Learning rate changed to {current_lr}')
        input_cpu, dec_in_cpu, dec_tgt_cpu = get_batch(src_ids_list, tgt_input_ids_list, tgt_output_ids_list, batch_size, pad_idx)
        
        # Convert to CuPy arrays
        input_seq = cp.asarray(input_cpu)
        decoder_src = cp.asarray(dec_in_cpu)
        decoder_tgt = cp.asarray(dec_tgt_cpu)
        
        optimizer.zero_grad()

        # Forward pass through the model
        logits = model.forward(input_seq, decoder_src)
        loss_gpu = criterion.forward(logits, decoder_tgt, pad_idx)

        loss_cpu = loss_gpu.get()
        
        # Backward pass
        d_logits = criterion.backward()
        model.backward(d_logits)

        if i % 100 == 0:
            # Calculate gradient norm for monitoring
            grad_norm = 0
            for layer in all_learnable_layers:
                if hasattr(layer, 'params'):
                    for key in layer.params:
                        grad = getattr(layer, 'd_' + key)
                        grad_norm += cp.sum(grad**2)
            grad_norm = float(cp.sqrt(grad_norm).get())
            
            print(f'Iteration {i}/{num_iterations}, Loss: {loss_cpu:.4f}, Grad Norm: {grad_norm:.4f}')
            
            # Only save every 1000 iterations to reduce I/O overhead
            if loss_cpu < best_loss and i % 1000 == 0:
                best_loss = loss_cpu
                print(f'  -> New best loss. Saving model.')
                
                model_to_cpu(all_learnable_layers)
                
                os.makedirs('models', exist_ok=True)
                with open('models/seq2seq.pkl', 'wb') as f:
                    pickle.dump(model, f)
                
                model_to_gpu(all_learnable_layers)

        optimizer.step()

    print('Training complete.')

    print('Saving final model...')
    model_to_cpu(all_learnable_layers)

    os.makedirs('models', exist_ok=True)
    with open('models/seq2seq.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('Model saved to models/seq2seq.pkl')
