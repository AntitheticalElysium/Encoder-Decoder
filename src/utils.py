import cupy as np
import numpy as host_np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) # For numerical stability
    return e_x / e_x.sum(axis=-1, keepdims=True)

def xavier_init(input_dim, output_dim):
    limit = np.sqrt(6.0 / (input_dim + output_dim))
    return np.random.uniform(-limit, limit, (input_dim, output_dim)).astype(np.float32)

class CrossEntropyLoss(object):
    def __init__(self):
        self.probs = None
        self.targets_flat = None
        self.mask = None
        self.original_shape = None
        self.vocab_size = None

    def forward(self, predictions, targets, pad_idx):
        self.original_shape = predictions.shape
        batch_size, _, self.vocab_size = predictions.shape
        
        preds_flat = predictions.reshape(-1, self.vocab_size)
        self.targets_flat = targets.flatten()
        # mask to ignore padding tokens in the loss calculation
        self.mask = (self.targets_flat != pad_idx)
        
        masked_preds = preds_flat[self.mask]
        masked_targets = self.targets_flat[self.mask]
        
        self.probs = softmax(masked_preds)
        
        num_masked_samples = len(masked_targets)
        correct_logprobs = -np.log(self.probs[np.arange(num_masked_samples), masked_targets] + 1e-9)
        
        loss = np.sum(correct_logprobs) / batch_size
        return loss

    def backward(self):
        # create one-hot encoded targets for the non-padded tokens
        num_masked_samples = self.probs.shape[0]
        masked_targets = self.targets_flat[self.mask]
        
        one_hot_targets = np.zeros_like(self.probs)
        one_hot_targets[np.arange(num_masked_samples), masked_targets] = 1
        
        d_masked_logits = self.probs - one_hot_targets
        d_logits_flat = np.zeros((self.targets_flat.shape[0], self.vocab_size))
        d_logits_flat[self.mask] = d_masked_logits
        
        return d_logits_flat.reshape(self.original_shape)

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



