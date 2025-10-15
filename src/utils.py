import cupy as cp
import numpy as np

def sigmoid(x):
    # Numerically stable sigmoid
    return cp.where(
        x >= 0,
        1 / (1 + cp.exp(-x)),
        cp.exp(x) / (1 + cp.exp(x))
    )

def softmax(x):
    e_x = cp.exp(x - cp.max(x, axis=-1, keepdims=True)) # For numerical stability
    return e_x / e_x.sum(axis=-1, keepdims=True)

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
        correct_logprobs = -cp.log(self.probs[cp.arange(num_masked_samples), masked_targets] + 1e-9)
        
        # Divide by number of non-padded tokens, not batch_size
        loss = cp.sum(correct_logprobs) / num_masked_samples
        return loss

    def backward(self):
        # create one-hot encoded targets for the non-padded tokens
        num_masked_samples = self.probs.shape[0]
        masked_targets = self.targets_flat[self.mask]
        
        one_hot_targets = cp.zeros_like(self.probs)
        one_hot_targets[cp.arange(num_masked_samples), masked_targets] = 1
        
        # Divide by number of non-padded tokens to match forward pass
        d_masked_logits = (self.probs - one_hot_targets) / num_masked_samples
        d_logits_flat = cp.zeros((self.targets_flat.shape[0], self.vocab_size))
        d_logits_flat[self.mask] = d_masked_logits
        
        return d_logits_flat.reshape(self.original_shape)

def get_batch(src_ids_list, tgt_input_ids_list, tgt_output_ids_list, batch_size, pad_idx):
    num_samples = len(src_ids_list)
    indices = np.random.choice(num_samples, batch_size, replace=False)

    batch_src = [src_ids_list[i] for i in indices]
    batch_tgt_in = [tgt_input_ids_list[i] for i in indices]
    batch_tgt_out = [tgt_output_ids_list[i] for i in indices]

    max_len_src = max(len(s) for s in batch_src)
    max_len_tgt = max(len(t) for t in batch_tgt_in)

    padded_src = np.full((batch_size, max_len_src), pad_idx, dtype=np.int32)
    padded_tgt_in = np.full((batch_size, max_len_tgt), pad_idx, dtype=np.int32)
    padded_tgt_out = np.full((batch_size, max_len_tgt), pad_idx, dtype=np.int32)

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
                setattr(layer, param_name, cp.asarray(cpu_param))

                grad_name = 'd_' + param_name
                cpu_grad = getattr(layer, grad_name)
                setattr(layer, grad_name, cp.asarray(cpu_grad))