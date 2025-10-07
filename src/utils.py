import cupy as np

def sigmoid(x):
    # Numerically stable sigmoid
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) # For numerical stability
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
        correct_logprobs = -np.log(self.probs[np.arange(num_masked_samples), masked_targets] + 1e-9)
        
        loss = np.sum(correct_logprobs) / batch_size
        return loss

    def backward(self):
        # create one-hot encoded targets for the non-padded tokens
        num_masked_samples = self.probs.shape[0]
        masked_targets = self.targets_flat[self.mask]
        batch_size = self.original_shape[0]
        
        one_hot_targets = np.zeros_like(self.probs)
        one_hot_targets[np.arange(num_masked_samples), masked_targets] = 1
        
        d_masked_logits = (self.probs - one_hot_targets) / batch_size
        d_logits_flat = np.zeros((self.targets_flat.shape[0], self.vocab_size))
        d_logits_flat[self.mask] = d_masked_logits
        
        return d_logits_flat.reshape(self.original_shape)
