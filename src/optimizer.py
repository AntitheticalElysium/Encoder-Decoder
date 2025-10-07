import cupy as cp

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
        # Collect all params and grads once (avoid double iteration)
        params_and_grads = list(self._get_params_and_grads())
        
        # Calculate total gradient norm
        total_norm = 0
        for _, grad in params_and_grads:
            total_norm += cp.sum(grad**2)
        total_norm = cp.sqrt(total_norm)
        
        clip_coef = self.clip_value / (total_norm + 1e-6)
        
        # Apply clipping and update parameters
        if clip_coef < 1:
            # Clip all gradients at once
            for param, grad in params_and_grads:
                grad *= clip_coef
                param -= self.learning_rate * grad
        else:
            # No clipping needed, just update
            for param, grad in params_and_grads:
                param -= self.learning_rate * grad

    def zero_grad(self):
        # Zero gradients for all parameters
        for _, grad in self._get_params_and_grads():
            grad.fill(0)
        # Also zero GRU cell gradients (these are not yielded by _get_params_and_grads)
        for layer in self.layers:
            if hasattr(layer, 'gru_cell') and hasattr(layer.gru_cell, 'zero_grad'):
                layer.gru_cell.zero_grad()
            elif hasattr(layer, 'forward_gru') and hasattr(layer, 'backward_gru'):
                # Bidirectional layer
                if hasattr(layer.forward_gru.gru_cell, 'zero_grad'):
                    layer.forward_gru.gru_cell.zero_grad()
                if hasattr(layer.backward_gru.gru_cell, 'zero_grad'):
                    layer.backward_gru.gru_cell.zero_grad()