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
        # avoid double iteration
        params_and_grads = list(self._get_params_and_grads())
        
        # Calculate total gradient norm
        total_norm = 0
        for _, grad in params_and_grads:
            total_norm += cp.sum(grad**2)
        total_norm = cp.sqrt(total_norm)
        
        clip_coef = self.clip_value / (total_norm + 1e-6)
        
        # Apply clipping and update parameters
        if clip_coef < 1:
            for param, grad in params_and_grads:
                grad *= clip_coef
                param -= self.learning_rate * grad
        else:
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

class Adam(object):
    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, clip_value=5.0):
        self.layers = layers
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clip_value = clip_value
        
        # First moment (momentum)
        self.m = {}
        # Second moment (variance)
        self.v = {}  
        self.t = 0  
        
        # Initialize moment vectors
        for layer in self.layers:
            if hasattr(layer, 'params'):
                layer_id = id(layer)
                self.m[layer_id] = {}
                self.v[layer_id] = {}
                for key in layer.params:
                    param = getattr(layer, key)
                    self.m[layer_id][key] = cp.zeros_like(param)
                    self.v[layer_id][key] = cp.zeros_like(param)

    def _get_params_and_grads(self):
        for layer in self.layers:
            if hasattr(layer, 'params'):
                for key in layer.params:
                    param = getattr(layer, key)
                    grad = getattr(layer, 'd_' + key)
                    yield layer, key, param, grad

    def step(self):
        """Perform one optimization step using Adam update rule"""
        self.t += 1
        params_grads = list(self._get_params_and_grads())
        
        # Calculate total gradient norm for clipping
        total_norm = 0
        for _, _, _, grad in params_grads:
            total_norm += cp.sum(grad**2)
        total_norm = cp.sqrt(total_norm)
        
        clip_coef = min(1.0, self.clip_value / (total_norm + 1e-6))
        
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        
        for layer, key, param, grad in params_grads:
            layer_id = id(layer)
            
            if clip_coef < 1:
                grad = grad * clip_coef
            
            # Update biased first moment estimate (momentum)
            self.m[layer_id][key] = self.beta1 * self.m[layer_id][key] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate (variance)
            self.v[layer_id][key] = self.beta2 * self.v[layer_id][key] + (1 - self.beta2) * (grad ** 2)
            # Compute bias-corrected moment estimates
            m_hat = self.m[layer_id][key] / bias_correction1
            v_hat = self.v[layer_id][key] / bias_correction2
            
            param -= self.learning_rate * m_hat / (cp.sqrt(v_hat) + self.epsilon)

    def zero_grad(self):
        """Reset all gradients to zero"""
        for layer in self.layers:
            if hasattr(layer, 'params'):
                for key in layer.params:
                    grad = getattr(layer, 'd_' + key)
                    grad.fill(0)
        
        # Also zero GRU cell gradients
        for layer in self.layers:
            if hasattr(layer, 'gru_cell') and hasattr(layer.gru_cell, 'zero_grad'):
                layer.gru_cell.zero_grad()
            elif hasattr(layer, 'forward_gru') and hasattr(layer, 'backward_gru'):
                # Bidirectional layer
                if hasattr(layer.forward_gru.gru_cell, 'zero_grad'):
                    layer.forward_gru.gru_cell.zero_grad()
                if hasattr(layer.backward_gru.gru_cell, 'zero_grad'):
                    layer.backward_gru.gru_cell.zero_grad()