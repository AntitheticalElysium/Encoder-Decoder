import cupy as np

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

class Adam(object):
    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, clip_value=5.0):
        self.layers = layers
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clip_value = clip_value

        self.t = 0 
        self.m = []
        self.v = []

        for layer in self.layers:
            if hasattr(layer, 'params'):
                for key in layer.params:
                    # 'param' is a tuple (weights, gradients)
                    param_tuple = layer.params[key] 
                    self.m.append(np.zeros_like(param_tuple[0]))
                    self.v.append(np.zeros_like(param_tuple[0]))

    def _get_params_and_grads(self):
        for layer in self.layers:
            if hasattr(layer, 'params'):
                for key in layer.params:
                    param_tuple = layer.params[key] 
                    yield param_tuple[0], param_tuple[1]

    def step(self):
        total_norm = 0
        for _, grad in self._get_params_and_grads():
            total_norm += np.sum(grad**2)
        total_norm = np.sqrt(total_norm)
        
        clip_coef = self.clip_value / (total_norm + 1e-6)

        if clip_coef < 1:
            for _, grad in self._get_params_and_grads():
                grad *= clip_coef

        self.t += 1
        param_index = 0
        for param, grad in self._get_params_and_grads():
            # Update biased first moment estimate
            self.m[param_index] = self.beta1 * self.m[param_index] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[param_index] = self.beta2 * self.v[param_index] + (1 - self.beta2) * (grad**2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[param_index] / (1 - self.beta1**self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[param_index] / (1 - self.beta2**self.t)
            
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            param_index += 1


    def zero_grad(self):
        for _, grad in self._get_params_and_grads():
            grad.fill(0)
