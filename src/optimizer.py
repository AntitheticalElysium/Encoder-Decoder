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
        total_norm = 0
        for _, grad in self._get_params_and_grads():
            total_norm += cp.sum(grad**2)
        total_norm = cp.sqrt(total_norm)
        
        clip_coef = self.clip_value / (total_norm + 1e-6)

        for param, grad in self._get_params_and_grads():
            if clip_coef < 1:
                grad *= clip_coef
            param -= self.learning_rate * grad

    def zero_grad(self):
        for _, grad in self._get_params_and_grads():
            grad.fill(0)
        # Also reset gradients in GRU cells
        for layer in self.layers:
            if hasattr(layer, 'zero_grad'):
                layer.zero_grad()