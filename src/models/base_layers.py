import cupy as cp
from src.utils import sigmoid


class Embedding(object):
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weights = cp.random.uniform(-0.1, 0.1, (vocab_size, embed_dim))

        self.d_weights = cp.zeros_like(self.weights)

        self.params = {
            'weights': (self.weights, self.d_weights)
        }
        # Cache for backward pass
        self.input_ids = None

    def forward(self, input_ids):
        self.input_ids = input_ids
        return self.weights[input_ids]

    def backward(self, grad_output):
        self.d_weights.fill(0)
        cp.add.at(self.d_weights, self.input_ids, grad_output)
        return None


class LinearLayer(object):
    def __init__(self, input_dim, output_dim):
        self.weights = cp.random.uniform(-0.1, 0.1, (input_dim, output_dim))
        self.bias = cp.zeros(output_dim)
    
        self.d_weights = cp.zeros_like(self.weights)
        self.d_bias = cp.zeros_like(self.bias)

        self.params = {
            'weights': (self.weights, self.d_weights),
            'bias': (self.bias, self.d_bias)
        }
        # Cache for backward pass
        self.input = None

    def forward(self, x):
        self.input = x
        return cp.dot(x, self.weights) + self.bias

    def backward(self, grad_output):
        input_flat = self.input.reshape(-1, self.input.shape[-1])
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])

        self.d_weights += cp.dot(input_flat.T, grad_output_flat)
        self.d_bias += cp.sum(grad_output_flat, axis=0)
        
        d_input_flat = cp.dot(grad_output_flat, self.weights.T)
        d_input = d_input_flat.reshape(self.input.shape)
        return d_input


class GRUCell(object):
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Update gate parameters (how much of old mem to keep)
        self.W_z = cp.random.uniform(-0.1, 0.1, (input_dim, hidden_dim))
        self.U_z = cp.random.uniform(-0.1, 0.1, (hidden_dim, hidden_dim))
        self.b_z = cp.zeros(hidden_dim)
        # Reset gate parameters (how much of old mem to forget)
        self.W_r = cp.random.uniform(-0.1, 0.1, (input_dim, hidden_dim))
        self.U_r = cp.random.uniform(-0.1, 0.1, (hidden_dim, hidden_dim))
        self.b_r = cp.zeros(hidden_dim)
        # New memory content parameters 
        self.W_h = cp.random.uniform(-0.1, 0.1, (input_dim, hidden_dim))
        self.U_h = cp.random.uniform(-0.1, 0.1, (hidden_dim, hidden_dim))
        self.b_h = cp.zeros(hidden_dim)

        self.d_W_z, self.d_U_z, self.d_b_z = [cp.zeros_like(p) for p in [self.W_z, self.U_z, self.b_z]]
        self.d_W_r, self.d_U_r, self.d_b_r = [cp.zeros_like(p) for p in [self.W_r, self.U_r, self.b_r]]
        self.d_W_h, self.d_U_h, self.d_b_h = [cp.zeros_like(p) for p in [self.W_h, self.U_h, self.b_h]]

        self.params = {
            'W_z': (self.W_z, self.d_W_z), 'U_z': (self.U_z, self.d_U_z), 'b_z': (self.b_z, self.d_b_z),
            'W_r': (self.W_r, self.d_W_r), 'U_r': (self.U_r, self.d_U_r), 'b_r': (self.b_r, self.d_b_r),
            'W_h': (self.W_h, self.d_W_h), 'U_h': (self.U_h, self.d_U_h), 'b_h': (self.b_h, self.d_b_h)
        }
        self.cache = {}

    def forward(self, x_t, h_prev):
        # "How much of the new information do we add?"
        z_t = sigmoid(cp.dot(x_t, self.W_z) + cp.dot(h_prev, self.U_z) + self.b_z)
        # "How much of the old information do we forget?"
        r_t = sigmoid(cp.dot(x_t, self.W_r) + cp.dot(h_prev, self.U_r) + self.b_r)

        h_tilde = cp.tanh(cp.dot(x_t, self.W_h) + cp.dot(r_t * h_prev, self.U_h) + self.b_h)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde

        self.cache = {'x_t': x_t, 'h_prev': h_prev, 'z_t': z_t, 'r_t': r_t, 'h_tilde': h_tilde}
        return h_t

    def zero_grad(self):
        self.d_W_z.fill(0)
        self.d_U_z.fill(0)
        self.d_b_z.fill(0)
        self.d_W_r.fill(0)
        self.d_U_r.fill(0)
        self.d_b_r.fill(0)
        self.d_W_h.fill(0)
        self.d_U_h.fill(0)
        self.d_b_h.fill(0)

    def backward(self, grad_output):
        x_t, h_prev, z_t, r_t, h_tilde = self.cache.values()
        
        d_h_tilde = grad_output * z_t
        d_z_t = grad_output * (h_tilde - h_prev)
        d_h_prev = grad_output * (1 - z_t)
        d_tanh_input = d_h_tilde * (1 - h_tilde**2)
        
        self.d_b_h += cp.sum(d_tanh_input, axis=0)
        self.d_W_h += cp.dot(x_t.T, d_tanh_input)
        
        d_rh = cp.dot(d_tanh_input, self.U_h.T)
        self.d_U_h += cp.dot((r_t * h_prev).T, d_tanh_input)
        
        d_r_t = d_rh * h_prev
        d_h_prev += d_rh * r_t
        d_xt = cp.dot(d_tanh_input, self.W_h.T)
        d_sigmoid_input_r = d_r_t * r_t * (1 - r_t)
        
        self.d_b_r += cp.sum(d_sigmoid_input_r, axis=0)
        self.d_W_r += cp.dot(x_t.T, d_sigmoid_input_r)
        self.d_U_r += cp.dot(h_prev.T, d_sigmoid_input_r)
        
        d_h_prev += cp.dot(d_sigmoid_input_r, self.U_r.T)
        d_xt += cp.dot(d_sigmoid_input_r, self.W_r.T)
        d_sigmoid_input_z = d_z_t * z_t * (1 - z_t)
        
        self.d_b_z += cp.sum(d_sigmoid_input_z, axis=0)
        self.d_W_z += cp.dot(x_t.T, d_sigmoid_input_z)
        self.d_U_z += cp.dot(h_prev.T, d_sigmoid_input_z)
        
        d_h_prev += cp.dot(d_sigmoid_input_z, self.U_z.T)
        d_xt += cp.dot(d_sigmoid_input_z, self.W_z.T)
        
        return d_xt, d_h_prev


class GRULayer(object):
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        self.gru_cell = GRUCell(input_dim, hidden_dim)

        self.caches = []
    
    @property
    def params(self):
        return self.gru_cell.params

    def forward(self, inputs, initial_hidden=None):
        batch_size, seq_len, _ = inputs.shape
        self.caches = []

        # Will use the last hidden state for the Decoder
        if initial_hidden is not None:
            h_t = initial_hidden
        else:
            h_t = cp.zeros((batch_size, self.hidden_dim))

        hidden_states = []
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            h_t = self.gru_cell.forward(x_t, h_t)
            hidden_states.append(h_t)
            self.caches.append(self.gru_cell.cache.copy())

        return cp.stack(hidden_states, axis=1) # Stack of all hidden states

    def backward(self, grad_output):
        batch_size, seq_len, _ = grad_output.shape
        
        d_initial_hidden = cp.zeros((batch_size, self.hidden_dim))
        d_inputs = cp.zeros((batch_size, seq_len, self.gru_cell.input_dim))
        d_h_next = cp.zeros((batch_size, self.hidden_dim))

        # Loop backward through time
        for t in reversed(range(seq_len)):
            d_ht = grad_output[:, t, :] + d_h_next
            self.gru_cell.cache = self.caches[t]
            d_xt, d_h_prev = self.gru_cell.backward(d_ht)
            d_inputs[:, t, :] = d_xt
            d_h_next = d_h_prev
        
        d_initial_hidden = d_h_next
        return d_inputs, d_initial_hidden


class BidirectionalGRULayer(object):
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        self.forward_gru = GRULayer(input_dim, hidden_dim)
        self.backward_gru = GRULayer(input_dim, hidden_dim)

    @property
    def params(self):
        # Combine params from both directions
        combined_params = {}
        for key, value in self.forward_gru.params.items():
            combined_params[f'fwd_{key}'] = value
        for key, value in self.backward_gru.params.items():
            combined_params[f'bwd_{key}'] = value
        return combined_params

    def forward(self, inputs):
        h_fwd = self.forward_gru.forward(inputs)
        # Reverse the input time axis for backward pass and reverse the output to align with h_fwd
        h_bwd = self.backward_gru.forward(inputs[:, ::-1, :])[:, ::-1, :]

        h_combined = cp.concatenate([h_fwd, h_bwd], axis=-1)
        return h_combined
    
    def backward(self, grad_output):
        half_hidden_dim = self.hidden_dim

        d_h_fwd = grad_output[:, :, :half_hidden_dim]
        d_h_bwd = grad_output[:, :, half_hidden_dim:]

        d_inputs_fwd, d_initial_hidden_fwd = self.forward_gru.backward(d_h_fwd)
        # Reverse the gradient time axis for backward GRU
        d_h_bwd_reversed = d_h_bwd[:, ::-1, :]
        d_inputs_bwd_reversed, d_initial_hidden_bwd = self.backward_gru.backward(d_h_bwd_reversed)
        d_inputs_bwd = d_inputs_bwd_reversed[:, ::-1, :]

        d_inputs = d_inputs_fwd + d_inputs_bwd
        return d_inputs, d_initial_hidden_fwd, d_initial_hidden_bwd
