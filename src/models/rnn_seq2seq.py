import cupy as np
from src.utils import sigmoid, softmax

class Embedding(object):
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weights = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim))

        self.d_weights = np.zeros_like(self.weights)

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
        np.add.at(self.d_weights, self.input_ids, grad_output)
        return None


class GRUCell(object):
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Update gate parameters (how much of old mem to keep)
        self.W_z = np.random.uniform(-0.1, 0.1, (input_dim, hidden_dim))
        self.U_z = np.random.uniform(-0.1, 0.1, (hidden_dim, hidden_dim))
        self.b_z = np.zeros(hidden_dim)
        # Reset gate parameters (how much of old mem to forget)
        self.W_r = np.random.uniform(-0.1, 0.1, (input_dim, hidden_dim))
        self.U_r = np.random.uniform(-0.1, 0.1, (hidden_dim, hidden_dim))
        self.b_r = np.zeros(hidden_dim)
        # New memory content parameters 
        self.W_h = np.random.uniform(-0.1, 0.1, (input_dim, hidden_dim))
        self.U_h = np.random.uniform(-0.1, 0.1, (hidden_dim, hidden_dim))
        self.b_h = np.zeros(hidden_dim)

        self.d_W_z, self.d_U_z, self.d_b_z = [np.zeros_like(p) for p in [self.W_z, self.U_z, self.b_z]]
        self.d_W_r, self.d_U_r, self.d_b_r = [np.zeros_like(p) for p in [self.W_r, self.U_r, self.b_r]]
        self.d_W_h, self.d_U_h, self.d_b_h = [np.zeros_like(p) for p in [self.W_h, self.U_h, self.b_h]]

        self.params = {
            'W_z': (self.W_z, self.d_W_z), 'U_z': (self.U_z, self.d_U_z), 'b_z': (self.b_z, self.d_b_z),
            'W_r': (self.W_r, self.d_W_r), 'U_r': (self.U_r, self.d_U_r), 'b_r': (self.b_r, self.d_b_r),
            'W_h': (self.W_h, self.d_W_h), 'U_h': (self.U_h, self.d_U_h), 'b_h': (self.b_h, self.d_b_h)
        }
        self.cache = {}

    def forward(self, x_t, h_prev):
        # "How much of the new information do we add?"
        z_t = sigmoid(np.dot(x_t, self.W_z) + np.dot(h_prev, self.U_z) + self.b_z)
        # "How much of the old information do we forget?"
        r_t = sigmoid(np.dot(x_t, self.W_r) + np.dot(h_prev, self.U_r) + self.b_r)

        h_tilde = np.tanh(np.dot(x_t, self.W_h) + np.dot(r_t * h_prev, self.U_h) + self.b_h)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde

        self.cache = {'x_t': x_t, 'h_prev': h_prev, 'z_t': z_t, 'r_t': r_t, 'h_tilde': h_tilde}
        return h_t

    def backward(self, grad_output):
        x_t, h_prev, z_t, r_t, h_tilde = self.cache.values()
        
        d_h_tilde = grad_output * z_t
        d_z_t = grad_output * (h_tilde - h_prev)
        d_h_prev = grad_output * (1 - z_t)
        d_tanh_input = d_h_tilde * (1 - h_tilde**2)
        
        self.d_b_h += np.sum(d_tanh_input, axis=0)
        self.d_W_h += np.dot(x_t.T, d_tanh_input)
        
        d_rh = np.dot(d_tanh_input, self.U_h.T)
        self.d_U_h += np.dot((r_t * h_prev).T, d_tanh_input)
        
        d_r_t = d_rh * h_prev
        d_h_prev += d_rh * r_t
        d_xt = np.dot(d_tanh_input, self.W_h.T)
        d_sigmoid_input_r = d_r_t * r_t * (1 - r_t)
        
        self.d_b_r += np.sum(d_sigmoid_input_r, axis=0)
        self.d_W_r += np.dot(x_t.T, d_sigmoid_input_r)
        self.d_U_r += np.dot(h_prev.T, d_sigmoid_input_r)
        
        d_h_prev += np.dot(d_sigmoid_input_r, self.U_r.T)
        d_xt += np.dot(d_sigmoid_input_r, self.W_r.T)
        d_sigmoid_input_z = d_z_t * z_t * (1 - z_t)
        
        self.d_b_z += np.sum(d_sigmoid_input_z, axis=0)
        self.d_W_z += np.dot(x_t.T, d_sigmoid_input_z)
        self.d_U_z += np.dot(h_prev.T, d_sigmoid_input_z)
        
        d_h_prev += np.dot(d_sigmoid_input_z, self.U_z.T)
        d_xt += np.dot(d_sigmoid_input_z, self.W_z.T)
        
        return d_xt, d_h_prev


class GRULayer(object):
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        self.gru_cell = GRUCell(input_dim, hidden_dim)

        self.caches = []

    def forward(self, inputs, initial_hidden=None):
        batch_size, seq_len, _ = inputs.shape
        self.caches = []

        # Will use the last hidden state for the Decoder
        if initial_hidden is not None:
            h_t = initial_hidden
        else:
            h_t = np.zeros((batch_size, self.hidden_dim))

        hidden_states = []
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            h_t = self.gru_cell.forward(x_t, h_t)
            hidden_states.append(h_t)
            self.caches.append(self.gru_cell.cache)

        return np.stack(hidden_states, axis=1) # Stack of all hidden states

    def backward(self, grad_output):
        batch_size, seq_len, _ = grad_output.shape
        
        d_initial_hidden = np.zeros((batch_size, self.hidden_dim))
        d_inputs = np.zeros((batch_size, seq_len, self.gru_cell.input_dim))
        d_h_next = np.zeros((batch_size, self.hidden_dim))

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

    def forward(self, inputs):
        h_fwd = self.forward_gru.forward(inputs)
        # Reverse the input time axis for backward pass and reverse the output to align with h_fwd
        h_bwd = self.backward_gru.forward(inputs[:, ::-1, :])[:, ::-1, :]

        h_combined = np.concatenate([h_fwd, h_bwd], axis=-1)
        return h_combined
    
    def backward(self, grad_output):
        batch_size, seq_len, _ = grad_output.shape
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


class Encoder(object):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.layers = []

        self.layers.append(BidirectionalGRULayer(embed_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.layers.append(BidirectionalGRULayer(hidden_dim * 2, hidden_dim))

        self.layers_outputs_shape = None

    def forward(self, input_ids):
        embedded = self.embedding.forward(input_ids)
        for layer in self.layers:
            embedded = layer.forward(embedded)
        self.layers_outputs_shape = embedded.shape
        return embedded

    def backward(self, d_context_vect):
        d_encoder_hidden = np.zeros(self.layers_outputs_shape)
        hidden_dim = self.layers[-1].hidden_dim # hidden_dim per direction

        d_last_fwd = d_context_vect[:, :hidden_dim]
        d_first_bwd = d_context_vect[:, hidden_dim:]

        # grad for the last forward state
        d_encoder_hidden[:, -1, :hidden_dim] = d_last_fwd
        # grad for the first backward state (at time step 0)
        d_encoder_hidden[:, 0, hidden_dim:] = d_first_bwd

        d_next_input = d_encoder_hidden
        for layer in reversed(self.layers):
            d_next_input, _, _ = layer.backward(d_next_input)

        d_embedded = d_next_input
        self.embedding.backward(d_embedded)
        return None


class LinearLayer(object):
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.uniform(-0.1, 0.1, (input_dim, output_dim))
        self.bias = np.zeros(output_dim)
    
        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)

        self.params = {
            'weights': (self.weights, self.d_weights),
            'bias': (self.bias, self.d_bias)
        }
        # Cache for backward pass
        self.input = None

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output):
        input_flat = self.input.reshape(-1, self.input.shape[-1])
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])

        self.d_weights = np.dot(input_flat.T, grad_output_flat)
        self.d_bias = np.sum(grad_output_flat, axis=0)
        
        d_input_flat = np.dot(grad_output_flat, self.weights.T)
        d_input = d_input_flat.reshape(self.input.shape)
        return d_input


class Decoder(object):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.layers = []
        # Adapt context vect size to decoder hidden size
        self.layers.append(GRULayer(embed_dim, hidden_dim * 2))
        for _ in range(1, num_layers):
            self.layers.append(GRULayer(hidden_dim * 2, hidden_dim * 2))

        self.fc = LinearLayer(hidden_dim * 2, vocab_size)

    def forward(self, input_ids, context_vector):
        layer_input = self.embedding.forward(input_ids)
        # Use the context_vector as the initial hidden state for the first  layer.
        layer_output = self.layers[0].forward(layer_input, initial_hidden=context_vector)
        layer_input = layer_output

        for i in range(1, len(self.layers)):
            layer_output = self.layers[i].forward(layer_input, initial_hidden=None) # None to use zero state
            layer_input = layer_output

        logits = self.fc.forward(layer_input)
        return logits

    def backward(self, grad_output):
        d_layer_output = self.fc.backward(grad_output)
        d_initial_hidden_for_encoder = None
    
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            # d_layer_input becomes the d_layer_output for the layer below
            d_layer_input, d_initial_hidden = layer.backward(d_layer_output)
            # We only care about the gradient of the initial hidden state for the first layer from the encoder.
            if i == 0:
                d_initial_hidden_for_encoder = d_initial_hidden
            d_layer_output = d_layer_input

        d_embedded = d_layer_output
        self.embedding.backward(d_embedded)
        return d_initial_hidden_for_encoder
