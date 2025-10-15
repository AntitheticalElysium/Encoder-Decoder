import cupy as cp

class BahdanauAttention(object):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim):
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.attention_dim = attention_dim
        # Project encoder states to attention space
        limit_encoder = cp.sqrt(6.0 / (encoder_hidden_dim + attention_dim))
        self.W_encoder = cp.random.uniform(-limit_encoder, limit_encoder, (encoder_hidden_dim, attention_dim))
        self.d_W_encoder = cp.zeros_like(self.W_encoder)
        # Project decoder state to attention space
        limit_decoder = cp.sqrt(6.0 / (decoder_hidden_dim + attention_dim))
        self.W_decoder = cp.random.uniform(-limit_decoder, limit_decoder, (decoder_hidden_dim, attention_dim))
        self.d_W_decoder = cp.zeros_like(self.W_decoder)
        # Final attention scoring vector - initialize with larger scale
        limit_v = cp.sqrt(6.0 / (attention_dim + 1))
        self.v = cp.random.uniform(-limit_v, limit_v, (attention_dim, 1))
        self.d_v = cp.zeros_like(self.v)
        
        self.params = {
            'W_encoder': (self.W_encoder, self.d_W_encoder),
            'W_decoder': (self.W_decoder, self.d_W_decoder),
            'v': (self.v, self.d_v)
        }

        # Cache for backward pass
        self.caches = [] 
    
    def forward(self, encoder_states, decoder_state):
        batch_size, src_seq_len, _ = encoder_states.shape

        # Project to attention space
        projected_encoder = cp.dot(encoder_states.reshape(-1, self.encoder_hidden_dim),
            self.W_encoder).reshape(batch_size, src_seq_len, self.attention_dim)
        projected_decoder = cp.dot(decoder_state, self.W_decoder)
        
        # "How does this src token relate to what we're decoding?"
        combined = projected_encoder + projected_decoder[:, None, :]
        combined_tanh = cp.tanh(combined)

        # Score for how relevant each token is to decoder state (0 to 1)
        alignment_scores = cp.dot(combined_tanh.reshape(-1, self.attention_dim),
            self.v).reshape(batch_size, src_seq_len)
        attention_weights = self._softmax(alignment_scores)
        
        context_vector = cp.matmul(attention_weights[:, None, :], encoder_states).squeeze(1)
        
        # Cache everything needed for backward
        cache = {
            'encoder_states': encoder_states,
            'decoder_state': decoder_state,
            'projected_encoder': projected_encoder,
            'projected_decoder': projected_decoder,
            'combined': combined,
            'combined_tanh': combined_tanh,
            'attention_weights': attention_weights
        }
        self.caches.append(cache)
        
        return context_vector, attention_weights
    
    def backward(self, grad_context, timestep):
        cache = self.caches[timestep]
        encoder_states = cache['encoder_states']
        decoder_state = cache['decoder_state']
        attention_weights = cache['attention_weights']
        
        batch_size, src_seq_len, _ = encoder_states.shape

        grad_encoder_from_sum = attention_weights[:, :, None] * grad_context[:, None, :]
        grad_attention_weights = cp.sum(
            encoder_states * grad_context[:, None, :],
            axis=2
        )
        # Backprop through softmax
        grad_alignment_scores = self._softmax_backward(
            attention_weights,
            grad_attention_weights
        )
        
        grad_alignment_scores_expanded = grad_alignment_scores[:, :, None]
        combined_tanh = cache['combined_tanh']
        self.d_v += cp.dot(
            combined_tanh.reshape(-1, self.attention_dim).T,
            grad_alignment_scores_expanded.reshape(-1, 1)
        )
        

        grad_combined_tanh = grad_alignment_scores_expanded * self.v.T
        
        # Backprop through tanh
        grad_combined = grad_combined_tanh * (1 - combined_tanh ** 2)
        
        # Split grads
        grad_projected_encoder = grad_combined
        grad_projected_decoder = cp.sum(grad_combined, axis=1)
        

        self.d_W_encoder += cp.dot(
            encoder_states.reshape(-1, self.encoder_hidden_dim).T,
            grad_projected_encoder.reshape(-1, self.attention_dim)
        )
        grad_encoder_from_proj = cp.dot(
            grad_projected_encoder.reshape(-1, self.attention_dim),
            self.W_encoder.T
        ).reshape(batch_size, src_seq_len, self.encoder_hidden_dim)
        

        self.d_W_decoder += cp.dot(
            decoder_state.T,
            grad_projected_decoder
        )
        
        # Combine grads
        grad_decoder_state = cp.dot(grad_projected_decoder, self.W_decoder.T)
        grad_encoder_states = grad_encoder_from_sum + grad_encoder_from_proj
        
        return grad_encoder_states, grad_decoder_state
    
    def _softmax(self, x):
        exp_x = cp.exp(x - cp.max(x, axis=1, keepdims=True))
        return exp_x / cp.sum(exp_x, axis=1, keepdims=True)
    
    def _softmax_backward(self, probs, grad_output):
        sum_term = cp.sum(probs * grad_output, axis=1, keepdims=True)
        return probs * (grad_output - sum_term)
    
    def zero_grad(self):
        self.d_W_encoder.fill(0)
        self.d_W_decoder.fill(0)
        self.d_v.fill(0)