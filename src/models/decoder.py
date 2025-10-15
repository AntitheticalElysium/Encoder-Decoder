import cupy as cp

from src.models.base_layers import Embedding, GRULayer, LinearLayer
from src.models.attention import BahdanauAttention


class Decoder(object):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, attention_dim):
        # Because encoder is bidirectional
        encoder_hidden_dim = hidden_dim * 2

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = Embedding(vocab_size, embed_dim)

        # Attention mechanisms
        self.attention = BahdanauAttention(encoder_hidden_dim, hidden_dim * 2, attention_dim)

        self.layers = []
        # Adapt context vect size to decoder hidden size
        self.layers.append(GRULayer(embed_dim + encoder_hidden_dim, hidden_dim * 2))
        for _ in range(1, num_layers):
            self.layers.append(GRULayer(hidden_dim * 2, hidden_dim * 2))

        self.fc = LinearLayer(hidden_dim * 2, vocab_size)
        # Cache for backward
        self.encoder_states = None
        self.attention_contexts = []
        self.attention_weights_list = []

    @property
    def params(self):
        combined_params = {
            'embedding': self.embedding.params,
            'attention': self.attention.params,
            'fc': self.fc.params
        }
        for i, layer in enumerate(self.layers):
            combined_params[f'layer_{i}'] = layer.params
        return combined_params

    def forward(self, input_ids, encoder_states, initial_hidden=None):
        batch_size, tgt_seq_len = input_ids.shape
        self.encoder_states = encoder_states
        self.attention_contexts = []
        self.attention_weights_list = []
        
        # Embed input tokens
        embedded = self.embedding.forward(input_ids)
        
        # Init hidden state
        if initial_hidden is not None:
            h_t = initial_hidden
        else:
            h_t = cp.zeros((batch_size, self.hidden_dim * 2))
        
        # Reset caches for first layer and attention
        self.layers[0].caches = []
        self.attention.caches = []
        
        # Process each timestep with attention
        outputs = []
        for t in range(tgt_seq_len):
            x_t = embedded[:, t, :] 
            # Compute attention context
            context_t, attn_weights = self.attention.forward(encoder_states, h_t)
            self.attention_contexts.append(context_t)
            self.attention_weights_list.append(attn_weights)
            # Concatenate embedding with context
            decoder_input = cp.concatenate([x_t, context_t], axis=-1)
            # Pass through first GRU layer
            h_t = self.layers[0].gru_cell.forward(decoder_input, h_t)
            # Store cache for backward pass
            self.layers[0].caches.append(self.layers[0].gru_cell.cache.copy())
            outputs.append(h_t)
        
        hidden_states = cp.stack(outputs, axis=1) 
        
        # Pass through remaining GRU layers
        for i in range(1, len(self.layers)):
            hidden_states = self.layers[i].forward(hidden_states, initial_hidden=None)
        
        logits = self.fc.forward(hidden_states)
        return logits

    def backward(self, grad_output):
        batch_size, tgt_seq_len, _ = grad_output.shape
        
        d_hidden = self.fc.backward(grad_output)
        
        # Backward through GRU layers (not first)
        for i in reversed(range(1, len(self.layers))):
            d_hidden, _ = self.layers[i].backward(d_hidden)
        # Backward through rest
        grad_encoder_states = cp.zeros(self.encoder_states.shape)
        d_h_next = cp.zeros((batch_size, self.hidden_dim * 2))
        d_embedded = cp.zeros((batch_size, tgt_seq_len, self.embedding.embed_dim))
        
        for t in reversed(range(tgt_seq_len)):
            d_h_t = d_hidden[:, t, :] + d_h_next
            # Backward through GRU cell
            self.layers[0].gru_cell.cache = self.layers[0].caches[t]
            d_decoder_input, d_h_prev = self.layers[0].gru_cell.backward(d_h_t)
            # Split grad
            d_x_t = d_decoder_input[:, :self.embedding.embed_dim]
            d_context_t = d_decoder_input[:, self.embedding.embed_dim:]
            # Backward through attention with correct timestep
            d_encoder_t, d_h_for_attn = self.attention.backward(d_context_t, t)
            grad_encoder_states += d_encoder_t
            # Accumulate gradients
            d_embedded[:, t, :] = d_x_t
            d_h_next = d_h_prev + d_h_for_attn
        
        # Backward through embedding
        self.embedding.backward(d_embedded)
        return grad_encoder_states
