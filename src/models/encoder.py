import cupy as cp
from src.models.base_layers import Embedding, BidirectionalGRULayer


class Encoder(object):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = Embedding(vocab_size, embed_dim)
        self.layers = []

        self.layers.append(BidirectionalGRULayer(embed_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.layers.append(BidirectionalGRULayer(hidden_dim * 2, hidden_dim))

        self.layers_outputs_shape = None

    @property
    def params(self):
        combined_params = {'embedding': self.embedding.params}
        for i, layer in enumerate(self.layers):
            combined_params[f'layer_{i}'] = layer.params
        return combined_params

    def forward(self, input_ids):
        embedded = self.embedding.forward(input_ids)
        for layer in self.layers:
            embedded = layer.forward(embedded)
        self.layers_outputs_shape = embedded.shape
        return embedded

    def backward(self, d_context_vect):
        d_encoder_hidden = cp.zeros(self.layers_outputs_shape)
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
