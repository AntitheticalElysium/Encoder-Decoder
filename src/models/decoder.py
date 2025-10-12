from src.models.base_layers import Embedding, GRULayer, LinearLayer


class Decoder(object):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = Embedding(vocab_size, embed_dim)
        self.layers = []
        # Adapt context vect size to decoder hidden size
        self.layers.append(GRULayer(embed_dim, hidden_dim * 2))
        for _ in range(1, num_layers):
            self.layers.append(GRULayer(hidden_dim * 2, hidden_dim * 2))

        self.fc = LinearLayer(hidden_dim * 2, vocab_size)

    @property
    def params(self):
        combined_params = {
            'embedding': self.embedding.params,
            'fc': self.fc.params
        }
        for i, layer in enumerate(self.layers):
            combined_params[f'layer_{i}'] = layer.params
        return combined_params

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
