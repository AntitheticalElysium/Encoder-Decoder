import cupy as cp
from src.models.encoder import Encoder
from src.models.decoder import Decoder


class Seq2Seq(object):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        
    @classmethod
    def create(cls, vocab_size_src, vocab_size_tgt, embed_dim, hidden_dim, num_layers):
        encoder = Encoder(vocab_size_src, embed_dim, hidden_dim, num_layers)
        decoder = Decoder(vocab_size_tgt, embed_dim, hidden_dim, num_layers)
        return cls(encoder, decoder)
    
    def encode(self, source_ids):
        return self.encoder.forward(source_ids)
    
    def get_context_vector(self, encoder_hidden_states):
        hidden_dim = self.encoder.hidden_dim
        last_fwd = encoder_hidden_states[:, -1, :hidden_dim]
        first_bwd = encoder_hidden_states[:, 0, hidden_dim:]
        context_vector = cp.concatenate([last_fwd, first_bwd], axis=-1)
        return context_vector
    
    def decode(self, target_ids, context_vector):
        return self.decoder.forward(target_ids, context_vector)
    
    def forward(self, source_ids, target_ids):
        encoder_hidden_states = self.encode(source_ids)
        context_vector = self.get_context_vector(encoder_hidden_states)
        logits = self.decode(target_ids, context_vector)
        return logits
    
    def backward(self, grad_output):
        d_context_vector = self.decoder.backward(grad_output)
        self.encoder.backward(d_context_vector)
    
    def get_all_params(self):
        all_layers = [
            self.encoder.embedding,
            self.decoder.embedding,
            self.decoder.fc
        ]
        # Add encoder GRU cells
        for layer in self.encoder.layers:
            all_layers.append(layer.forward_gru.gru_cell)
            all_layers.append(layer.backward_gru.gru_cell)
        # Add decoder GRU cells
        for layer in self.decoder.layers:
            all_layers.append(layer.gru_cell)
        
        return all_layers
    
    def predict(self, sentence, vocab_src, vocab_tgt, max_len=10):
        # Check if input is a string or already token IDs
        if isinstance(sentence, str):
            # Preprocess
            sentence = sentence.lower()
            for symbol in '.,!?':
                sentence = sentence.replace(symbol, f' {symbol} ')
            # Tokenize
            tokens = [token for token in sentence.split(' ') if token]
            # Encode tokens to IDs
            token_ids = vocab_src.encode(tokens)
            token_ids.append(vocab_src.stoi['<eos>'])
            # Convert to cupy array
            source_ids = cp.array(token_ids)
        else:
            source_ids = sentence
        
        # Handle both batched and single sequence input
        if len(source_ids.shape) == 1:
            source_ids = source_ids.reshape(1, -1)
        
        # Encode the source sequence
        encoder_hidden_states = self.encode(source_ids)
        context_vector = self.get_context_vector(encoder_hidden_states)
        
        generated_ids = [vocab_tgt.stoi['<sos>']]
        translated_token_ids = []
        
        for _ in range(max_len):
            # Pass the full sequence generated so far
            decoder_input = cp.array(generated_ids).reshape(1, -1)
            logits = self.decode(decoder_input, context_vector)
            
            predicted_id = int(cp.argmax(logits[:, -1, :], axis=-1).item())
            
            if predicted_id == vocab_tgt.stoi['<eos>']:
                break
            
            translated_token_ids.append(predicted_id)
            generated_ids.append(predicted_id)
        
        # Decode back to string
        translated_tokens = vocab_tgt.decode(translated_token_ids, skip_special_tokens=True)
        translated_sentence = " ".join(translated_tokens)
        
        return translated_sentence
