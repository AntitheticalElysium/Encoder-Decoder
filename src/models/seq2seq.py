import cupy as cp
from src.models.encoder import Encoder
from src.models.decoder import Decoder


class Seq2Seq(object):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        
    @classmethod
    def create(cls, vocab_size_src, vocab_size_tgt, embed_dim, hidden_dim, num_layers, attention_dim):
        encoder = Encoder(vocab_size_src, embed_dim, hidden_dim, num_layers)
        decoder = Decoder(vocab_size_tgt, embed_dim, hidden_dim, num_layers, attention_dim)
        return cls(encoder, decoder)
    
    def encode(self, source_ids):
        return self.encoder.forward(source_ids)
    
    def get_initial_decoder_state(self, encoder_states):
        hidden_dim = self.encoder.hidden_dim
        last_fwd = encoder_states[:, -1, :hidden_dim]
        first_bwd = encoder_states[:, 0, hidden_dim:]
        return cp.concatenate([last_fwd, first_bwd], axis=-1)
    
    def decode(self, target_ids, encoder_states):
        initial_hidden = self.get_initial_decoder_state(encoder_states)
        return self.decoder.forward(target_ids, encoder_states, initial_hidden)
    
    def forward(self, source_ids, target_ids):
        encoder_states = self.encode(source_ids)
        logits = self.decode(target_ids, encoder_states)
        return logits
    
    def backward(self, grad_output):
        # Get gradients from decoder: attention grads and initial hidden grads
        grad_encoder_states, grad_initial_hidden = self.decoder.backward(grad_output)
        
        # Backprop through get_initial_decoder_state
        # initial_hidden = concat([last_fwd, first_bwd])
        # where last_fwd = encoder_states[:, -1, :hidden_dim]
        # and first_bwd = encoder_states[:, 0, hidden_dim:]
        hidden_dim = self.encoder.hidden_dim
        
        # Split gradient
        grad_last_fwd = grad_initial_hidden[:, :hidden_dim]
        grad_first_bwd = grad_initial_hidden[:, hidden_dim:]
        
        # Add to appropriate positions in encoder_states
        grad_encoder_states[:, -1, :hidden_dim] += grad_last_fwd
        grad_encoder_states[:, 0, hidden_dim:] += grad_first_bwd
        
        self.encoder.backward(grad_encoder_states)
    
    def get_all_params(self):
        all_layers = [
            self.encoder.embedding,
            self.decoder.embedding,
            self.decoder.attention, 
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
    
    def predict(self, sentence, vocab_src, vocab_tgt, max_len=15):
        # Preprocess if string
        if isinstance(sentence, str):
            sentence = sentence.lower()
            for symbol in '.,!?':
                sentence = sentence.replace(symbol, f' {symbol} ')
            tokens = [token for token in sentence.split(' ') if token]
            token_ids = vocab_src.encode(tokens)
            token_ids.append(vocab_src.stoi['<eos>'])
            source_ids = cp.array(token_ids)
        else:
            source_ids = sentence
        
        if len(source_ids.shape) == 1:
            source_ids = source_ids.reshape(1, -1)
        
        # Encode source
        encoder_states = self.encode(source_ids)
        initial_hidden = self.get_initial_decoder_state(encoder_states)
        
        generated_ids = [vocab_tgt.stoi['<sos>']]
        translated_token_ids = []
        h_t = initial_hidden
        
        for _ in range(max_len):
            # Get last generated token
            last_token = cp.array([[generated_ids[-1]]])
            embedded = self.decoder.embedding.forward(last_token)[:, 0, :]
            # Compute attention
            context, _ = self.decoder.attention.forward(encoder_states, h_t)
            # Concatenate and pass through GRU
            decoder_input = cp.concatenate([embedded, context], axis=-1)
            h_t = self.decoder.layers[0].gru_cell.forward(decoder_input, h_t)
            # Pass through remaining layers
            h_seq = h_t[:, None, :]
            for i in range(1, len(self.decoder.layers)):
                h_seq = self.decoder.layers[i].forward(h_seq, initial_hidden=None)
            # Get logits
            logits = self.decoder.fc.forward(h_seq)
            predicted_id = int(cp.argmax(logits[:, 0, :], axis=-1).item())
            
            if predicted_id == vocab_tgt.stoi['<eos>']:
                break
            
            translated_token_ids.append(predicted_id)
            generated_ids.append(predicted_id)
        
        # Decode to string
        translated_tokens = vocab_tgt.decode(translated_token_ids, skip_special_tokens=True)
        return " ".join(translated_tokens)