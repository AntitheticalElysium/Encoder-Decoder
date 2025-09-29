import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Embedding(object):
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weights = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim))

    def forward(self, input_ids):
        return self.weights[input_ids]

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

    def forward(self, x_t, h_prev):
        # "How much of the new information do we add?"
        z_t = sigmoid(np.dot(x_t, self.W_z) + np.dot(h_prev, self.U_z) + self.b_z)
        # "How much of the old information do we forget?"
        r_t = sigmoid(np.dot(x_t, self.W_r) + np.dot(h_prev, self.U_r) + self.b_r)

        h_tilde = np.tanh(np.dot(x_t, self.W_h) + np.dot(r_t * h_prev, self.U_h) + self.b_h)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t

class GRULayer(object):
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        self.gru_cell = GRUCell(input_dim, hidden_dim)

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.shape
        h_t = np.zeros((batch_size, self.hidden_dim))
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            h_t = self.gru_cell.forward(x_t, h_t)
        return h_t

class Encoder(object):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.gru = GRULayer(embed_dim, hidden_dim)

    def forward(self, input_ids):
        embedded = self.embedding.forward(input_ids)
        hidden = self.gru.forward(embedded)
        return hidden

if __name__ == "__main__":
    vocab_size = 100
    embed_dim = 64
    hidden_dim = 128
    batch_size = 32
    seq_len = 10

    encoder = Encoder(vocab_size, embed_dim, hidden_dim)
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    hidden = encoder.forward(input_ids)
    print("Encoder output shape:", hidden.shape)  # Should be (batch_size, hidden_dim)
