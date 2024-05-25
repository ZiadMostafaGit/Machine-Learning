import numpy as np

np.random.seed(10)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def lstm_forward_pass(inputs, initial_states, Wf, Wi, Wc, Wo, Uf, Ui, Uc, Uo, bf, bi, bc, bo):
    """
    Forward pass for an LSTM.

    inputs: Input sequence, shape (sequence_length, input_dim)
    initial_states: Tuple of initial states (h_0, C_0), each of shape (hidden_dim,)
    Wf, Wi, Wc, Wo: Weight matrices for the forget, input, candidate, and output gates, respectively
    Uf, Ui, Uc, Uo: Recurrent weight matrices for the forget, input, candidate, and output gates, respectively
    bf, bi, bc, bo: Bias vectors for the forget, input, candidate, and output gates, respectively

    Returns: Final hidden and cell states
    """

    def transform(Wx, x_t, Wh, h_t, b):
        # Transform the 2 vectors and do element-wise addition
        return np.dot(Wx, x_t) + np.dot(Wh, h_t) + b

    def gate(Wx, x_t, Wh, h_t, b):
        t = transform(Wx, x_t, Wh, h_t, b)
        return sigmoid(t)   # [0-1] for scaling values



    # Code before gates
    h_t, C_t = initial_states
    for x_t in inputs:
        # non-linear Transformation for input and history (like RNN)
        fused_state = tanh(transform(Wc, x_t, Uc, h_t, bc))

        # ACCUMULATE to the previous history (Cell state)
        C_t = C_t + fused_state
        # Output hidden state
        h_t = tanh(C_t)




    h_t, C_t = initial_states
    for x_t in inputs:
        # What to keep from the history
        f_t = gate(Wf, x_t, Uf, h_t, bf)
        # What to keep from the input / fused state
        i_t = gate(Wi, x_t, Ui, h_t, bi)
        # What to keep from the output
        o_t = gate(Wo, x_t, Uo, h_t, bo)

        fused_state = tanh(transform(Wc, x_t, Uc, h_t, bc))

        # Select from old state + select from fused state
        C_t = f_t * C_t + i_t * fused_state

        # Select what to output
        h_t = o_t * tanh(C_t)

        ...


    return h_t, C_t

# Example of usage
sequence_length = 10
input_dim = 7
hidden_dim = 4

# Randomly initializing weights and biases
np.random.seed(0)
Wf, Wi, Wc, Wo = [np.random.randn(hidden_dim, input_dim) for _ in range(4)]
Uf, Ui, Uc, Uo = [np.random.randn(hidden_dim, hidden_dim) for _ in range(4)]
bf, bi, bc, bo = [np.random.randn(hidden_dim) for _ in range(4)]

# Initial states
h_0 = np.zeros(hidden_dim)
C_0 = np.zeros(hidden_dim)

# Input sequence (random)
inputs = np.random.randn(sequence_length, input_dim)

# Forward pass
final_h, final_C = lstm_forward_pass(inputs, (h_0, C_0), Wf, Wi, Wc, Wo, Uf, Ui, Uc, Uo, bf, bi, bc, bo)

print("Final Hidden State:", final_h)
print("Final Cell State:", final_C)
