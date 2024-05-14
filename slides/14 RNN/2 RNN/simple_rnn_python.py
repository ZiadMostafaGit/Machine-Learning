import numpy as np


class SimpleRNN:
    def __init__(self, input_size=9, hidden_size=4, output_size=3):
        # W_xh and W_ho are like old ones.
        self.W_xh = np.random.randn(hidden_size, input_size)    # input to hidden      4x9
        self.W_ho = np.random.randn(output_size, hidden_size)   # hidden to output     3x4
        self.b_xh = np.zeros((hidden_size, 1))  # hidden bias
        self.b_ho = np.zeros((output_size, 1))  # output bias

        # New one: connects the hidden layer to itself at the next time step (recurrent link)
        self.W_hh = np.random.randn(hidden_size, hidden_size) # hidden to hidden        4x4
        self.b_hh = np.zeros((hidden_size, 1))  # hidden-hidden bias

    def forward(self, inputs):
        steps_output, hidden_states = {}, {}
        hidden_states[-1] = np.zeros((self.W_xh.shape[0], 1))   # no history at idx -1

        # feed each input while utilizing its history
        for t in range(len(inputs)):
            x = np.array(inputs[t]).reshape(-1, 1)              # 9x1
            # Normal input to hidden transformation (embedding)
            hidden_cur = np.dot(self.W_xh, x) + self.b_xh       # 4x9 * 9x1 + 4x1 = 4x1

            # Transform the previous hidden state to the current timestep (avoid naive addition)
            hidden_states[t] = np.dot(self.W_hh, hidden_states[t-1]) + self.b_hh
            hidden_states[t] += hidden_cur              # Element-wise addition cur + old
            hidden_states[t] = np.tanh(hidden_states[t])# Non-linear transformation to enhance addition

            # Normal hidden to output transformation
            steps_output[t] = np.dot(self.W_ho, hidden_states[t]) + self.b_ho

        return steps_output, hidden_states


if __name__ == '__main__':
    sequence_length = 10
    input_size = 9
    hidden_size = 4
    output_size = 3

    rnn = SimpleRNN(input_size, hidden_size, output_size)

    # 10 steps, each step has feature vector of length 9
    inputs = [np.random.randn(input_size) for _ in range(sequence_length)]

    output, hidden_states = rnn.forward(inputs)
    print(output.keys())
