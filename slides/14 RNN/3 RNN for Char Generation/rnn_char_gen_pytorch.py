# ChatGPT
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    # Forward pass
    def forward(self, x, hidden):
        #out, hidden = self.rnn(x)          # default
        out, hidden = self.rnn(x, hidden)   # allows u resume from a state
        out = self.fc(out)
        return out, hidden

    def nohistory_hidden_state(self, batch_size):
        # return tensor of zeros as a begin
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)


def generate(model, prefix_str, size, eof='$'):
    model.eval()
    chars = []

    # Build the initial hidden from the given prefix
    hidden = model.nohistory_hidden_state(1)
    for char in prefix_str:
        # Build a one-hot-encoding for the current char
        char_tensor = torch.zeros(1, 1, vocab_size)
        char_tensor[0, 0, char2int[char]] = 1
        out, hidden = model(char_tensor, hidden)
        #print(char, int2char[out.argmax().item()])

    # Now hidden represents all input letters and its out can predict a letter

    def generate_letter():
        # given logits, compute the probabilities and sample a letter
        p = torch.nn.functional.softmax(out[0, 0], dim=0).detach().numpy()

        # select a random index (generation) based on the distribution (weights)
        # This allows us to generate several possible answers, like chatgpt
        char_idx = np.random.choice(vocab_size, p=p)
        #char_idx = out.argmax().item() # this just select a single most probable answer

        char = int2char[char_idx]
        return char

    for _ in range(size):
        # use the last out logits to generate a new letter
        char = generate_letter()
        chars.append(char)

        if char == eof:
            break

        char_tensor = torch.zeros(1, 1, vocab_size)
        char_tensor[0, 0, char2int[char]] = 1
        # Predict the next letter given the current one and its history
        out, hidden = model(char_tensor, hidden)

    return ''.join(chars)


if __name__ == '__main__':

    # lets use some letter as EOF like $
    sequences = [
        "Get Skilled in Machine Learning$$$$",
        "By CS-Get Skilled Academy$$$$",
        "Instructor Mostafa Saad Ibrahim$$$$"
    ]
    all_sequences_data = ''.join(sequences)
    chars = tuple(set(all_sequences_data))
    vocab_size = len(chars)

    # Character to index and index to character mappings
    char2int = {ch: ii for ii, ch in enumerate(chars)}
    int2char = {ii: ch for ii, ch in enumerate(chars)}

    # Prepare the model and optimizer
    hidden_size = 128
    n_layers = 1
    batch_size = 1
    n_epochs = 100
    learning_rate = 0.01

    model = CharRNN(vocab_size, hidden_size, vocab_size, n_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training the model
    model.train()
    for epoch in range(n_epochs):
        for data in sequences:
            # the whole document is a single sequence (one hidden state)

            hidden = model.nohistory_hidden_state(batch_size)
            # Below, I feed the whole sequence as a single batch (better performance for a few examples)
            # We also can divide sequences to subsequences (but all on SAME hidden state): e.g. seq_length = 10
            seq_length = len(data)  # explore values like 10 and 20
            for batch in range(0, len(data) - seq_length + 1, seq_length):
                X = torch.zeros(batch_size, seq_length, vocab_size)         # 1x35x30  (35 seq len, 30 voc size)
                y = torch.zeros(batch_size, seq_length, dtype=torch.long)   # 1x30
                s1, s2 = '', ''
                for i in range(seq_length):
                    s1, s2 = data[batch + i], data[batch + i + 1 if batch + i + 1 < len(data) else 0]
                    X[0, i, char2int[s1]] = 1
                    y[0, i] = char2int[s2]
                    #print(s1, '*', s2)


                optimizer.zero_grad()
                output, hidden = model(X, hidden)
                loss = criterion(output.squeeze(0), y.squeeze(0))
                loss.backward()
                optimizer.step()
                hidden.detach_()    # to avoid pytorch error for seq_length < len(data). comment and see

                answers = torch.max(output.squeeze(0), dim=1)[1]
                train_acc = torch.sum(answers == y.squeeze(0)) / y.squeeze(0).size()[0]


            print(f'Epoch {epoch+1}, Loss: {loss.item():.3f} - Accuracy: {train_acc.item():.2f}')


    # Generate some text that starts with this prefix
    print(generate(model, size=50, prefix_str='Get Skilled'))
    print(generate(model, size=50, prefix_str='By'))
    print(generate(model, size=50, prefix_str='Instructor'))
    print(generate(model, size=50, prefix_str='lls'))
