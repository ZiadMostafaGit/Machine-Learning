# ChatGPT
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class WordRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1):
        super(WordRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    # Forward pass
    def forward(self, x, hidden = None):
        x = self.embedding(x) # convert idx to embedding_dim: words x embedding_dim

        if hidden is None:
            out, hidden = self.rnn(x)
        else:
            out, hidden = self.rnn(x, hidden)

        out = self.fc(out)
        return out, hidden



def prepare_sequence(tokens, word_to_idx):
    idxs = [word_to_idx[w] for w in tokens]
    return torch.tensor(idxs, dtype=torch.long)


def generate(model, size, prefix_str, word_to_idx, idx_to_word, eos):
    model.eval()

    tokens = prefix_str.split()
    output_words = []

    # Build the initial hidden from the given prefix
    X = prepare_sequence(tokens, word_to_idx)
    out, hidden = model(X)


    def generate_letter(logits):
        p = torch.nn.functional.softmax(logits, dim=0).detach().numpy()
        word_idx = np.random.choice(vocab_size, p=p)

        return word_idx, idx_to_word[word_idx]

    for _ in range(size):
        # use the last out logits to generate a new letter
        word_idx, word = generate_letter(out[-1])

        if word == eos:
            break

        output_words.append(word)

        word_idx = torch.tensor([word_idx], dtype=torch.long)
        out, hidden = model(word_idx, hidden)

    return ' '.join(output_words)




if __name__ == '__main__':
    EOS = '<EOS>'
    sequences = [
        "Get Skilled in Machine Learning",
        "By CS-Get Skilled Academy",
        "Instructor Mostafa Saad Ibrahim"
    ]

    words = [word for sentence in sequences for word in sentence.split()]
    words.append(EOS)
    vocab = set(words)

    # Word to index mapping
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}
    vocab_size = len(vocab)

    # Prepare the model and optimizer
    embedding_dim = 50
    hidden_dim = 128
    n_layers = 1
    batch_size = 1
    n_epochs = 10        # way less epochs than char generator
    learning_rate = 0.01

    model = WordRNN(vocab_size, embedding_dim, hidden_dim, n_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training the model
    model.train()
    for epoch in range(n_epochs):
        for data in sequences:
            words = data.split()
            X = prepare_sequence(words, word_to_idx)
            words.append(EOS)
            del words[0]
            y = prepare_sequence(words, word_to_idx)
            optimizer.zero_grad()
            output, hidden = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            answers = torch.max(output, dim=1)[1]
            train_acc = torch.sum(answers == y) / y.shape[0]

        print(f'Epoch {epoch+1}, Loss: {loss.item():.3f} - Accuracy: {train_acc.item():.2f}')

    # Generate some text that starts with this prefix
    print(generate(model, 50, 'Get Skilled', word_to_idx, idx_to_word, EOS))
    print(generate(model, 50, 'By', word_to_idx, idx_to_word, EOS))
    print(generate(model, 50, 'Instructor', word_to_idx, idx_to_word, EOS))
    print(generate(model, 50, 'Academy', word_to_idx, idx_to_word, EOS))
    # don't use a word not in dict

