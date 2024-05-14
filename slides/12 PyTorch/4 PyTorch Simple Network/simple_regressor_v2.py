


import torch
import torch.nn as nn
import torch.optim as optim



class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, 1)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    input_dim = 65
    model = SimpleNet(input_dim)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    # try lr=0.01
    #optimizer = optim.SGD(model.parameters(), lr=0.0001)

    x_train = torch.rand(1000, input_dim)   # 1000 examples
    y_train = 5 * torch.sum(x_train, dim=1)

    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(500):
        # Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train)

        # Zero gradients, backward pass, optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print(f"Epoch {epoch+1}, Batch Loss: {loss.item()}")
        # observe this loss sums all the batch.
        # you can average for a single example average loss


    # Save entire model - limited loading
    torch.save(model, 'model_v2.pth')

    # Save model weights (recommended): save only the model's parameters
    torch.save(model.state_dict(), 'model_weights_v2.pth')


    # Evaluation
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for i in range(10):
            x_test = torch.rand(1, input_dim)  # 10 test samples
            y_test = model(x_test)
            print(f"Prediction: {y_test} vs gt {5 * torch.sum(x_test, dim=1)}", )
