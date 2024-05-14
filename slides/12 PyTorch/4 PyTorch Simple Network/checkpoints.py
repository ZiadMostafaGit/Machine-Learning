import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)

    def forward(self, x):
        return torch.relu(self.fc1(x))


model = SimpleNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epoch = 10
loss = 0.5

checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}

# Do some training and Save the checkpoint
torch.save(checkpoint, 'checkpoint.pth')


# Initialize model and optimizer
model = SimpleNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

checkpoint = torch.load('checkpoint.pth')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

epoch = checkpoint['epoch']
loss = checkpoint['loss']










# Load checkpoint
checkpoint = torch.load('checkpoint.pth')

# Restore states from checkpoint
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# Don't forget to set the model to the appropriate mode
model.eval()
# - or -
# model.train()
