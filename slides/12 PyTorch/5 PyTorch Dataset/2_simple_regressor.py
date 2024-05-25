import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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


class CustomRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


input_dim = 65
model = SimpleNet(input_dim)

optimizer = optim.AdamW(model.parameters(), lr=0.001)
# try lr=0.01
#optimizer = optim.SGD(model.parameters(), lr=0.0001)

x_train = torch.rand(1000, input_dim)   # 1000 examples
y_train = 5 * torch.sum(x_train, dim=1)

criterion = nn.MSELoss()

cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")
model = model.to(device)


train_dataset = CustomRegressionDataset(x_train, y_train)
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    for batch_idx, (X_batch, y_batch) in enumerate(train_data_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Batch Loss: {loss.item()}")



x_test = torch.rand(10, input_dim)
y_test = 5 * torch.sum(x_test, dim=1)

test_dataset = CustomRegressionDataset(x_train, y_train)
# avoid shuffling in testing for easy debugging
train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for batch_idx, (X_batch, y_batch) in enumerate(train_data_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_predict = model(X_batch)
        print(f"Prediction: {y_predict} vs gt {y_batch}", )
