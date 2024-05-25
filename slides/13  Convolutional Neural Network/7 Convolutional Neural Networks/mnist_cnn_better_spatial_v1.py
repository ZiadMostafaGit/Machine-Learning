import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        # input is 1x28x28, so input channels = 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding='valid')

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Average pool the last 2 dimensions into [B, C, 1, 1]
        self.adaptivepool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Linear(128, 10)
        self.activation = nn.functional.relu

    def forward(self, x):
        x = self.conv1(x)  # [B, 32, 28, 28]
        x = self.activation(x)
        x = self.pool(x)  # [B, 32, 14, 14]   28/2 = 14

        x = self.conv2(x)
        x = self.activation(x)  # [B, 64, 14, 14]
        x = self.pool(x)  # [B, 64, 7, 7]     14/2 = 7

        x = self.conv3(x)  # [B, 128, 5, 5]
        x = self.activation(x)
        x = self.adaptivepool(x)  # [B, 128, 1, 1]

        x = x.view(x.size(0), -1)  # [B, 128]
        x = self.fc(x)  # [B, 10]

        return x


def get_data_loaders(path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.MNIST(root=path, train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.MNIST(root=path, train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)

    return trainloader, testloader


if __name__ == '__main__':
    root = '/home/moustafa/0hdd/research/ndatasets/mnist/data'
    trainloader, testloader = get_data_loaders(root)
    model = MnistCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Model Training
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in trainloader:
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(trainloader)}")

    print("Finished Training")

    # Model Evaluation
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the test set: {accuracy}%")
