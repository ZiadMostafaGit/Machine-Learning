import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Load pre-trained AlexNet model
model = torchvision.models.alexnet(pretrained=True)

# Get the first 3 Conv2D compoenets
features = list(model.features.children())[:8]

# Freeze their layers (never update weights)
for layer in features:
    for param in layer.parameters():
        param.requires_grad = False

# Extend the model with custom layers
features.extend([
    nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.AdaptiveAvgPool2d((6, 6))
])

# Replace the feature layers
model.features = nn.Sequential(*features)

# Add custom classifier layers (fully connected layers)
model.classifier = nn.Sequential(
    nn.Dropout(),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 10)
)

# Hyperparameters
learning_rate = 0.001
batch_size = 64
epochs = 10

# Transform and Data
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

data_root = '/home/moustafa/0hdd/research/ndatasets/cifar10'
trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize the loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# Training
for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}")

# Testing
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
