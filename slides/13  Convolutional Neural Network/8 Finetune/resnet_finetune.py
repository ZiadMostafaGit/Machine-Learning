import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader

# Data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

batch_size = 64
data_root = '/home/moustafa/0hdd/research/ndatasets/cifar10'

trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


original_model = resnet50(pretrained=True)

# Remove the last 4 main blocks (layer3, layer4, avgpool, fc)
layers = list(original_model.children())[:-4]
# By debugging, our last layer's shape: 512 * 28 * 28

# Create truncated model
truncated_model = nn.Sequential(*layers)


# Extend the truncated model with custom layers
class ExtendedModel(nn.Module):
    def __init__(self, backbone):
        super(ExtendedModel, self).__init__()
        self.backbone = backbone
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 28 * 28, 265),
            nn.ReLU(),
            nn.Linear(265, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

model = ExtendedModel(truncated_model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)  # Only optimize unfrozen params

# Training loop
for epoch in range(5):  # Number of epochs
    for batch_idx, (data, target) in enumerate(trainloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch} Batch: {batch_idx} Loss: {loss.item()}")

# Model evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in testloader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"Accuracy: {(correct / total) * 100}%")
