# pip install tensorboard

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

from torchvision import transforms, datasets
from torch.utils.data import random_split


# classifier on 10 classes
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        return self.fc(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='/home/moustafa/0hdd/research/ndatasets/mnist/full', train=True, download=True, transform=transform)

train_set, val_set = random_split(dataset, [55000, 5000])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

# Create the model, send it to device, define loss function and optimizer
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters())


writer = SummaryWriter('runs/mnist_experiment_1')


def train(num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:  # Log every 100 mini-batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

                # Log the running loss averaged per mini-batch
                # epoch * len(train_loader) + i = global step count
                writer.add_scalar('training_loss', running_loss / 100, epoch * len(train_loader) + i)

                # Log a random batch of images
                img_grid = torchvision.utils.make_grid(inputs[:4].cpu().data)
                writer.add_image('four_mnist_images', img_grid, epoch * len(train_loader) + i)

                running_loss = 0.0

        # Validation loss
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Optionally, also log images for the validation set
                # (You may want to do this less frequently or only once)
                if epoch == num_epochs - 1:  # Only on last epoch
                    img_grid = torchvision.utils.make_grid(inputs[:4].cpu().data)
                    writer.add_image('four_validation_mnist_images', img_grid)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        print(f'Validation loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

        # Log validation loss and accuracy
        writer.add_scalar('validation_loss', avg_val_loss, epoch)
        writer.add_scalar('validation_accuracy', val_accuracy, epoch)

    print('Finished Training')


# Call the training loop
train(num_epochs=5)

# Close the TensorBoard writer
writer.close()

