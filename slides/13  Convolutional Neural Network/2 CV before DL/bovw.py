# ChatGPT with some my fixes

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


transform = transforms.Compose([transforms.ToTensor()])
cifar10_train = CIFAR10(root='/home/moustafa/0hdd/research/ndatasets/cifar10', train=True, download=True, transform=transform)
cifar10_test = CIFAR10(root='/home/moustafa/0hdd/research/ndatasets/cifar10', train=False, download=True, transform=transform)


def convert(image):
    image = image.numpy().transpose((1, 2, 0))

    image = (image * 255).round().astype(np.uint8)
    return image


train_images = [convert(image) for image, _ in cifar10_train]
train_labels = [label for _, label in cifar10_train]
test_images = [convert(image) for image, _ in cifar10_test]
test_labels = [label for _, label in cifar10_test]

# Initialize SIFT feature extractor
sift = cv2.SIFT_create()


# Function to convert the images to grayscale and extract SIFT features
def extract_sift_features(images):
    descriptors_list = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        assert gray is not None
        kp, des = sift.detectAndCompute(gray, None)
        if des is not None:
            descriptors_list.append(des)
    descriptors = np.vstack(descriptors_list)
    return descriptors


# Extract SIFT features from training images
all_descriptors = extract_sift_features(train_images)

# K-Means to cluster the descriptors and create a vocabulary
k = 200  # number of visual words
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(all_descriptors)


# Function to build histograms of visual word occurrences
def build_histograms(images, kmeans):
    histograms = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        if des is not None:
            pred = kmeans.predict(des)
            hist = np.histogram(pred, bins=np.arange(k + 1), density=True)[0]
            histograms.append(hist)
        else:
            histograms.append(np.zeros(k))
    return np.array(histograms)


# Build histograms for the training and test sets
x_train_hist = build_histograms(train_images, kmeans)
x_test_hist = build_histograms(test_images, kmeans)

# Normalize the histograms
scaler = StandardScaler()
x_train_hist = scaler.fit_transform(x_train_hist)
x_test_hist = scaler.transform(x_test_hist)

# Convert the histograms and labels into PyTorch datasets
train_dataset = TensorDataset(torch.tensor(x_train_hist, dtype=torch.float32),
                              torch.tensor(train_labels, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(x_test_hist, dtype=torch.float32),
                             torch.tensor(test_labels, dtype=torch.long))

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define a simple neural network classifier
class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(k, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


# Initialize the classifier, loss function, and optimizer
model = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Function to perform training
def train(model, criterion, optimizer, train_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        for histograms, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(histograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# Train the model
train(model, criterion, optimizer, train_loader)


# Function to perform testing
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for histograms, labels in test_loader:
            outputs = model(histograms)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')


# Test the model
test(model, test_loader)
