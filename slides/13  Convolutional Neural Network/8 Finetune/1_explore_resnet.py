import torch
import torchvision.models as models
import torch.nn as nn

model = models.resnet50(pretrained=True)

print("\nEntire Architecture:")
print(model)

print("\nFirst Convolutional Layer:")
print(model.conv1)

print("\nFirst Residual Block in Layer 1:")
print(model.layer1[0])

# Generate a random image (3 channels, 224x224 dimensions)
random_image = torch.randn(1, 3, 224, 224)

# Perform forward pass (inference)
with torch.no_grad():
    output = model(random_image)
    print("\nOutput shape after Forward Pass:")
    print(output.shape)  # Should be [1, 1000] indicating the 1000 classes in ImageNet

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last fully connected layer (classification layer)
for param in model.fc.parameters():
    param.requires_grad = True

# Check if layers are frozen
print("\nIs grad enabled for first Convolutional Layer?:", model.conv1.weight.requires_grad)
print("Is grad enabled for Last Fully Connected Layer?:", model.fc.weight.requires_grad)


weights_lst = list(model.parameters())
print(len(weights_lst))
