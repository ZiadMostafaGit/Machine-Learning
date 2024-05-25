import torch
import torchvision.models as models
import torch.nn as nn

# Load a pre-trained AlexNet model
alexnet_model = models.alexnet(pretrained=True)
print("Full AlexNet model:")
print(alexnet_model)
print("\n")

# Exploring model features (Convolutional layers)
print("Feature part of the model (Convolutional layers):")
print(alexnet_model.features)
print("\n")

# Exploring classifier part (Fully connected layers)
print("Classifier part of the model (Fully Connected layers):")
print(alexnet_model.classifier)
print("\n")

# Extracting a single layer
single_layer = alexnet_model.features[0]
print("Single layer (First Convolutional Layer):")
print(single_layer)
print("\n")

# Extracting multiple layers
multiple_layers = nn.Sequential(*list(alexnet_model.features)[:4])
print("Multiple layers (First 4 layers):")
print(multiple_layers)
print("\n")

# Run a simple forward pass with a dummy image
# Create a random 3x224x224 image (Batch size=1)
dummy_image = torch.randn(1, 3, 224, 224)

# Forward pass through the full AlexNet model
output_full_model = alexnet_model(dummy_image)
print("Output shape from full model:", output_full_model.shape)

# Forward pass through the feature part
output_features = alexnet_model.features(dummy_image)
print("Output shape after going through feature layers:", output_features.shape)

# Forward pass through the first 4 layers
output_multiple_layers = multiple_layers(dummy_image)
print("Output shape after going through first 4 layers:", output_multiple_layers.shape)
