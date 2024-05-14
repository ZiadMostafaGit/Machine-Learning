


from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])



transform2 = transforms.Compose([
    transforms.ToTensor(),  #               # Normalize to [0, 1]
    transforms.Lambda(lambda x: 2 * x - 1)  # Normalize to [-1, 1]
])


