import torch
from torch.utils.data import Dataset


class CustomRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class CustomImageDatasetV2(Dataset):
    def __init__(self, image_paths, labels, metadata, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        meta = self.metadata[idx]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label, 'metadata': meta}
        return sample


from torch.utils.data import DataLoader

X = torch.rand(1000, 65)
y = 5 * torch.sum(X, dim=1)

dataset = CustomRegressionDataset(X, y)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    for batch_idx, (X_batch, y_batch) in enumerate(data_loader):
        print(X_batch.shape)




