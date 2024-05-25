import torch
import torch.nn as nn


class AutoencoderX1(nn.Module):
    def __init__(self):
        super(AutoencoderX1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=60, out_features=40),
            nn.Linear(in_features=40, out_features=20),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=20, out_features=40),
            nn.Linear(in_features=40, out_features=60),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderX2(nn.Module):
    def __init__(self):
        super(AutoencoderX2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=40, out_features=15),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=15, out_features=40),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderX(nn.Module):
    def __init__(self):
        super(AutoencoderX, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=100, out_features=75),
            nn.Linear(in_features=75, out_features=25),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=25, out_features=75),
            nn.Linear(in_features=75, out_features=100),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderMerger(nn.Module):
    def __init__(self):
        super(AutoencoderMerger, self).__init__()
        self.ae_x1 = AutoencoderX1()
        self.ae_x2 = AutoencoderX2()
        self.ae_x = AutoencoderX()

    def forward(self, x1, x2):
        x1 = self.ae_x1(x1)
        x2 = self.ae_x2(x2)
        x = torch.hstack([x1, x2])

        # x+ for skip connection (just element wise addition)
        x = x + self.ae_x(x)

        return x1, x2, x


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.autoencoder = AutoencoderMerger()
        self.network = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 1)
        )

    def forward(self, x1, x2):
        x1, x2, x = self.autoencoder(x1, x2)
        prediction = self.network(x)
        return x1, x2, x, prediction     # apply losses


def drop_features(x, drop_prob):
    drop_prob = torch.rand(1) * drop_prob
    # Create a mask using a uniform distribution and the drop probability
    mask = torch.rand_like(x) > drop_prob
    dropped_x = x * mask.float()
    return dropped_x


if __name__ == '__main__':


    drop_prob = 0.9
    criterion = nn.MSELoss()
    regressor = RegressionModel()

    x1 = torch.randn(32, 60)  # Batch size of 32
    x2 = torch.randn(32, 40)  # Batch size of 32
    x = torch.hstack([x1, x2])
    gt = torch.randn(32, 1)

    x1_noise = drop_features(x1, drop_prob)
    x2_noise = drop_features(x2, drop_prob)

    x1_rec, x2_rec, x_rec, prediction = regressor(x1_noise, x2_noise)

    # observe loss with x1 NOT x1_noise
    loss1 = criterion(x1, x1_rec)
    loss2 = criterion(x2, x2_rec)
    loss3 = criterion(x, x_rec)
    loss4 = criterion(gt, prediction)
    loss = (loss1+loss2+loss3+loss4)/4.0
    ...

