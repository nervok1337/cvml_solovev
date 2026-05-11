import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn
import torch.optim as optim


class ImageDataset(Dataset):
    def __init__(self, n=200, size=256, variant=1):
        super().__init__()
        self.n = n
        self.size = size
        self.variant = variant

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.font = ImageFont.load_default()
        self.fixed_text = "ABC"
        self.fixed_text_len = 3

    def __len__(self):
        return self.n

    def make_random_text(self, length):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        text = ""

        for _ in range(length):
            text += letters[np.random.randint(0, len(letters))]

        return text

    def __getitem__(self, idx):
        image = Image.new("L", (self.size, self.size), color=255)
        draw = ImageDraw.Draw(image)

        # fixtext posrnd
        if self.variant == 1:
            text = self.fixed_text
            x = np.random.randint(10, self.size - 60)
            y = np.random.randint(10, self.size - 30)

        # rndtext fixlen fixpos
        elif self.variant == 2:
            text = self.make_random_text(self.fixed_text_len)
            x = 30
            y = 30

        # rndtext rndlen fixpos
        elif self.variant == 3:
            text_len = np.random.randint(1, 8)
            text = self.make_random_text(text_len)
            x = 30
            y = 30

        # rndtext rndlen rndpos
        elif self.variant == 4:
            text_len = np.random.randint(1, 8)
            text = self.make_random_text(text_len)
            x = np.random.randint(10, self.size - 60)
            y = np.random.randint(10, self.size - 30)
        
        draw.text((x, y), text, fill=0, font=self.font)

        tensor = self.transform(image)

        return tensor, tensor


class Encoder(nn.Module):
    def __init__(self, latent_size=512):
        super().__init__()
        self.latent_size = latent_size
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.bottleneck = nn.Linear(256 * 16 * 16, latent_size)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_size=512):
        super().__init__()
        self.bottleneck = nn.Linear(latent_size, 256 * 16 * 16)
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.bottleneck(x)
        x = x.view(x.size(0), 256, 16, 16)
        x = self.features(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    ds = ImageDataset(2000, 256, 4)

    dataloader = DataLoader(ds, batch_size=64, shuffle=True)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

    encoder.train()
    decoder.train()

    epoch = 15
    for epoch in range(epoch):
        epoch_loss = 0.0

        for imgs, _ in dataloader:
            imgs = imgs.to(device)

            optimizer.zero_grad()

            latent = encoder(imgs)
            output = decoder(latent)

            loss = criterion(imgs, output)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"{epoch=}, {avg_loss=:.2f}")

    torch.save(encoder.state_dict(), "encoder.pth")
    torch.save(decoder.state_dict(), "decoder.pth")
