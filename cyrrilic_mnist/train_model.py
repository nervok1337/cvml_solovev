from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from collections import defaultdict
import copy
import numpy as np


class CyrillicDataset(Dataset):
    def __init__(self, root_dir, transform=None, seed=42):
        self.root = Path(root_dir)
        self.transform = transform
        self.seed = seed

        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = [
            (p, self.class_to_idx[p.parent.name])
            for p in self.root.rglob("*.png")
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = Image.open(path).convert("RGBA")
        image = img.getchannel('A')

        if self.transform:
            image = self.transform(image)

        return image, label

    def split(self, train_ratio=0.8, train_transform=None, test_transform=None):
        class_samples = defaultdict(list)

        for path, label in self.samples:
            class_samples[label].append((path, label))

        train_samples = []
        test_samples = []

        rng = np.random.default_rng(self.seed)

        for label, samples in class_samples.items():
            rng.shuffle(samples)

            split = int(len(samples) * train_ratio)
            if split == 0:
                split = 1

            train_samples.extend(samples[:split])
            test_samples.extend(samples[split:])

        rng.shuffle(train_samples)
        rng.shuffle(test_samples)

        train = copy.deepcopy(self)
        test = copy.deepcopy(self)

        train.samples = train_samples
        test.samples = test_samples

        train.transform = train_transform or self.transform
        test.transform = test_transform or self.transform

        return train, test

class rCNN(nn.Module):
    def __init__(self, cnt_classes):
        super(rCNN, self).__init__()

        # block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2) # 28, 28 -> 14, 14

        # block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2) # 14, 14 -> 7, 7
        
        # block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2,2)

        # block result
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, cnt_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


train_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if __name__ == "__main__": 
    save_path = Path("./")
    dataset = CyrillicDataset("./Cyrillic/")

    d_train, d_test = dataset.split(0.8, train_transform, test_transform)

    len_classes = len(dataset.classes)
    model = rCNN(len_classes)

    batch_size = 128
    train_loader = DataLoader(d_train, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(d_test, batch_size=batch_size, shuffle=False, num_workers=4,persistent_workers=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params=}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epoch = 10
    train_loss = []
    train_accuracy = []

    model_path = save_path / "model.pth"
    for epoch in range(num_epoch):
        model.train()
        run_loss = 0.0
        total = 0
        correct = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        scheduler.step()
        epoch_loss = run_loss / len(train_loader)
        epoch_acc = 100 * (correct / total)
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_acc)
        print(f"Epoch {epoch}, {epoch_loss:=.3f}, {epoch_acc:=.3f}")
    torch.save(model.state_dict(), model_path)
    plt.figure()
    plt.subplot(121)
    plt.title("Loss")
    plt.plot(train_loss)
    plt.subplot(122)
    plt.title("Accuracy")
    plt.plot(train_accuracy)
    plt.savefig("train.png")
    plt.show()

