import torch
from torch.utils.data import DataLoader
from train_model import CyrillicDataset, rCNN, test_transform

dataset = CyrillicDataset("./Cyrillic/")

_, test_dataset = dataset.split(0.6, test_transform=test_transform)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

len_classes = len(dataset.classes)
model = rCNN(len_classes)

model.load_state_dict(torch.load("model.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")