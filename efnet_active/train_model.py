from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import time
import numpy as np
from collections import deque
import cv2
import subprocess

class Buffer():
    def __init__(self, maxsize=16):
        self.frames = deque(maxlen=maxsize)
        self.labels = deque(maxlen=maxsize)
    def append(self, tensor, label):
        self.frames.append(tensor)
        self.labels.append(label)
    def __len__(self):
        return len(self.frames)
    def get_batch(self):
        images = torch.stack(list(self.frames))
        labels = torch.tensor(list(self.labels), dtype=torch.float32)
        return images, labels 

def build_model():
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b0(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False
    
    features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(features, 1)
    return model

def train(buffer):
    if len(buffer) < 10:
        return None
    model.train()
    images,labels = buffer.get_batch()
    optimizer.zero_grad()
    predictions = model(images).squeeze(1)
    loss = criterion(predictions,labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def predict(frame, thr=0.5):
    model.eval()
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        predicted = model(tensor).squeeze()
        prob = torch.sigmoid(predicted).item()
    label = "person" if prob > thr else "no_person"
    return label, prob

model = build_model()
model_path = Path("alexnet.pth")
if model_path.exists():
    model.load_state_dict(torch.load(model_path))
    print("Model loaded")

total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params=}")

# print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229,0.224,0.225])
])

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)


auto_exp = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
print(f"{auto_exp=}, {exposure=}")

buffer = Buffer()
count_labeled = 0

while True:
    _,frame = cap.read()
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if key == ord("q"):
        break
    elif key == ord("1"): #is person
        tensor = transform(image)
        buffer.append(tensor, 1.0)
        count_labeled += 1
    elif key == ord("2"): #no person
        tensor = transform(image)
        buffer.append(tensor, 0.0)
        count_labeled += 1
    elif key == ord("p"): #predict
        t = time.perf_counter()
        label, confidence = predict(frame)
        print(f"Elapsed time {time.perf_counter() - t}")
        print(label, confidence)
    elif key == ord("s"): #save model
        torch.save(model.state_dict(), model_path)
        print("Model saved")
    
    print(len(buffer), count_labeled)
    if count_labeled >= buffer.frames.maxlen:
        loss = train(buffer)
        if loss:
            print(f"Loss = {loss}")
        count_labeled = 0