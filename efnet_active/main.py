from pathlib import Path
import time

import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


def build_model():
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b0(weights=weights)

    features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(features, 1)

    return model


def predict(model, frame, transform, threshold=0.5, device="cpu"):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor).squeeze()
        prob = torch.sigmoid(logits).item()

    label = "person" if prob > threshold else "no_person"
    return label, prob


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_path = Path(__file__).resolve().parent / "model.pth"
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    model = build_model()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print("Model loaded")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Controls:")
    print("p - predict")
    print("q - quit")

    last_label = "no_person"
    last_prob = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break

        frame_to_show = frame.copy()
        text = f"{last_label} | {last_prob:.4f}"

        cv2.imshow("Camera", frame_to_show)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            start = time.perf_counter()
            last_label, last_prob = predict(model, frame, transform, device=device)
            elapsed = time.perf_counter() - start

            print(f"Prediction: {last_label}")
            print(f"Confidence: {last_prob:.4f}")
            print(f"Elapsed time: {elapsed:.4f} sec")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()