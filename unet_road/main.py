import torch
from pathlib import Path
import matplotlib.pyplot as plt

from unet_road import RoadsDataset, UNet


path = Path("roads")

ds = RoadsDataset(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(features=[8, 16, 32, 64]).to(device)
model.load_state_dict(torch.load("unet_road.pth", map_location=device))
model.eval()

print(sum(p.numel() for p in model.parameters()))

image, mask = ds[2]

with torch.no_grad():
    x = image.unsqueeze(0).float().to(device)
    pred = model(x)
    pred = torch.sigmoid(pred)
    pred_mask = (pred > 0.5).float()

image_show = image.permute(1, 2, 0).numpy()
mask_show = mask.squeeze(0).numpy()
pred_show = pred_mask.squeeze(0).squeeze(0).cpu().numpy()
diff_show = abs(mask_show - pred_show)

plt.figure(figsize=(20, 6))

plt.subplot(1, 4, 1)
plt.imshow(image_show)
plt.title("image")

plt.subplot(1, 4, 2)
plt.imshow(mask_show, cmap="gray")
plt.title("true mask")

plt.subplot(1, 4, 3)
plt.imshow(pred_show, cmap="gray")
plt.title("pred mask")

plt.subplot(1, 4, 4)
plt.imshow(diff_show, cmap="gray")
plt.title("difference")

plt.savefig("result.png")
plt.show()