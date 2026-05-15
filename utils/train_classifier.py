import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict


IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {
    0: "pricetag",
    1: "price_block",
    2: "price_digits",
    3: "product_name",
    4: "qr_code",
    5: "barcode",
    6: "promo_sticker",
}

LABELS_DIR = Path("data/yolov12n/labels_full/train")
IMAGES_DIR = Path("data/yolov12n/images_full/train")
MODEL_SAVE_PATH = Path("models/region_classifier.pth")


class RegionDataset(Dataset):
    def __init__(self, labels_dir, images_dir, transform=None):
        self.samples = []
        self.transform = transform

        for lf in sorted(labels_dir.glob("*.txt")):
            img_path = images_dir / (lf.stem + ".jpg")
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            with open(lf) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    if cls == 0:
                        continue
                    x_c, y_c, bw, bh = map(float, parts[1:5])

                    x1 = int((x_c - bw / 2) * w)
                    y1 = int((y_c - bh / 2) * h)
                    x2 = int((x_c + bw / 2) * w)
                    y2 = int((y_c + bh / 2) * h)

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                        continue

                    self.samples.append((crop, cls))

        print(f"Loaded {len(self.samples)} region crops")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        crop, cls = self.samples[idx]
        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = crop.astype(np.float32) / 255.0
        crop = torch.from_numpy(crop).permute(2, 0, 1)

        if self.transform:
            crop = self.transform(crop)

        return crop, cls


class RegionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def train():
    print(f"Device: {DEVICE}")

    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    ])

    dataset = RegionDataset(LABELS_DIR, IMAGES_DIR, transform=transform)

    class_counts = defaultdict(int)
    for _, cls in dataset.samples:
        class_counts[cls] += 1
    print(f"Class distribution: {dict(sorted(class_counts.items()))}")

    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = RegionClassifier(num_classes=7).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        scheduler.step()

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Val Loss: {val_loss / len(val_loader):.4f} | "
              f"Val Acc: {accuracy:.2f}%")

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Model saved (acc={accuracy:.2f}%)")

    print(f"\nBest accuracy: {best_acc:.2f}%")
    print(f"Model saved: {MODEL_SAVE_PATH}")


def predict(crop, model=None):
    if model is None:
        model = RegionClassifier(num_classes=7)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.resize(crop_rgb, (IMG_SIZE, IMG_SIZE))
    tensor = torch.from_numpy(crop_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)

    return int(predicted[0])


if __name__ == "__main__":
    train()
