import torch
from torch.utils.data import DataLoader

from models.yolo import YOLO
from utils.dataset import YOLODataset
from utils.loss import YOLOLoss


# Configuration
IMAGE_SIZE = 448
GRID_SIZE = 7
NUM_CLASSES = 3

BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001


# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Training device:", device)


# Dataset
train_dataset = YOLODataset(
    image_dir="dataset/images/train",
    label_dir="dataset/labels/train",
    image_size=IMAGE_SIZE,
    grid_size=GRID_SIZE,
    num_classes=NUM_CLASSES
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# Model
model = YOLO(
    num_classes=NUM_CLASSES,
    grid_size=GRID_SIZE
).to(device)


# Loss
criterion = YOLOLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# Training loop
for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for images, targets in train_loader:

        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        predictions = model(images)

        # Calculate loss
        loss = criterion(predictions, targets)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / max(len(train_loader), 1)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f}"
    )

    # Save model
    torch.save(
        model.state_dict(),
        "yolo_model.pth"
    )

print("Training completed!")
