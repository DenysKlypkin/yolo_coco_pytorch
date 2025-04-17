# set random seed
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from config import ANN_FILE, DATA_DIR
from dataset import COCODataset
from loss import YOLOLoss
from model import CustomYOLO


# def train():
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

coco_dataset = COCODataset(
    ann_file=ANN_FILE,
    img_dir=DATA_DIR,
)

print(f"Number of images: {len(coco_dataset)}")

train_loader = torch.utils.data.DataLoader(
    coco_dataset,
    batch_size=16,
    shuffle=True,
)
model = CustomYOLO(
    num_classes=coco_dataset.C,
).to(device)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = YOLOLoss()
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch}/{num_epochs}")
    for i, elem in enumerate(train_loader):
        print(f"Batch {i}/{len(train_loader)}")
        images = elem["image"].to(device)
        targets = elem["target"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss_value = loss(outputs, targets)
        loss_value.backward()
        optimizer.step()

        print(
            f"Epoch [{epoch}/{num_epochs}], Step [{i}], Loss: {loss_value.item():.4f}"
        )
