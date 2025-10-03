import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from custom_dataset import PennFudanDataset  # Ensure correct import path
import os

# ---------------- Paths ----------------
IMAGES_DIR = r"C:\Users\tskii\OneDrive\Desktop\IITB\2nd year\Sem 1\SeDriCa\Q2 perception\dataset\PNGImages"
ANNOTATIONS_DIR = r"C:\Users\tskii\OneDrive\Desktop\IITB\2nd year\Sem 1\SeDriCa\Q2 perception\dataset\Annotation"

# ---------------- Transforms ----------------
def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# ---------------- Dataset ----------------
dataset = PennFudanDataset(
    images_dir=IMAGES_DIR,
    annotations_dir=ANNOTATIONS_DIR,
    transforms=get_transform(train=True)
)

# ---------------- Train/Test split ----------------
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ---------------- Model ----------------
num_classes = 2  # 1 pedestrian + background
model = fasterrcnn_resnet50_fpn(weights='DEFAULT')

# Replace classifier head
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# ---------------- Optimizer & Scheduler ----------------
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# ---------------- Training loop ----------------
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    i = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i}/{len(train_loader)}] Loss: {losses.item():.4f}")
        i += 1

    lr_scheduler.step()

# ---------------- Save the model ----------------
torch.save(model.state_dict(), "fasterrcnn_pedestrian.pth")
print("Training complete and model saved!")
