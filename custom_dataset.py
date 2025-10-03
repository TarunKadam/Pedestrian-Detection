import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import re
import torchvision.transforms as T

class PennFudanDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transforms=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.image_files = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, image_file)
        img = Image.open(img_path).convert("RGB")

        annot_file = os.path.join(self.annotations_dir, os.path.splitext(image_file)[0] + ".txt")
        boxes = []
        labels = []

        if os.path.exists(annot_file):
            with open(annot_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Bounding box"):
                        m = re.search(r"\((\d+), (\d+)\) - \((\d+), (\d+)\)", line)
                        if m:
                            x_min, y_min, x_max, y_max = map(float, m.groups())
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(1)  # pedestrian class

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target
