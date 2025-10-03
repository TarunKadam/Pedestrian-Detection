import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torchvision.transforms as T
import re

class PennFudanDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, masks_dir=None, transforms=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.masks_dir = masks_dir
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
        mask_files = []

        if os.path.exists(annot_file):
            with open(annot_file, "r") as f:
                for line in f:
                    line = line.strip()
                    # Parse bounding box lines
                    if line.startswith("Bounding box"):
                        # Extract numbers using regex
                        m = re.search(r"\((\d+), (\d+)\) - \((\d+), (\d+)\)", line)
                        if m:
                            x_min, y_min, x_max, y_max = map(float, m.groups())
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(1)  # pedestrian = 1
                    # Parse mask lines if masks_dir is given
                    if self.masks_dir and line.startswith("Pixel mask"):
                        mask_path_str = line.split(":")[1].strip().strip('"')
                        mask_file = os.path.basename(mask_path_str)
                        mask_files.append(mask_file)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        # Load masks if available
        if self.masks_dir and mask_files:
            masks = []
            for mask_file in mask_files:
                mask_path = os.path.join(self.masks_dir, mask_file)
                mask = Image.open(mask_path).convert("L")
                mask = torch.as_tensor(np.array(mask), dtype=torch.uint8)
                masks.append(mask)
            if masks:
                target["masks"] = torch.stack(masks)

        if self.transforms:
            img = self.transforms(img)

        return img, target

# ---------------- Example usage ----------------
transform = T.Compose([T.ToTensor()])
dataset = PennFudanDataset(
    images_dir=r"C:\Users\tskii\OneDrive\Desktop\IITB\2nd year\Sem 1\SeDriCa\Q2 perception\dataset\PNGImages",
    annotations_dir=r"C:\Users\tskii\OneDrive\Desktop\IITB\2nd year\Sem 1\SeDriCa\Q2 perception\dataset\Annotation",
    masks_dir=r"C:\Users\tskii\OneDrive\Desktop\IITB\2nd year\Sem 1\SeDriCa\Q2 perception\dataset\PedMasks",
    transforms=transform
)

img, target = dataset[0]
print(img.shape)
print(target)

