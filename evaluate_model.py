import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from PIL import Image, ImageDraw
import random
from custom_dataset import PennFudanDataset  # your dataset class

# ---------------- Paths ----------------
IMAGES_DIR = r"C:\Users\tskii\OneDrive\Desktop\IITB\2nd year\Sem 1\SeDriCa\Q2 perception\dataset\PNGImages"
ANNOTATIONS_DIR = r"C:\Users\tskii\OneDrive\Desktop\IITB\2nd year\Sem 1\SeDriCa\Q2 perception\dataset\Annotation"

# ---------------- Device ----------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ---------------- Model ----------------
num_classes = 2  # background + pedestrian
model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load("fasterrcnn_pedestrian.pth", map_location=device))
model.to(device)
model.eval()

# ---------------- Dataset & Loader ----------------
def get_transform(train=False):
    transforms = [T.ToTensor()]
    return T.Compose(transforms)

test_dataset = PennFudanDataset(
    images_dir=IMAGES_DIR,
    annotations_dir=ANNOTATIONS_DIR,
    transforms=get_transform(train=False)
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ---------------- IoU & Metrics ----------------
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou

all_ious = []
all_precisions = []
all_recalls = []

for images, targets in test_loader:
    images = list(img.to(device) for img in images)
    outputs = model(images)

    for target, output in zip(targets, outputs):
        gt_boxes = target['boxes'].cpu().numpy()
        pred_boxes = output['boxes'].detach().cpu().numpy()
        scores = output['scores'].detach().cpu().numpy()
        
        # threshold predictions
        pred_boxes = pred_boxes[scores >= 0.5]

        # compute IoU for each ground truth box
        ious = []
        for gt in gt_boxes:
            iou_max = 0
            for pred in pred_boxes:
                iou = compute_iou(gt, pred)
                if iou > iou_max:
                    iou_max = iou
            ious.append(iou_max)
        all_ious.extend(ious)

        # precision/recall
        tp = sum(iou >= 0.5 for iou in ious)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        all_precisions.append(precision)
        all_recalls.append(recall)

mean_iou = sum(all_ious) / len(all_ious)
mean_precision = sum(all_precisions) / len(all_precisions)
mean_recall = sum(all_recalls) / len(all_recalls)

print(f"Mean IoU: {mean_iou:.4f}")
print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")

# ---------------- Visualize random test images ----------------
num_visualize = 3  # number of random images to visualize
random_indices = random.sample(range(len(test_dataset)), num_visualize)

for idx in random_indices:
    img, target = test_dataset[idx]
    with torch.no_grad():
        pred = model([img.to(device)])[0]

    img_pil = T.ToPILImage()(img)
    draw = ImageDraw.Draw(img_pil)

    for box, score in zip(pred['boxes'], pred['scores']):
        if score >= 0.5:
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
    
    img_pil.show()
