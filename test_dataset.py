import torch
import torchvision
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader
from custom_dataset import PennFudanDataset
import torchvision.transforms as T
import numpy as np
import cv2

# ---------------- Paths ----------------
IMAGES_DIR = r"C:\Users\tskii\OneDrive\Desktop\IITB\2nd year\Sem 1\SeDriCa\Q2 perception\dataset\PNGImages"
ANNOTATIONS_DIR = r"C:\Users\tskii\OneDrive\Desktop\IITB\2nd year\Sem 1\SeDriCa\Q2 perception\dataset\Annotation"

# ---------------- Dataset ----------------
transform = T.Compose([T.ToTensor()])
test_dataset = PennFudanDataset(images_dir=IMAGES_DIR, annotations_dir=ANNOTATIONS_DIR, transforms=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ---------------- Load model ----------------
num_classes = 2
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("fasterrcnn_pedestrian.pth"))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

# ---------------- Evaluation ----------------
all_scores = []
all_labels = []
all_preds = []

with torch.no_grad():
    for images, targets in test_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for i, output in enumerate(outputs):
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            boxes = output['boxes'].cpu().numpy()

            all_scores.extend(scores)
            all_labels.extend([1] * len(scores))  # Pedestrian class only
            all_preds.extend(scores)  # Confidence as prediction

# ---------------- Precision-Recall ----------------
precision, recall, _ = precision_recall_curve(all_labels, all_preds)
ap_score = average_precision_score(all_labels, all_preds)

plt.figure()
plt.plot(recall, precision, marker='.', label=f'AP={ap_score:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# ---------------- Visualization ----------------
def visualize_predictions(img_tensor, boxes, scores, threshold=0.5):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8).copy()

    for box, score in zip(boxes, scores):
        if score >= threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Show predictions for first few images
for images, _ in list(test_loader)[:5]:
    with torch.no_grad():
        output = model([images[0].to(device)])[0]
        visualize_predictions(images[0], output['boxes'], output['scores'])
