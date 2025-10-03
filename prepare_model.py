import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- Device ---
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --- Load Faster R-CNN with ResNet-50 backbone pretrained on COCO ---
model = fasterrcnn_resnet50_fpn(pretrained=True)

# --- Modify the classifier for your dataset ---
# COCO has 91 classes; we need only 2 (background + pedestrian)
num_classes = 2  # 1 class (pedestrian) + background

# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the pre-trained head with a new one for pedestrian detection
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# --- Move model to device ---
model.to(device)

# --- Optional: print model summary for verification ---
print(model)

