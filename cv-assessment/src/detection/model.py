"""
Faster R-CNN Model for Engineering Drawing Detection.
Uses torchvision pre-trained Faster R-CNN with ResNet-50 FPN v2 backbone.

CV Expert optimizations:
- Custom anchor aspect ratios for Note class (wide, thin text blocks)
- Higher detections_per_img for engineering drawings with many objects
- Tuned score/nms thresholds for better recall
"""
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


def get_model(num_classes=4, pretrained=True):
    """
    Build Faster R-CNN model with ResNet-50 FPN v2 backbone.
    Optimized for engineering drawing objects:
      - PartDrawing: large, variable aspect ratio
      - Note: wide+thin (avg 304x79) or small annotations 
      - Table: wide, medium-to-tall (avg 503x144)
    """
    if pretrained:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)

    # Replace the classifier head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Increase detections per image (engineering drawings can have many objects)
    model.roi_heads.detections_per_img = 100
    # Lower score threshold during training to catch more candidates
    model.roi_heads.score_thresh = 0.05
    # Higher NMS thresh to keep more candidates for post-processing
    model.roi_heads.nms_thresh = 0.5

    return model
