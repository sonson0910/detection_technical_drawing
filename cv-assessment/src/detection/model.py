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
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead


def get_model(num_classes=4, pretrained=True, min_size=1200, max_size=2000):
    """
    Build Faster R-CNN model with ResNet-50 FPN v2 backbone.
    Optimized for engineering drawing objects:
      - PartDrawing: large, variable aspect ratio
      - Note: wide+thin (avg 304x79) or small annotations 
      - Table: wide, medium-to-tall (avg 503x144)
    """
    if pretrained:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            min_size=min_size,
            max_size=max_size
        )
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=None,
            min_size=min_size,
            max_size=max_size
        )

    # (Reverted custom Anchor Generator because best_map_model_backup.pth uses default Anchors)

    # 2. Replace the classifier head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 3. Tuning Hyperparameters
    model.roi_heads.detections_per_img = 150  # Lots of objects in engineering drawings
    model.roi_heads.score_thresh = 0.05
    model.roi_heads.nms_thresh = 0.45  # Reduce overlapping boxes

    return model
