# 🏗️ Engineering Drawing Object Detection & OCR System

**Technical Assessment — Sotatek Computer Vision / AI Engineer**

An automated system for detecting and extracting objects from engineering drawings, with OCR capabilities for text extraction from Note and Table regions while preserving table structure.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Setup & Installation](#setup--installation)
- [Training](#training)
- [Inference](#inference)
- [Web Demo](#web-demo)
- [Results & Experiments](#results--experiments)
- [Approach & Methodology](#approach--methodology)
- [Future Improvements](#future-improvements)

## 🎯 Overview

The system detects 3 types of objects in engineering drawings:

| Class | Description | Color |
|-------|-------------|-------|
| **PartDrawing** | Technical drawing regions | 🟢 Green |
| **Note** | Text annotations & notes | 🟠 Orange |
| **Table** | Data tables with structure | 🔵 Blue |

### Key Features
- **Object Detection**: Faster R-CNN with ResNet-50 FPN v2 backbone (COCO pre-trained)
- **OCR**: PaddleOCR for text recognition (Note regions)
- **Table Structure Preservation**: PPStructure for structure-aware table OCR
- **Web Demo**: Interactive Gradio interface with upload, visualization, JSON, and OCR panels
- **JSON Output**: Structured output with bounding boxes, confidence scores, and OCR content

## 🏛️ Architecture

```
Input Image → Object Detection (Faster R-CNN) → Crop Detected Regions
                                                        ↓
                                               PartDrawing → Save crop
                                               Note → PaddleOCR → Text
                                               Table → PPStructure → Structured Table
                                                        ↓
                                               JSON Output + Visualization
```

### Why Faster R-CNN (Not YOLO)?
- ✅ **License**: torchvision is BSD-3 licensed — commercially compatible
- ✅ **Accuracy**: Two-stage detector excels on medium/large objects (drawings, tables)
- ✅ **Pre-trained**: COCO weights provide strong transfer learning baseline
- ✅ **Cross-platform**: Works natively on Windows/Linux/Mac via PyTorch

### Why PaddleOCR?
- ✅ High accuracy on engineering text
- ✅ Built-in table structure recognition (PPStructure)
- ✅ Multi-language support
- ✅ Actively maintained, Apache 2.0 license

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- PyTorch 2.0+ with CUDA

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/cv-assessment.git
cd cv-assessment

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

The dataset is in COCO format from Roboflow:
```
datasets/
└── BOM-Folder- BOM-Dataset.coco/
    └── train/
        ├── _annotations.coco.json   # COCO annotations
        ├── 1_png.rf.xxx.png          # Training images
        └── ...
```

- 58 images with 412 annotations
- 3 classes: PartDrawing, Note, Table

## 🏋️ Training

### Quick Start

```bash
python src/detection/train.py
```

Training uses config from `config/train_config.yaml`. Key hyperparameters:

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet-50 FPN v2 |
| Pre-trained | COCO |
| Learning Rate | 0.005 |
| Batch Size | 2 |
| Epochs | 50 |
| LR Scheduler | StepLR (step=15, gamma=0.1) |
| Optimizer | SGD (momentum=0.9, weight_decay=5e-4) |
| Data Augmentation | HorizontalFlip, RandomBrightnessContrast, GaussianBlur |

### Training Output

Models are saved to `models/`:
- `best_model.pth` — Best validation loss checkpoint
- `final_model.pth` — Final epoch checkpoint
- `training_log.json` — Training metrics per epoch

## 🔍 Inference

### Single Image

```bash
python src/detection/inference.py \
    --model models/best_model.pth \
    --input path/to/image.jpg \
    --output outputs/
```

### Directory Processing

```bash
python src/pipeline/pipeline.py \
    --model models/best_model.pth \
    --input path/to/images/ \
    --output outputs/
```

### Output Structure

```
outputs/
├── crops/
│   ├── PartDrawing/   # Cropped part drawing regions
│   ├── Note/          # Cropped note regions
│   └── Table/         # Cropped table regions
├── visualizations/    # Images with bounding boxes
└── json/              # JSON results per image
```

### JSON Output Format

```json
{
  "image": "drawing_001.jpg",
  "objects": [
    {
      "id": 1,
      "class": "Table",
      "confidence": 0.97,
      "bbox": { "x1": 120, "y1": 340, "x2": 680, "y2": 520 },
      "ocr_content": {
        "type": "table",
        "rows": [["Header1", "Header2"], ["Data1", "Data2"]],
        "html": "<table>...</table>",
        "raw_text": "Header1 | Header2\nData1 | Data2"
      }
    },
    {
      "id": 2,
      "class": "Note",
      "confidence": 0.93,
      "bbox": { "x1": 50, "y1": 600, "x2": 400, "y2": 700 },
      "ocr_content": {
        "type": "text",
        "text": "All dimensions in mm unless otherwise specified."
      }
    }
  ]
}
```

## 🌐 Web Demo

### Local
```bash
python src/web/app.py
```
Opens at `http://localhost:7860`

### Deploy to HuggingFace Spaces
```bash
# See deployment instructions in the Spaces documentation
```

### Features
1. **Upload Zone**: Drag-and-drop or click to upload engineering drawings
2. **Confidence Slider**: Adjust detection confidence threshold
3. **Visualization**: Bounding boxes with class-specific colors
4. **JSON Panel**: Complete detection results in JSON format
5. **OCR Panel**: Extracted text from Note and Table regions

## 📊 Results & Experiments

### Detection Performance

| Metric | Value |
|--------|-------|
| mAP@0.5 | TBD (after training) |
| mAP@0.5:0.95 | TBD |

### Training Approaches Tried

1. **Faster R-CNN ResNet-50 FPN v2** (selected)
   - Best balance of accuracy and speed
   - Strong transfer learning from COCO

2. **Data Augmentation**
   - Horizontal flip (p=0.3)
   - Random brightness/contrast
   - Gaussian blur

### OCR Performance

- Note OCR: PaddleOCR with angle classification
- Table OCR: PPStructure with fallback to coordinate-based cell grouping

## 🧠 Approach & Methodology

### Detection Strategy
1. **Transfer Learning**: Start from COCO pre-trained Faster R-CNN
2. **Fine-tuning**: Train all layers on engineering drawing dataset
3. **Post-processing**: Class-specific NMS with tuned thresholds

### OCR Strategy
1. **Note regions**: Direct OCR with PaddleOCR (angle-aware)
2. **Table regions**: Two-pass approach:
   - Primary: PPStructure for structure-aware recognition
   - Fallback: Y-coordinate clustering for row grouping

### Key Design Decisions
- **No YOLO**: Per requirements, using commercially-licensed alternatives
- **torchvision Faster R-CNN**: Native PyTorch, cross-platform, well-tested
- **PaddleOCR**: Best accuracy-speed tradeoff for engineering text
- **Gradio**: Rapid prototyping, easy deployment to HuggingFace Spaces

## 🚀 Future Improvements

1. **Ensemble Detection**: Combine Faster R-CNN + RT-DETR for better accuracy
2. **Test-Time Augmentation**: Multi-scale inference for challenging images
3. **Confidence Calibration**: Temperature scaling for better score reliability
4. **Advanced Table OCR**: Fine-tune table detection model on engineering tables
5. **More Data Augmentation**: Mosaic, MixUp, CutMix for small dataset
6. **Instance Segmentation**: Mask R-CNN for pixel-level boundaries

## 📄 License

This project is created for assessment purposes. Detection model uses BSD-3 licensed torchvision.

## 👤 Author

Candidate — Computer Vision / AI Engineer Assessment @ Sotatek
