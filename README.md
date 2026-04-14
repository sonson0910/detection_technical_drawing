---
python_version: "3.10"
title: Engineering Drawing Detection
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: "5.15.0"
app_file: app.py
pinned: false
---

# Engineering Drawing Object Detection and OCR System

**Technical Assessment — Sotatek Computer Vision / AI Engineer**

This project provides an automated pipeline for detecting and extracting objects from engineering drawings. It also includes OCR capabilities for text extraction from Note and Table regions, with a structured layout preservation mechanism for tables.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Setup and Installation](#setup-and-installation)
- [Training](#training)
- [Inference](#inference)
- [Web Demo](#web-demo)
- [Deploy and Submission Assets](#deploy-and-submission-assets)
- [Results and Experiments](#results-and-experiments)
- [Approach and Methodology](#approach-and-methodology)
- [Future Improvements](#future-improvements)

## Overview

The system is designed to detect three classes of objects in engineering drawings:

| Class | Description | Display Color |
|-------|-------------|---------------|
| **PartDrawing** | Technical drawing regions | Green |
| **Note** | Text annotations and notes | Orange |
| **Table** | Data tables | Blue |

### Features
- **Object Detection**: Implemented using torchvision's Faster R-CNN with a ResNet-50 FPN v2 backbone.
- **OCR**: Integrated PaddleOCR for text recognition in Note regions.
- **Table Structure Preservation**: Utilized PPStructure to extract and maintain table formats.
- **Web Demo**: A Gradio interface for uploading drawings, visualizing bounding boxes, and displaying OCR and JSON outputs.
- **JSON Output**: Standardized output containing bounding boxes, confidence scores, and extracted text.

## Architecture

```text
Input Image 
  -> Faster R-CNN Detection 
  -> Crop Detected Regions
      |-> PartDrawing: Save as image
      |-> Note: Process via PaddleOCR -> Extract Text
      |-> Table: Process via PPStructure -> Extract Structured Table HTML
  -> Aggregate JSON Output + Bounding Box Visualization
```

### Rationale for Faster R-CNN
- **Licensing**: torchvision is BSD-3 licensed, highly compatible with commercial environments, unlike AGPL-restricted YOLO models.
- **Accuracy**: Two-stage detectors generally excel at finding medium to large objects with complex aspect ratios like engineering tables and wide note blocks.
- **Pre-trained Weights**: Starting from COCO weights provides a solid transfer learning baseline.
- **Cross-platform**: It runs consistently across environments utilizing PyTorch.

### Rationale for PaddleOCR
- Achieves high accuracy for dense technical text.
- Provides built-in table structure recognition through PPStructure.
- Supported actively and released under Apache 2.0 license.

## Setup and Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- PyTorch 2.0+ with CUDA support

### Installation

```bash
git clone https://github.com/<your-username>/cv-assessment.git
cd cv-assessment

python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### Dataset Setup

The dataset follows the standard COCO format:
```text
datasets/
└── BOM-Folder- BOM-Dataset.coco/
    └── train/
        ├── _annotations.coco.json
        ├── 1_png.rf.xxx.png
        └── ...
```

- 58 images with 412 annotations across 3 classes.

## Training

### Quick Start

```bash
python src/detection/train.py
```

The training process uses configurations defined in `config/train_config.yaml`. The key parameters include:

- **Backbone**: ResNet-50 FPN v2
- **Pre-trained Weights**: COCO
- **Learning Rate**: 0.005
- **Batch Size**: 2
- **Epochs**: 50
- **LR Scheduler**: StepLR (step=15, gamma=0.1)
- **Optimizer**: SGD (momentum=0.9, weight_decay=5e-4)

### Training Artifacts

Trained models and logs are saved to the `models/` directory:
- `best_model.pth` — Checkpoint with the best validation loss
- `final_model.pth` — Checkpoint from the last epoch
- `training_log.json` — Epoch-by-epoch loss metrics

## Inference

### Single Image

```bash
python src/detection/inference.py \
    --model models/best_map_model_backup.pth \
    --input path/to/image.jpg \
    --output outputs/
```

### Batch Directory Processing

```bash
python src/pipeline/pipeline.py \
    --model models/best_map_model_backup.pth \
    --input path/to/images/ \
    --output outputs/
```

### Output Directory Structure

```text
outputs/
├── crops/
│   ├── PartDrawing/
│   ├── Note/
│   └── Table/
├── visualizations/
└── json/
```

### JSON Definition

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
    }
  ]
}
```

## Web Demo

To start the Gradio interface locally:
```bash
python src/web/app.py
```
Access the application at `http://localhost:7860`.

## Deploy and Submission Assets

- **Web Demo URL:** [https://huggingface.co/spaces/sonson0910/engineering-drawing-detection](https://huggingface.co/spaces/sonson0910/engineering-drawing-detection)
- **Model Weights Link:** [Download best_model.pth (Google Drive)](https://drive.google.com/file/d/1JKR3tnbwY43VRUgC-1BNzXqY3BOtpD5y/view?usp=sharing)

## Technical Report

This section serves as the required short report detailing the methodology, experiments, achieved results, and proposed future improvements.

### 1. Approach and Methodology

**Detection Strategy:**
- Our primary directive was to adhere to the strict licensing constraints that explicitly prohibited YOLO models.
- We selected torchvision's Faster R-CNN with a ResNet-50 FPN v2 backbone. It provides an excellent balance between precision and commercial compliance (BSD-3 License). 
- The model was initialized with COCO pre-trained weights to leverage strong transfer learning, then fine-tuned on the provided engineering drawing dataset.
- Post-processing includes Non-Maximum Suppression (NMS) tailored with class-specific confidence thresholds to resolve overlapping bounding box predictions effectively.

**OCR and Table Structure Strategy:**
- **Note Extraction:** We integrated PaddleOCR to perform batch text extraction on cropped Note regions. PaddleOCR's angle classification ensures text is read correctly regardless of rotation.
- **Table Preservation:** To satisfy the strict requirement of preserving table structures (rows, columns, cell alignment), we utilized PPStructure. It processes the cropped table regions and outputs correctly aligned HTML table formats. A custom coordinate-clustering fallback was retained for edge cases.

### 2. Conducted Experiments

During development, several configurations were tested:
- **Baseline Models:** Evaluated baseline Faster R-CNN against SSD and RetinaNet. Faster R-CNN consistently yielded higher recall for small text blocks (Notes) and large tables.
- **Data Augmentation Strategies:** Implemented a pipeline using Horizontal Flip (p=0.3), Random Brightness/Contrast, and Gaussian Blur to simulate scan artifacts and noise. 
- **Anchor Box Tuning:** Briefly experimented with custom anchor box sizes tailored for extremely tall tables, but reverted to default FPN anchors as they generalized better across the validation set.

### 3. Achieved Results

- **Detection Pipeline:** Successfully deployed a robust detection model capable of classifying and cropping the required `PartDrawing`, `Note`, and `Table` regions.
- **OCR Pipeline:** Successfully integrated a text extraction pipeline that generates structured JSON configurations per image.
- **Table Formatting:** Met the table structure preservation criteria by exporting tabular data into structured Markdown/HTML layouts within the JSON output.
- **Web Functionality:** Fully deployed an interactive Gradio web application covering upload, detection, bounding box visualization, JSON payload viewing, and OCR preview.

### 4. Direction for Improvements

To further enhance the system in a production environment:
- **Model Ensembling:** Combine Faster R-CNN predictions with a transformer-based detector (like RT-DETR) to maximize bounding box precision on dense engineering schematics.
- **Test-Time Augmentation (TTA):** Implement multi-scale inference to better handle varying resolutions of scanned blueprints.
- **Specialized Table Finetuning:** Fine-tune PPStructure's table recognition on a dedicated dataset of engineering layouts to improve cell alignment accuracy on faded drawings.
- **Instance Segmentation:** Upgrade to Mask R-CNN to capture non-rectangular boundaries, delivering cleaner crops for OCR.

## License

This project was developed for technical assessment purposes. The detection components rely on BSD-3 licensed torchvision models, fully complying with standard commercial requirements.
