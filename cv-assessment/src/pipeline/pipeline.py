"""
Full Pipeline: Detection → Crop → OCR → JSON Output.
Orchestrates the complete engineering drawing analysis workflow.
"""
import os
import sys
import json
import yaml
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.detection.inference import (
    load_model,
    preprocess_image,
    detect_objects,
    detect_note_regions,
    crop_objects,
    draw_detections,
    CLASS_NAMES,
)
from src.detection.postprocess import post_process_detections
from src.ocr.note_ocr import init_ocr_engine, ocr_note
from src.ocr.table_ocr import init_table_engine, ocr_table_ppstructure


class EngineeringDrawingPipeline:
    """Complete pipeline for engineering drawing analysis."""

    def __init__(self, model_path, device="cuda", conf_threshold=0.5,
                 ocr_lang="en", use_gpu=True):
        print("Loading detection model...")
        self.model = load_model(model_path, device=device)
        self.device = device
        self.conf_threshold = conf_threshold

        print("Loading OCR engines...")
        self.ocr_engine = init_ocr_engine(lang=ocr_lang, use_gpu=use_gpu)
        self.table_engine = init_table_engine(use_gpu=use_gpu)
        print("Pipeline ready!")

    def process_image(self, image_path, output_dir=None):
        """
        Process a single engineering drawing image.

        Args:
            image_path: Path to input image.
            output_dir: Optional output directory for saving results.

        Returns:
            Tuple of (json_result, visualization_image, crop_results)
        """
        image_name = os.path.basename(image_path)
        image_tensor, image_np = preprocess_image(image_path)

        # Step 1: Object Detection (PartDrawing + Table via model)
        boxes, labels, scores = detect_objects(
            self.model, image_tensor, self.device, self.conf_threshold
        )

        # Step 1b: Hybrid Note Detection (morphological analysis)
        # Only pass non-Note model detections as "covered" areas
        non_note_mask = labels != 2
        non_note_boxes = boxes[non_note_mask] if len(boxes) > 0 else np.zeros((0, 4))
        non_note_labels = labels[non_note_mask] if len(labels) > 0 else np.zeros((0,), dtype=int)
        note_boxes = detect_note_regions(image_np, non_note_boxes, non_note_labels)
        
        # Merge Note detections into results (avoid duplicating model-detected Notes)
        model_note_mask = labels == 2
        model_note_boxes = boxes[model_note_mask] if np.any(model_note_mask) else np.zeros((0, 4))
        
        all_boxes = list(boxes)
        all_labels = list(labels)
        all_scores = list(scores)
        for nb in note_boxes:
            # Check if hybrid Note overlaps with any model-detected Note
            is_duplicate = False
            for mnb in model_note_boxes:
                ix1 = max(nb[0], mnb[0])
                iy1 = max(nb[1], mnb[1])
                ix2 = min(nb[2], mnb[2])
                iy2 = min(nb[3], mnb[3])
                if ix1 < ix2 and iy1 < iy2:
                    inter = (ix2-ix1) * (iy2-iy1)
                    nb_area = (nb[2]-nb[0]) * (nb[3]-nb[1])
                    if inter / nb_area > 0.3:
                        is_duplicate = True
                        break
            if not is_duplicate:
                all_boxes.append(np.array(nb, dtype=float))
                all_labels.append(2)       # Note = class 2
                all_scores.append(0.85)    # Rule-based confidence
        
        all_boxes = np.array(all_boxes) if all_boxes else np.zeros((0, 4))
        all_labels = np.array(all_labels) if all_labels else np.zeros((0,), dtype=int)
        all_scores = np.array(all_scores) if all_scores else np.zeros((0,))

        if len(all_boxes) == 0:
            return (
                {"image": image_name, "objects": []},
                image_np,
                [],
            )

        # Step 1c: Post-processing — tighten boxes, resolve overlaps, exclude footer
        all_boxes, all_labels, all_scores = post_process_detections(
            image_np, all_boxes, all_labels, all_scores
        )

        if len(all_boxes) == 0:
            return (
                {"image": image_name, "objects": []},
                image_np,
                [],
            )

        # Step 2: Crop objects
        if output_dir:
            crops_dir = os.path.join(output_dir, "crops")
        else:
            crops_dir = os.path.join("outputs", "crops")

        detections = crop_objects(image_np, all_boxes, all_labels, all_scores, crops_dir, image_name)

        # Step 3: OCR on Note and Table regions
        ocr_results = {}
        for det in detections:
            if det["class"] == "Note":
                try:
                    crop = image_np[
                        det["bbox"]["y1"]:det["bbox"]["y2"],
                        det["bbox"]["x1"]:det["bbox"]["x2"],
                    ]
                    ocr_result = ocr_note(self.ocr_engine, crop)
                    ocr_results[det["id"]] = ocr_result
                except Exception as e:
                    print(f"  OCR error (Note): {e}")
                    ocr_results[det["id"]] = {"type": "text", "text": "", "lines": []}

            elif det["class"] == "Table":
                try:
                    crop = image_np[
                        det["bbox"]["y1"]:det["bbox"]["y2"],
                        det["bbox"]["x1"]:det["bbox"]["x2"],
                    ]
                    ocr_result = ocr_table_ppstructure(self.table_engine, crop)
                    ocr_results[det["id"]] = ocr_result
                except Exception as e:
                    print(f"  OCR error (Table): {e}")
                    ocr_results[det["id"]] = {"type": "table", "rows": [], "raw_text": ""}

        # Step 4: Generate JSON output
        objects = []
        for det in detections:
            obj = {
                "id": det["id"],
                "class": det["class"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
            }
            if det["id"] in ocr_results:
                obj["ocr_content"] = ocr_results[det["id"]]
            objects.append(obj)

        json_result = {"image": image_name, "objects": objects}

        # Step 5: Draw visualization (includes hybrid Note detections)
        vis_image = draw_detections(image_np, all_boxes, all_labels, all_scores)

        # Step 6: Save outputs
        if output_dir:
            # Save visualization
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, f"vis_{image_name}")
            Image.fromarray(vis_image).save(vis_path)

            # Save JSON
            json_dir = os.path.join(output_dir, "json")
            os.makedirs(json_dir, exist_ok=True)
            json_path = os.path.join(json_dir, f"{os.path.splitext(image_name)[0]}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_result, f, indent=2, ensure_ascii=False)

        return json_result, vis_image, detections

    def process_directory(self, input_dir, output_dir):
        """Process all images in a directory."""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        image_files = [
            f for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]

        print(f"Processing {len(image_files)} images...")
        all_results = []

        for img_file in sorted(image_files):
            print(f"  Processing {img_file}...")
            img_path = os.path.join(input_dir, img_file)
            result, _, _ = self.process_image(img_path, output_dir)
            all_results.append(result)

        # Save combined results
        combined_path = os.path.join(output_dir, "all_results.json")
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\nDone! Processed {len(all_results)} images.")
        print(f"Results saved to: {output_dir}")
        return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Engineering Drawing Analysis Pipeline")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--input", required=True, help="Input image or directory")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    pipeline = EngineeringDrawingPipeline(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.confidence,
    )

    if os.path.isdir(args.input):
        pipeline.process_directory(args.input, args.output)
    else:
        result, vis, _ = pipeline.process_image(args.input, args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
