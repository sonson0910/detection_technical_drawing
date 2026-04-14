"""
Gradio Web Demo for Engineering Drawing Detection & OCR.
Provides upload, visualization, JSON panel, and OCR panel.
"""
import os
import sys
import json
import tempfile
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import gradio as gr

# Lazy-load pipeline to avoid import errors when modules aren't installed yet
_pipeline = None


def get_pipeline():
    """Lazy-initialize the pipeline."""
    global _pipeline
    if _pipeline is None:
        from src.pipeline.pipeline import EngineeringDrawingPipeline

        model_path = os.environ.get(
            "MODEL_PATH",
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "best_map_model_backup.pth"),
        )
        import torch
        device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Set MODEL_PATH environment variable or train the model first."
            )
        _pipeline = EngineeringDrawingPipeline(
            model_path=model_path,
            device=device,
            conf_threshold=0.5,
        )
    return _pipeline


def process_drawing(input_image, confidence_threshold=0.5):
    """
    Process an uploaded engineering drawing.

    Returns:
        visualization_image: Image with bounding boxes
        json_output: JSON string with detection results
        ocr_output: Formatted OCR results
    """
    if input_image is None:
        return None, "No image uploaded", "No image uploaded"

    try:
        pipeline = get_pipeline()
        pipeline.conf_threshold = confidence_threshold

        # Save input image temporarily
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            if isinstance(input_image, np.ndarray):
                Image.fromarray(input_image).save(tmp.name)
            else:
                input_image.save(tmp.name)
            tmp_path = tmp.name

        # Process
        with tempfile.TemporaryDirectory() as tmp_dir:
            result, vis_image, detections = pipeline.process_image(tmp_path, tmp_dir)

        os.unlink(tmp_path)

        # Format JSON output
        json_str = json.dumps(result, indent=2, ensure_ascii=False)

        # Format OCR output
        ocr_lines = []
        for obj in result.get("objects", []):
            if obj["class"] in ["Note", "Table"] and "ocr_content" in obj and obj["ocr_content"]:
                ocr_content = obj["ocr_content"]
                header = f"{'='*50}\n📌 {obj['class']} (ID: {obj['id']}, Conf: {obj['confidence']:.2f})\n{'='*50}"
                ocr_lines.append(header)

                if ocr_content["type"] == "text":
                    ocr_lines.append(ocr_content.get("text", ""))
                elif ocr_content["type"] == "table":
                    rows = ocr_content.get("rows", [])
                    if rows:
                        # Format as markdown table
                        max_cols = max(len(row) for row in rows)
                        for i, row in enumerate(rows):
                            # Pad shorter rows
                            while len(row) < max_cols:
                                row.append("")
                            ocr_lines.append("| " + " | ".join(row) + " |")
                            if i == 0:
                                ocr_lines.append("|" + "|".join(["---"] * max_cols) + "|")
                    else:
                        ocr_lines.append(ocr_content.get("raw_text", "No content detected"))

                ocr_lines.append("")

        ocr_text = "\n".join(ocr_lines) if ocr_lines else "No Note or Table regions detected."

        return vis_image, json_str, ocr_text

    except FileNotFoundError as e:
        error_msg = str(e)
        return None, error_msg, error_msg
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return None, error_msg, error_msg


# Build Gradio Interface
def create_demo():
    """Create the Gradio demo interface."""
    with gr.Blocks(
        title="Engineering Drawing Detection & OCR",
    ) as demo:
        gr.Markdown(
            """
            # 🏗️ Engineering Drawing Detection & OCR System
            
            Upload an engineering drawing image to automatically detect and extract:
            - **PartDrawing** — Technical drawing regions
            - **Note** — Annotations and text notes  
            - **Table** — Data tables with structure preservation
            
            The system uses **Faster R-CNN** for object detection and **PaddleOCR** for text recognition.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="📤 Upload Engineering Drawing",
                    type="numpy",
                    height=400,
                )
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.99,
                    value=0.5,
                    step=0.05,
                    label="Confidence Threshold",
                )
                detect_btn = gr.Button("🔍 Detect & Analyze", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="📊 Detection Results (Bounding Boxes)",
                    type="numpy",
                    height=400,
                )

        with gr.Row():
            with gr.Column(scale=1):
                json_output = gr.Code(
                    label="📋 JSON Output",
                    language="json",
                    lines=20,
                )
            with gr.Column(scale=1):
                ocr_output = gr.Textbox(
                    label="📝 OCR Results",
                    lines=20,
                    max_lines=30,
                )

        # Color legend
        gr.Markdown(
            """
            ### 🎨 Color Legend
            | Class | Color | Description |
            |-------|-------|-------------|
            | PartDrawing | 🟢 Green | Technical drawing regions |
            | Note | 🟠 Orange | Text annotations |
            | Table | 🔵 Blue | Data tables |
            """
        )

        detect_btn.click(
            fn=process_drawing,
            inputs=[input_image, confidence_slider],
            outputs=[output_image, json_output, ocr_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        share=False,
    )
