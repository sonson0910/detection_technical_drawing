"""
Table OCR Module for Engineering Drawings.
Preserves table structure (rows, columns, cell alignment).
Uses EasyOCR with y-coordinate clustering for row detection.
"""
import os
import numpy as np
from PIL import Image
import cv2


def init_table_engine(use_gpu=True):
    """Initialize EasyOCR-based table engine."""
    import easyocr
    reader = easyocr.Reader(
        ["en"],
        gpu=use_gpu,
        verbose=False,
    )
    return reader


def ocr_table_ppstructure(table_engine, image_input):
    """
    Perform table OCR using EasyOCR with structure reconstruction.

    Args:
        table_engine: Initialized EasyOCR Reader.
        image_input: PIL Image, numpy array, or path to image.

    Returns:
        Dict with structured table data:
        {
            "type": "table",
            "rows": [["cell1", "cell2", ...], ...],
            "html": "<table>...</table>",
            "raw_text": "all text concatenated"
        }
    """
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        image = np.array(image_input.convert("RGB"))
    else:
        image = image_input

    if image is None:
        return {"type": "table", "rows": [], "html": "", "raw_text": ""}

    # EasyOCR returns list of (bbox, text, confidence)
    results = table_engine.readtext(image)

    if not results:
        return {"type": "table", "rows": [], "html": "", "raw_text": ""}

    # Group text by rows using y-coordinate clustering
    text_items = []
    for (bbox, text, confidence) in results:
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        y_center = np.mean(y_coords)
        x_center = np.mean(x_coords)
        text_items.append({"text": text.strip(), "y": y_center, "x": x_center, "conf": confidence})

    if not text_items:
        return {"type": "table", "rows": [], "html": "", "raw_text": ""}

    # Cluster by y-coordinate to form rows
    text_items.sort(key=lambda t: t["y"])
    
    # Dynamic threshold based on image height
    img_height = image.shape[0] if len(image.shape) >= 2 else 500
    y_threshold = max(15, img_height * 0.03)
    
    rows = []
    current_row = [text_items[0]]

    for item in text_items[1:]:
        if abs(item["y"] - current_row[-1]["y"]) < y_threshold:
            current_row.append(item)
        else:
            # Sort current row by x-coordinate
            current_row.sort(key=lambda t: t["x"])
            rows.append([t["text"] for t in current_row])
            current_row = [item]

    if current_row:
        current_row.sort(key=lambda t: t["x"])
        rows.append([t["text"] for t in current_row])

    # Build raw text
    raw_text = "\n".join(" | ".join(row) for row in rows)

    # Build simple HTML table
    html = "<table>\n"
    for row in rows:
        html += "  <tr>\n"
        for cell in row:
            html += f"    <td>{cell}</td>\n"
        html += "  </tr>\n"
    html += "</table>"

    return {
        "type": "table",
        "rows": rows,
        "html": html,
        "raw_text": raw_text,
    }


def ocr_table_from_crop(table_engine, crop_path):
    """Convenience function to OCR a cropped Table image file."""
    return ocr_table_ppstructure(table_engine, crop_path)
