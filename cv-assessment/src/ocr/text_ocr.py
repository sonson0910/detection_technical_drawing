"""
OCR Module for Note Regions in Engineering Drawings.
Uses EasyOCR for text recognition with post-processing.
"""
import os
import numpy as np
from PIL import Image


def init_ocr_engine(lang="en", use_gpu=True):
    """Initialize EasyOCR engine."""
    import easyocr
    reader = easyocr.Reader(
        [lang],
        gpu=use_gpu,
        verbose=False,
    )
    return reader


def ocr_note(ocr_engine, image_input):
    """
    Perform OCR on a Note region.

    Args:
        ocr_engine: Initialized EasyOCR Reader.
        image_input: PIL Image, numpy array, or path to image file.

    Returns:
        Dict with OCR results:
        {
            "type": "text",
            "text": "extracted text content",
            "lines": [{"text": "line text", "confidence": 0.95, "bbox": [x1,y1,x2,y2]}, ...]
        }
    """
    if isinstance(image_input, str):
        image = np.array(Image.open(image_input).convert("RGB"))
    elif isinstance(image_input, Image.Image):
        image = np.array(image_input.convert("RGB"))
    else:
        image = image_input

    # EasyOCR returns list of (bbox, text, confidence)
    results = ocr_engine.readtext(image)

    if not results:
        return {"type": "text", "text": "", "lines": []}

    lines = []
    for (bbox, text, confidence) in results:
        # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)

        lines.append({
            "text": text.strip(),
            "confidence": round(float(confidence), 4),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })

    # Sort lines by y-coordinate (top to bottom), then x (left to right)
    lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))

    # Combine into full text
    full_text = "\n".join(line["text"] for line in lines)

    return {
        "type": "text",
        "text": full_text,
        "lines": lines,
    }


def ocr_note_from_crop(ocr_engine, crop_path):
    """Convenience function to OCR a cropped Note image file."""
    return ocr_note(ocr_engine, crop_path)
