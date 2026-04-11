"""
OCR Module for Note Regions in Engineering Drawings.
Uses PaddleOCR for text recognition with post-processing.
"""
import os
import numpy as np
from PIL import Image


def init_ocr_engine(lang="vi", use_gpu=True):
    """Initialize PaddleOCR engine."""
    from paddleocr import PaddleOCR
    # Use English and Vietnamese if mixed, typically passing 'vi' handles both well in PaddleOCR
    reader = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)
    return reader


def ocr_note(ocr_engine, image_input):
    """
    Perform OCR on a Note region.

    Args:
        ocr_engine: Initialized PaddleOCR Reader.
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

    # PaddleOCR returns list of [[bbox, (text, confidence)], ...]
    results = ocr_engine.ocr(image, cls=True)

    if not results or not results[0]: # Sometimes returns [None] or empty
        return {"type": "text", "text": "", "lines": []}

    lines = []
    # results is a list of results for each image. We only passed one image.
    for res in results[0]:
        if res is None:
            continue
        bbox, (text, confidence) = res
        
        # bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
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
    # Give some tolerance to y to allow reading left-to-right on roughly same lines
    # PaddleOCR usually gives them in a reasonable order but we can sort just in case.
    lines.sort(key=lambda l: (l["bbox"][1] // 10, l["bbox"][0]))

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
