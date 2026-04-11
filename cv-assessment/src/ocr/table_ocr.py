"""
Table OCR Module for Engineering Drawings.
Preserves table structure (rows, columns, cell alignment).
Uses PaddleOCR PPStructure for actual table reconstruction.
"""
import os
import numpy as np
from PIL import Image
import cv2


def init_table_engine(use_gpu=True):
    """Initialize PPStructure engine."""
    from paddleocr import PPStructure
    # Use English and Vietnamese if mixed
    table_engine = PPStructure(show_log=False, use_gpu=use_gpu, lang='en')
    return table_engine


def ocr_table_ppstructure(table_engine, image_input):
    """
    Perform table OCR using PPStructure with structure reconstruction.

    Args:
        table_engine: Initialized PPStructure engine.
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
            pass # Keep BGR for cv2 logic if needed, but PPStructure expects numpy array
    elif isinstance(image_input, Image.Image):
        # PIL to cv2 format directly
        image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    else:
        # Check if RGB, standard models expect BGR in cv2, but paddle handles BGR by default if passed as cv2 image
        if image_input.shape[-1] == 3:
            # Assume it was passed as RGB from inference.py
            image = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        else:
            image = image_input

    if image is None:
        return {"type": "table", "rows": [], "html": "", "raw_text": ""}

    # PPStructure returns list of dicts for each region, including 'type': 'table'
    results = table_engine(image)

    if not results:
        return {"type": "table", "rows": [], "html": "", "raw_text": ""}

    html = ""
    # Find the table result, if any. PPStructure might classify parts as text or figure.
    # If it classifies as table, we get res['res']['html'].
    for region in results:
        if region['type'] == 'table':
            res = region['res']
            html = res.get('html', '')
            break

    # If no table found by PPStructure, fallback to standard text grouping using PaddleOCR results
    if not html:
        text_items = []
        for region in results:
            if region['type'] == 'text':
                for box_info in region['res']:
                    text = box_info['text']
                    conf = box_info['confidence']
                    bbox = box_info['text_region'] 
                    x_coords = [pt[0] for pt in bbox]
                    y_coords = [pt[1] for pt in bbox]
                    y_center = np.mean(y_coords)
                    x_center = np.mean(x_coords)
                    text_items.append({"text": text.strip(), "y": y_center, "x": x_center, "conf": conf})
        
        if not text_items:
            return {"type": "table", "rows": [], "html": "", "raw_text": ""}

        # Cluster by y-coordinate to form rows
        text_items.sort(key=lambda t: t["y"])
        
        img_height = image.shape[0] if len(image.shape) >= 2 else 500
        y_threshold = max(15, img_height * 0.03)
        
        rows = []
        current_row = [text_items[0]]

        for item in text_items[1:]:
            if abs(item["y"] - current_row[-1]["y"]) < y_threshold:
                current_row.append(item)
            else:
                current_row.sort(key=lambda t: t["x"])
                rows.append([t["text"] for t in current_row])
                current_row = [item]

        if current_row:
            current_row.sort(key=lambda t: t["x"])
            rows.append([t["text"] for t in current_row])
            
        raw_text = "\n".join(" | ".join(row) for row in rows)

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

    # If HTML was found directly by Table structure:
    # We can try to parse rows out of HTML for standard representation or leave it. 
    # For now, just store raw text as the combined HTML without tags.
    import re
    # Remove tags to form raw text
    clean_text = re.sub('<[^<]+>', ' | ', html).replace('|  |', '|').strip(' |')
    
    return {
        "type": "table",
        "rows": [["(Structured from HTML)"]], 
        "html": html,
        "raw_text": clean_text,
    }


def ocr_table_from_crop(table_engine, crop_path):
    """Convenience function to OCR a cropped Table image file."""
    return ocr_table_ppstructure(table_engine, crop_path)
