"""
Post-Processing Module for Engineering Drawing Detection.
Refines bounding boxes to eliminate "lẹm" (overlap/imprecision) issues.

Techniques:
1. Content-aware box tightening - shrink boxes to actual ink/text boundaries
2. Cross-class overlap resolution - prioritize Table > Note > PartDrawing
3. Footer strip exclusion - remove footer text from PartDrawing boxes
4. Structural line-based boundary refinement - use drawing border lines
"""
import cv2
import numpy as np


# Class priority for overlap resolution (higher = keep, lower = trim)
CLASS_PRIORITY = {
    3: 3,   # Table - highest priority (most structurally defined)
    2: 2,   # Note - medium priority
    1: 1,   # PartDrawing - lowest priority (fills remaining space)
}


def post_process_detections(image_np, boxes, labels, scores,
                             footer_ratio=0.06, overlap_iou_thresh=0.15):
    """
    Full post-processing pipeline for detected bounding boxes.

    Args:
        image_np: Original image as numpy array (H, W, 3)
        boxes: Detected boxes as numpy array (N, 4) [x1, y1, x2, y2]
        labels: Class labels (N,)
        scores: Confidence scores (N,)
        footer_ratio: Bottom fraction of image considered as footer
        overlap_iou_thresh: Minimum overlap ratio to trigger resolution

    Returns:
        Refined (boxes, labels, scores) after post-processing
    """
    if len(boxes) == 0:
        return boxes, labels, scores

    h, w = image_np.shape[:2]
    boxes = boxes.astype(float).copy()
    labels = labels.copy()
    scores = scores.copy()

    # Step -1: Filter fake tables (tables with no vertical grid lines)
    boxes, labels, scores = _filter_fake_tables(image_np, boxes, labels, scores)

    if len(boxes) == 0:
        return boxes, labels, scores

    # Step 0: Suppress duplicate regions (same area, different class labels)
    boxes, labels, scores = _suppress_duplicate_regions(boxes, labels, scores)

    if len(boxes) == 0:
        return boxes, labels, scores

    # Step 1: Footer strip exclusion for PartDrawing
    boxes, labels, scores = _exclude_footer_strip(
        boxes, labels, scores, h, w, footer_ratio
    )

    # Step 2: Content-aware box tightening (gentle)
    boxes = _tighten_boxes(image_np, boxes, labels)

    # Step 2.3: Filter low-confidence Tables outside footer zone
    boxes, labels, scores = _filter_low_conf_tables_outside_footer(
        boxes, labels, scores, h, min_table_conf=0.15, footer_pct=0.25
    )

    # Step 2.5: Same-class overlap resolution (e.g., adjacent PartDrawings)
    # Disabled: overlaps are natural in bounding boxes and trimming amputates drawing details (e.g., numbers/lines)
    # boxes = _resolve_same_class_overlap(image_np, boxes, labels)

    # Step 3: Cross-class overlap resolution (only partial overlaps)
    # Re-enabled: Note should be trimmed when overlapping Table (PartDrawing is inherently protected)
    boxes = _resolve_cross_class_overlap(boxes, labels, scores, overlap_iou_thresh)

    # Step 4: Remove invalid boxes (zero area, out of bounds)
    boxes, labels, scores = _filter_invalid_boxes(boxes, labels, scores, h, w)

    return boxes, labels, scores


def _filter_fake_tables(image_np, boxes, labels, scores):
    """Remove tables that don't have grid lines.

    True Tables in engineering drawings typically have vertical and horizontal grid lines.
    Text blocks (Notes) might erroneously be predicted as Tables, but they uniformly
    lack strictly drawn vertical grid lines.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    keep = np.ones(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        if labels[i] == 3:  # Table
            x1, y1, x2, y2 = map(int, boxes[i])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)

            bw = x2 - x1
            bh = y2 - y1
            if bw < 40 or bh < 20:
                continue

            roi = gray[y1:y2, x1:x2]
            binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 15, 8)

            # Detect vertical lines specifically (which paragraph text lacks)
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, bh // 3)))
            v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
            v_proj = np.sum(v_lines > 0, axis=0)
            v_line_count = len(np.where(v_proj > bh * 0.25)[0])

            # Detect horizontal lines too (BOM tables might have horizontal separators but no vertical lines)
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, bw // 3), 1))
            h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
            h_proj = np.sum(h_lines > 0, axis=1)
            h_line_count = len(np.where(h_proj > bw * 0.25)[0])
            
            # If a large region is classified as Table but lacks BOTH vertical/horizontal dividers, it's just Note
            if v_line_count < 2 and h_line_count < 2:
                keep[i] = False

    return boxes[keep], labels[keep], scores[keep]


def _filter_low_conf_tables_outside_footer(boxes, labels, scores, img_h,
                                            min_table_conf=0.15, footer_pct=0.25):
    """Filter low-confidence Table detections that are NOT in the footer zone.

    Engineering drawings typically have small title-block/BOM tables at the
    bottom of the page. The model may detect these at low confidence.
    We keep low-conf Tables if their bottom edge is in the footer zone,
    but remove them if they're in the main drawing area (likely false positives).
    """
    footer_y = img_h * (1 - footer_pct)  # Top of footer zone

    keep = np.ones(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        if labels[i] == 3 and scores[i] < min_table_conf:
            # Low-confidence Table — only keep if its bottom edge is in footer
            if boxes[i][3] < footer_y:
                keep[i] = False  # Table NOT in footer zone, remove it

    return boxes[keep], labels[keep], scores[keep]


def _exclude_footer_strip(boxes, labels, scores, img_h, img_w, footer_ratio):
    """Remove footer region from PartDrawing detections.

    Engineering drawings often have a footer strip at the bottom with
    metadata text (e.g., "TÔP BẢN VẼ LẮP - ĐHBK HÀ NỘI") that should
    not be part of any PartDrawing.
    """
    footer_y = img_h * (1 - footer_ratio)

    new_boxes = []
    new_labels = []
    new_scores = []

    for box, lbl, scr in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box

        if lbl == 1:  # PartDrawing only
            # If box extends into footer, trim it
            if y2 > footer_y and y1 < footer_y:
                y2 = footer_y
            elif y1 >= footer_y:
                # Entire box is in footer - skip it
                continue

        new_boxes.append([x1, y1, x2, y2])
        new_labels.append(lbl)
        new_scores.append(scr)

    return (
        np.array(new_boxes) if new_boxes else np.zeros((0, 4)),
        np.array(new_labels) if new_labels else np.zeros((0,), dtype=int),
        np.array(new_scores) if new_scores else np.zeros((0,)),
    )


def _tighten_boxes(image_np, boxes, labels):
    """Content-aware box tightening using contour analysis.

    For each box, analyze the content within to find actual ink/drawing
    boundaries and shrink the box to fit. Uses different strategies per class:
    - PartDrawing: Find structural border lines (thick lines marking boundaries)
    - Note: Tighten to text content boundaries
    - Table: Tighten to table border lines
    """
    h, w = image_np.shape[:2]
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    refined_boxes = []

    for box, lbl in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 10 or bh <= 10:
            refined_boxes.append([x1, y1, x2, y2])
            continue

        roi = gray[y1:y2, x1:x2]

        if lbl == 1:  # PartDrawing
            # Disabled: tightening PartDrawings blindly trims out sparse details like numbers and leader lines
            refined = [x1, y1, x2, y2]
        elif lbl == 2:  # Note
            refined = _tighten_note(roi, x1, y1, x2, y2)
        elif lbl == 3:  # Table
            refined = _tighten_table(roi, x1, y1, x2, y2)
        else:
            refined = [x1, y1, x2, y2]

        refined_boxes.append(refined)

    return np.array(refined_boxes) if refined_boxes else np.zeros((0, 4))


def _tighten_partdrawing(roi, x1, y1, x2, y2, full_gray, img_h, img_w):
    """Tighten PartDrawing box using structural line detection.

    Engineering drawings typically have clear border lines separating sections.
    We detect these strong horizontal/vertical lines to find precise boundaries.
    """
    bw = x2 - x1
    bh = y2 - y1

    # Use edge detection to find strong lines in the ROI
    # Look for content boundaries by analyzing row/column projections
    binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 8)

    # Horizontal projection (sum along rows)
    h_proj = np.sum(binary > 0, axis=1)
    # Vertical projection (sum along columns)
    v_proj = np.sum(binary > 0, axis=0)

    # Find content boundaries (where ink density drops to near zero)
    ink_threshold = bw * 0.02  # 2% of width = "has content"
    ink_threshold_v = bh * 0.02  # 2% of height

    # Find top boundary (first row with significant content)
    top_offset = 0
    for i in range(min(bh // 4, len(h_proj))):
        if h_proj[i] > ink_threshold:
            top_offset = max(0, i - 2)
            break

    # Find bottom boundary (last row with significant content)
    bottom_offset = bh
    for i in range(bh - 1, max(bh * 3 // 4, 0), -1):
        if i < len(h_proj) and h_proj[i] > ink_threshold:
            bottom_offset = min(bh, i + 3)
            break

    # Find left boundary
    left_offset = 0
    for i in range(min(bw // 4, len(v_proj))):
        if v_proj[i] > ink_threshold_v:
            left_offset = max(0, i - 2)
            break

    # Find right boundary
    right_offset = bw
    for i in range(bw - 1, max(bw * 3 // 4, 0), -1):
        if i < len(v_proj) and v_proj[i] > ink_threshold_v:
            right_offset = min(bw, i + 3)
            break

    # Don't over-tighten (max 10% reduction per side)
    max_trim = 0.10
    top_offset = min(top_offset, int(bh * max_trim))
    left_offset = min(left_offset, int(bw * max_trim))
    bottom_offset = max(bottom_offset, int(bh * (1 - max_trim)))
    right_offset = max(right_offset, int(bw * (1 - max_trim)))

    return [
        x1 + left_offset,
        y1 + top_offset,
        x1 + right_offset,
        y1 + bottom_offset,
    ]


def _tighten_note(roi, x1, y1, x2, y2):
    """Tighten Note box to text content boundaries.

    Notes are text-heavy regions. We find the actual text extent
    and add minimal padding.
    """
    bw = x2 - x1
    bh = y2 - y1

    binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 10)

    # Find content extent
    h_proj = np.sum(binary > 0, axis=1)
    v_proj = np.sum(binary > 0, axis=0)

    content_rows = np.where(h_proj > bw * 0.01)[0]
    content_cols = np.where(v_proj > bh * 0.01)[0]

    if len(content_rows) == 0 or len(content_cols) == 0:
        return [x1, y1, x2, y2]

    # Add generous padding to avoid clipping content (8% of dimensions)
    pad_x = max(10, int(bw * 0.08))
    pad_y = max(8, int(bh * 0.08))

    new_x1 = x1 + max(0, content_cols[0] - pad_x)
    new_y1 = y1 + max(0, content_rows[0] - pad_y)
    new_x2 = x1 + min(bw, content_cols[-1] + pad_x)
    new_y2 = y1 + min(bh, content_rows[-1] + pad_y)

    # Don't over-tighten for notes (max 8% trim per side)
    max_trim = 0.08
    new_x1 = max(new_x1, x1 + int(bw * 0))
    new_y1 = max(new_y1, y1 + int(bh * 0))
    new_x2 = min(new_x2, x2)
    new_y2 = min(new_y2, y2)

    return [new_x1, new_y1, new_x2, new_y2]


def _tighten_table(roi, x1, y1, x2, y2):
    """Tighten Table box to table border lines.

    Tables have clear structural borders (lines). We find the outermost
    horizontal and vertical lines to define precise boundaries.
    """
    bw = x2 - x1
    bh = y2 - y1

    # Skip tightening for small tables — morphological ops unreliable on tiny ROIs
    if bw < 100 or bh < 40:
        return [x1, y1, x2, y2]

    # Detect strong horizontal and vertical lines
    binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 8)

    # Detect horizontal lines (must be at least 1/2 of table width to be considered a grid line)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, bw // 2), 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Detect vertical lines (must be at least 1/2 of table height)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, bh // 2)))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Allow 2px tolerance for slightly slanted lines
    h_lines = cv2.dilate(h_lines, np.ones((3, 1), np.uint8))
    v_lines = cv2.dilate(v_lines, np.ones((1, 3), np.uint8))

    # Find line positions
    h_proj = np.sum(h_lines > 0, axis=1)
    v_proj = np.sum(v_lines > 0, axis=0)

    # Find first/last strong horizontal lines (table borders) requiring 60% span completeness
    h_line_rows = np.where(h_proj > bw * 0.6)[0]
    v_line_cols = np.where(v_proj > bh * 0.6)[0]

    if len(h_line_rows) >= 2:
        new_y1 = y1 + max(0, h_line_rows[0] - 1)
        new_y2 = y1 + min(bh, h_line_rows[-1] + 2)
    else:
        new_y1, new_y2 = y1, y2

    if len(v_line_cols) >= 2:
        new_x1 = x1 + max(0, v_line_cols[0] - 1)
        new_x2 = x1 + min(bw, v_line_cols[-1] + 2)
    else:
        new_x1, new_x2 = x1, x2

    return [new_x1, new_y1, new_x2, new_y2]



def _resolve_same_class_overlap(image_np, boxes, labels):
    """Resolve overlaps between boxes of the SAME class (e.g., adjacent PartDrawings).

    For overlapping same-class boxes, we find the structural dividing line
    in the overlap region and split the boundary there. If no clear line
    is found, we split at the midpoint of the overlap.
    """
    n = len(boxes)
    refined_boxes = boxes.copy()
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                continue  # Only handle same-class overlaps

            # Check overlap
            ix1 = max(refined_boxes[i][0], refined_boxes[j][0])
            iy1 = max(refined_boxes[i][1], refined_boxes[j][1])
            ix2 = min(refined_boxes[i][2], refined_boxes[j][2])
            iy2 = min(refined_boxes[i][3], refined_boxes[j][3])

            if ix1 >= ix2 or iy1 >= iy2:
                continue  # No overlap

            overlap_w = ix2 - ix1
            overlap_h = iy2 - iy1

            area_i = (refined_boxes[i][2] - refined_boxes[i][0]) * (refined_boxes[i][3] - refined_boxes[i][1])
            area_j = (refined_boxes[j][2] - refined_boxes[j][0]) * (refined_boxes[j][3] - refined_boxes[j][1])
            inter_area = overlap_w * overlap_h

            if inter_area / min(area_i, area_j) < 0.01:
                continue  # Negligible overlap

            # Determine overlap direction (horizontal or vertical adjacency)
            if overlap_w > overlap_h:
                # Horizontal overlap - boxes are arranged vertically
                # Split horizontally (adjust y boundaries)
                split_y = _find_structural_split(
                    gray, int(ix1), int(iy1), int(ix2), int(iy2), direction='horizontal'
                )

                # Determine which box is on top
                if refined_boxes[i][1] < refined_boxes[j][1]:
                    top_idx, bottom_idx = i, j
                else:
                    top_idx, bottom_idx = j, i

                # Top box: trim bottom to split line
                refined_boxes[top_idx][3] = min(refined_boxes[top_idx][3], split_y)
                # Bottom box: trim top to split line
                refined_boxes[bottom_idx][1] = max(refined_boxes[bottom_idx][1], split_y)

            else:
                # Vertical overlap - boxes are arranged horizontally
                # Split vertically (adjust x boundaries)
                split_x = _find_structural_split(
                    gray, int(ix1), int(iy1), int(ix2), int(iy2), direction='vertical'
                )

                # Determine which box is on the left
                if refined_boxes[i][0] < refined_boxes[j][0]:
                    left_idx, right_idx = i, j
                else:
                    left_idx, right_idx = j, i

                # Left box: trim right to split line
                refined_boxes[left_idx][2] = min(refined_boxes[left_idx][2], split_x)
                # Right box: trim left to split line
                refined_boxes[right_idx][0] = max(refined_boxes[right_idx][0], split_x)

    return refined_boxes


def _find_structural_split(gray, x1, y1, x2, y2, direction='horizontal'):
    """Find the structural dividing line in the overlap region.

    Looks for strong lines (drawing borders) in the overlap area.
    Falls back to geometric midpoint if no clear line is found.

    Args:
        gray: Grayscale image
        x1, y1, x2, y2: Overlap region coordinates
        direction: 'horizontal' for y-split, 'vertical' for x-split

    Returns:
        Split coordinate (y for horizontal, x for vertical)
    """
    h, w = gray.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        if direction == 'horizontal':
            return (y1 + y2) // 2
        else:
            return (x1 + x2) // 2

    # Detect edges
    binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 8)

    if direction == 'horizontal':
        # Look for strong horizontal lines
        rw = x2 - x1
        if rw < 5:
            return (y1 + y2) // 2
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, rw // 2), 1))
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        proj = np.sum(lines > 0, axis=1)

        # Find strongest horizontal line position
        threshold = rw * 0.4
        line_positions = np.where(proj > threshold)[0]

        if len(line_positions) > 0:
            # Use the line closest to the center
            center = (y2 - y1) // 2
            closest = line_positions[np.argmin(np.abs(line_positions - center))]
            return y1 + closest
        else:
            return (y1 + y2) // 2
    else:
        # Look for strong vertical lines
        rh = y2 - y1
        if rh < 5:
            return (x1 + x2) // 2
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, rh // 2)))
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        proj = np.sum(lines > 0, axis=0)

        # Find strongest vertical line position
        threshold = rh * 0.4
        line_positions = np.where(proj > threshold)[0]

        if len(line_positions) > 0:
            center = (x2 - x1) // 2
            closest = line_positions[np.argmin(np.abs(line_positions - center))]
            return x1 + closest
        else:
            return (x1 + x2) // 2


def _suppress_duplicate_regions(boxes, labels, scores, containment_thresh=0.7):
    """Suppress duplicate detections where same region gets different class labels.

    When two boxes of different classes nearly fully overlap (>70% containment),
    keep the one with higher confidence and suppress the other.
    This prevents the cross-class overlap resolver from incorrectly trimming.
    """
    n = len(boxes)

    # Pre-process Note vs Table overlap (clip Note if Table is nested inside it)
    for i in range(n):
        for j in range(n):
            if labels[i] == 2 and labels[j] == 3: # Note vs Table
                nx1, ny1, nx2, ny2 = boxes[i]
                tx1, ty1, tx2, ty2 = boxes[j]

                ix1 = max(nx1, tx1)
                iy1 = max(ny1, ty1)
                ix2 = min(nx2, tx2)
                iy2 = min(ny2, ty2)

                if ix1 < ix2 and iy1 < iy2:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    table_area = max((tx2 - tx1) * (ty2 - ty1), 1)

                    if inter_area / table_area > 0.6:  # Table is ~60%+ inside Note
                        # If Table is positioned at the bottom of the Note
                        if ty1 > ny1 + (ny2 - ny1) * 0.3:
                            boxes[i][3] = min(boxes[i][3], ty1 - 2)
                        # If Table is positioned at the top of the Note
                        elif ny2 > ty2 and ny1 >= ty1 - 10:
                            boxes[i][1] = max(boxes[i][1], ty2 + 2)

    keep = np.ones(n, dtype=bool)

    # Recalculate areas after clipping
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            
            # Allow SAME-CLASS suppression if one contains the other. (NMS normally handles this, but might fail on big vs tiny overlap IoUs).
            
            ix1 = max(boxes[i][0], boxes[j][0])
            iy1 = max(boxes[i][1], boxes[j][1])
            ix2 = min(boxes[i][2], boxes[j][2])
            iy2 = min(boxes[i][3], boxes[j][3])

            if ix1 >= ix2 or iy1 >= iy2:
                continue

            inter = (ix2 - ix1) * (iy2 - iy1)

            # Check if one box nearly contains the other
            containment_i = inter / max(areas[i], 1)  # How much of i is covered by j
            containment_j = inter / max(areas[j], 1)  # How much of j is covered by i

            if containment_i > containment_thresh or containment_j > containment_thresh:
                # Suppress the lower-confidence one
                if scores[i] >= scores[j]:
                    keep[j] = False
                else:
                    keep[i] = False

    return boxes[keep], labels[keep], scores[keep]


def _resolve_cross_class_overlap(boxes, labels, scores, iou_thresh=0.15):
    """Resolve overlaps between different classes using priority system.

    When boxes of different classes overlap:
    - Higher priority class keeps its box unchanged
    - Lower priority class's box is trimmed to remove the overlapping region
    - BUT only for partial overlaps (not near-duplicate regions)

    Priority: Table (3) > Note (2) > PartDrawing (1)
    """
    n = len(boxes)
    refined_boxes = boxes.copy()

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if labels[i] == labels[j]:
                continue  # Same class - handled by NMS

            # SPECIAL RULE: Protect PartDrawing (1) from ever being trimmed by others
            if labels[i] == 1:
                continue

            # Check overlap
            ix1 = max(refined_boxes[i][0], refined_boxes[j][0])
            iy1 = max(refined_boxes[i][1], refined_boxes[j][1])
            ix2 = min(refined_boxes[i][2], refined_boxes[j][2])
            iy2 = min(refined_boxes[i][3], refined_boxes[j][3])

            if ix1 >= ix2 or iy1 >= iy2:
                continue  # No overlap

            inter_area = (ix2 - ix1) * (iy2 - iy1)
            area_i = (refined_boxes[i][2] - refined_boxes[i][0]) * (refined_boxes[i][3] - refined_boxes[i][1])
            area_j = (refined_boxes[j][2] - refined_boxes[j][0]) * (refined_boxes[j][3] - refined_boxes[j][1])

            min_area = min(area_i, area_j)
            if min_area <= 0:
                continue

            overlap_ratio = inter_area / min_area
            if overlap_ratio < iou_thresh:
                continue

            # Only trim for PARTIAL overlaps, not when one box contains the other
            if overlap_ratio > 0.6:
                continue  # Near-containment handled by _suppress_duplicate_regions

            # Determine which box to trim (lower priority)
            pri_i = CLASS_PRIORITY.get(labels[i], 0)
            pri_j = CLASS_PRIORITY.get(labels[j], 0)

            if pri_i <= pri_j:
                # Trim box i (it has lower or equal priority)
                refined_boxes[i] = _trim_box_away_from(
                    refined_boxes[i], refined_boxes[j]
                )
            # Note: we don't trim j here; if j also needs trimming,
            # it will be handled when i is the "other" box in the loop

    return refined_boxes


def _trim_box_away_from(box_to_trim, priority_box):
    """Trim a box to remove overlap with a priority box.

    Strategy: find which side has the least overlap and trim from that side.
    This preserves the maximum area of the lower-priority box.
    """
    x1, y1, x2, y2 = box_to_trim
    px1, py1, px2, py2 = priority_box

    bw = x2 - x1
    bh = y2 - y1

    # Calculate how much we'd need to trim from each side
    # to eliminate the overlap
    trim_options = []

    # Trim from right (move x2 left to px1)
    if x2 > px1 and x1 < px1:
        trim_amount = x2 - px1
        remaining_area = (px1 - x1) * bh
        trim_options.append(('right', trim_amount, remaining_area,
                            [x1, y1, px1, y2]))

    # Trim from left (move x1 right to px2)
    if x1 < px2 and x2 > px2:
        trim_amount = px2 - x1
        remaining_area = (x2 - px2) * bh
        trim_options.append(('left', trim_amount, remaining_area,
                            [px2, y1, x2, y2]))

    # Trim from bottom (move y2 up to py1)
    if y2 > py1 and y1 < py1:
        trim_amount = y2 - py1
        remaining_area = bw * (py1 - y1)
        trim_options.append(('bottom', trim_amount, remaining_area,
                            [x1, y1, x2, py1]))

    # Trim from top (move y1 down to py2)
    if y1 < py2 and y2 > py2:
        trim_amount = py2 - y1
        remaining_area = bw * (y2 - py2)
        trim_options.append(('top', trim_amount, remaining_area,
                            [x1, py2, x2, y2]))

    if not trim_options:
        return box_to_trim  # No valid trim possible

    # Choose the trim that preserves the most area
    best = max(trim_options, key=lambda t: t[2])
    return best[3]


def _filter_invalid_boxes(boxes, labels, scores, img_h, img_w):
    """Remove boxes with invalid dimensions (zero area, out of bounds)."""
    valid_boxes = []
    valid_labels = []
    valid_scores = []

    for box, lbl, scr in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box

        # Clamp to image bounds
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        w = x2 - x1
        h = y2 - y1

        if w >= 10 and h >= 10:
            valid_boxes.append([x1, y1, x2, y2])
            valid_labels.append(lbl)
            valid_scores.append(scr)

    return (
        np.array(valid_boxes) if valid_boxes else np.zeros((0, 4)),
        np.array(valid_labels) if valid_labels else np.zeros((0,), dtype=int),
        np.array(valid_scores) if valid_scores else np.zeros((0,)),
    )
