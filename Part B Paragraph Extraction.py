import cv2
import numpy as np
import os
import argparse

def preprocess_image(image_path):
    """
    Loads an image, converts it to grayscale, and applies an inverted binary threshold.
    Text becomes white (255) and background becomes black (0).

    Args:
        image_path (str): The path to the input image.

    Returns:
        tuple: A tuple containing the original image, the grayscale image, 
               and the thresholded binary image. Returns (None, None, None) if file not found.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read image from {image_path}")
        return None, None, None
    
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding. THRESH_BINARY_INV makes text white and background black.
    # THRESH_OTSU automatically determines the optimal threshold value.
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    return original_image, gray, thresh

def remove_figures_and_tables(binary_img, config):
    """
    Uses a robust method to remove tables by finding the bounding box of the table grid.
    """
    cleaned_img = binary_img.copy()
    h, w = cleaned_img.shape

    # --- Table Removal Logic ---
    line_mask = np.zeros_like(binary_img)

    horizontal_kernel_len = int(w / config['LINE_REMOVAL_HORIZONTAL_KERNEL_RATIO'])
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_len, 1))
    detect_horizontal = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    contours_h, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(line_mask, contours_h, -1, (255, 255, 255), 2)

    vertical_kernel_len = int(h / config['LINE_REMOVAL_VERTICAL_KERNEL_RATIO'])
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_len))
    detect_vertical = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    contours_v, _ = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(line_mask, contours_v, -1, (255, 255, 255), 2)

    table_contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for tc in table_contours:
        if cv2.contourArea(tc) > config['TABLE_MIN_AREA_THRESHOLD']:
            x, y, w_c, h_c = cv2.boundingRect(tc)
            cv2.rectangle(cleaned_img, (x, y), (x + w_c, y + h_c), (0, 0, 0), -1)

    # --- Figure Removal Logic ---
    contours, _ = cv2.findContours(cleaned_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > (w * h * config['REMOVE_FIGURES_AREA_THRESHOLD_RATIO']): 
            x, y, w_c, h_c = cv2.boundingRect(c)
            box_area = w_c * h_c
            if box_area == 0: continue
            density = cv2.contourArea(c) / box_area
            if density > config['REMOVE_FIGURES_DENSITY_THRESHOLD']:
                cv2.drawContours(cleaned_img, [c], -1, (0, 0, 0), -1)

    return cleaned_img


def detect_columns(binary_img, min_gap_ratio=0.03):
    """
    Detects text columns in a binary image using vertical projection profile.
    """
    h, w = binary_img.shape
    min_gap_width = int(w * min_gap_ratio)

    vertical_projection = np.sum(binary_img, axis=0) / 255
    column_gaps = np.where(vertical_projection == 0)[0]

    if len(column_gaps) == 0:
        return [(0, w)]

    gap_diffs = np.diff(column_gaps)
    gap_breaks = np.where(gap_diffs > 1)[0]
    
    potential_splits = []
    start = column_gaps[0]
    for break_idx in gap_breaks:
        end = column_gaps[break_idx]
        if (end - start) > min_gap_width:
            potential_splits.append(start + (end - start) // 2)
        start = column_gaps[break_idx + 1]
    
    end = column_gaps[-1]
    if (end - start) > min_gap_width:
        potential_splits.append(start + (end - start) // 2)

    boundaries = sorted(list(set([0] + potential_splits + [w])))
    
    columns = []
    for i in range(len(boundaries) - 1):
        if (boundaries[i+1] - boundaries[i]) > min_gap_width:
            columns.append((boundaries[i], boundaries[i+1]))

    return columns

def group_lines_into_paragraphs(lines, median_line_height, config):
    """
    ### CHANGED ### - Groups lines into paragraphs using a simpler, more robust vertical spacing check.
    Relies on a stable median line height calculation.
    """
    if not lines or median_line_height == 0:
        return []

    # A gap is a paragraph break if it's significantly larger than the median line height.
    paragraph_gap_threshold = median_line_height * config['PARAGRAPH_GROUPING_GAP_FACTOR']
    
    paragraphs = []
    current_paragraph_lines = []
    if lines:
        current_paragraph_lines.append(lines[0])

    for i in range(1, len(lines)):
        prev_line = lines[i-1]
        current_line = lines[i]
        
        vertical_gap = current_line[1] - (prev_line[1] + prev_line[3])
        
        # If the gap between lines is too large, start a new paragraph.
        if vertical_gap > paragraph_gap_threshold:
            paragraphs.append(current_paragraph_lines)
            current_paragraph_lines = [current_line]
        else:
            # Otherwise, it's part of the same paragraph.
            current_paragraph_lines.append(current_line)
    
    if current_paragraph_lines:
        paragraphs.append(current_paragraph_lines)
    
    paragraph_boxes = []
    for para_lines in paragraphs:
        if not para_lines: continue
        x_coords = [line[0] for line in para_lines]
        y_coords = [line[1] for line in para_lines]
        w_coords = [line[0] + line[2] for line in para_lines]
        h_coords = [line[1] + line[3] for line in para_lines]
        
        min_x, min_y = min(x_coords), min(y_coords)
        max_x, max_y = max(w_coords), max(h_coords)
        
        paragraph_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))
        
    return paragraph_boxes

def extract_paragraphs_from_column(column_img, config, debug_path_prefix=None):
    """
    Finds and extracts paragraphs from a single column image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config['LINE_DILATION_KERNEL_WIDTH'], 3))
    dilated = cv2.dilate(column_img, kernel, iterations=1)
    
    if debug_path_prefix:
        cv2.imwrite(f"{debug_path_prefix}_dilated.png", dilated)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
        
    lines = [cv2.boundingRect(c) for c in contours]
    
    h, w = column_img.shape
    lines = [b for b in lines if b[2] * b[3] > config['LINE_EXTRACTION_MIN_AREA']]
    lines = [b for b in lines if b[3] < h * config['LINE_EXTRACTION_MAX_HEIGHT_RATIO']] 
    
    if not lines:
        return []
        
    lines.sort(key=lambda b: b[1])
    
    # ### CHANGED ### - Use median instead of mean for more robust line height calculation.
    median_line_height = np.median([b[3] for b in lines]) if lines else 0
    
    paragraph_boxes = group_lines_into_paragraphs(lines, median_line_height, config)
    
    return paragraph_boxes

def run_extraction(args):
    """
    Main function to run the paragraph extraction process.
    """
    input_dir, output_dir = args.input, args.output
    
    # ### CHANGED ### - Re-tuned config values for the new median-based logic.
    config = {
        'REMOVE_FIGURES_DENSITY_THRESHOLD': 0.7,
        'REMOVE_FIGURES_AREA_THRESHOLD_RATIO': 0.02,
        'LINE_REMOVAL_HORIZONTAL_KERNEL_RATIO': 40,
        'LINE_REMOVAL_VERTICAL_KERNEL_RATIO': 40,
        'TABLE_MIN_AREA_THRESHOLD': 4000, 
        'COLUMN_DETECTION_MIN_GAP_RATIO': 0.03,
        'PARAGRAPH_GROUPING_GAP_FACTOR': 1.5, # Tuned for median line height.
        'PARAGRAPH_MIN_WIDTH': 100,
        'PARAGRAPH_MIN_HEIGHT': 20,
        'LINE_EXTRACTION_MIN_AREA': 500,
        'LINE_EXTRACTION_MAX_HEIGHT_RATIO': 0.1,
        'LINE_DILATION_KERNEL_WIDTH': 25,
    }

    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    except FileNotFoundError:
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    if not image_files:
        print(f"No images found in '{input_dir}'.")
        return

    total_paragraphs_extracted = 0
    for filename in image_files:
        print(f"\n--- Processing {filename} ---")
        image_path = os.path.join(input_dir, filename)
        
        original_img, _, binary_img = preprocess_image(image_path)
        if original_img is None:
            continue

        base_filename = os.path.splitext(filename)[0]
        file_output_dir = os.path.join(output_dir, base_filename)
        if not os.path.exists(file_output_dir):
            os.makedirs(file_output_dir)

        if args.debug:
            cv2.imwrite(os.path.join(file_output_dir, f"{base_filename}_0_binary.png"), binary_img)

        # 1. Clean the image
        text_only_img = remove_figures_and_tables(binary_img, config)
        
        if args.debug:
            cv2.imwrite(os.path.join(file_output_dir, f"{base_filename}_1_cleaned.png"), text_only_img)
        
        # 2. Detect text columns
        columns = detect_columns(text_only_img, config['COLUMN_DETECTION_MIN_GAP_RATIO'])
        print(f"  Found {len(columns)} column(s).")
        
        # 3. Extract paragraphs from each column
        all_paragraph_boxes = []
        for col_idx, (start_x, end_x) in enumerate(columns):
            margin = 5
            col_start = max(0, start_x - margin)
            col_end = min(text_only_img.shape[1], end_x + margin)
            
            column_img = text_only_img[:, col_start:col_end]
            
            debug_prefix = None
            if args.debug:
                debug_prefix = os.path.join(file_output_dir, f"{base_filename}_col{col_idx}")

            paragraph_boxes_in_col = extract_paragraphs_from_column(column_img, config, debug_prefix)
            
            for p_box in paragraph_boxes_in_col:
                x, y, w, h = p_box
                if w < config['PARAGRAPH_MIN_WIDTH'] or h < config['PARAGRAPH_MIN_HEIGHT']: 
                    continue
                
                abs_x = x + col_start
                all_paragraph_boxes.append((abs_x, y, w, h))

        # 4. Save the extracted paragraphs
        print(f"  Extracting and saving {len(all_paragraph_boxes)} paragraphs to '{file_output_dir}'...")
        paragraph_count_for_file = 0
        for i, (x, y, w, h) in enumerate(all_paragraph_boxes):
            paragraph_img = original_img[y:y+h, x:x+w]
            
            if paragraph_img.size == 0:
                print(f"    Skipping empty paragraph crop for box {i+1}.")
                continue

            output_filename = f"paragraph_{paragraph_count_for_file + 1}.png"
            output_path = os.path.join(file_output_dir, output_filename)
            
            cv2.imwrite(output_path, paragraph_img)
            paragraph_count_for_file += 1
        
        total_paragraphs_extracted += paragraph_count_for_file
            
    print("\n--- Processing complete! ---")
    print(f"Extracted a total of {total_paragraphs_extracted} paragraphs.")
    print(f"All extracted paragraphs have been saved to '{args.output}'.")
    if args.debug:
        print("Debug images have been saved to the respective output sub-directories.")

# --- Run the main function ---
if __name__ == '__main__':
    DEFAULT_INPUT_DIR = r'C:\Users\User\Desktop\Digital Image\Group Assignment\CSC2014- Group Assignment_Aug-2025\Converted Paper (8)'
    DEFAULT_OUTPUT_DIR = r'C:\Users\User\Desktop\Digital Image\Group Assignment\CSC2014- Group Assignment_Aug-2025\Extracted Paragraphs'
    
    parser = argparse.ArgumentParser(description="Extract paragraphs from document images.")
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_DIR,
                        help=f"Path to the directory containing input images. Defaults to: {DEFAULT_INPUT_DIR}")
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Path to the directory where extracted paragraphs will be saved. Defaults to: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument('--debug', action='store_true',
                        help="If set, save intermediate processing images for debugging.")
    
    args = parser.parse_args()
    
    run_extraction(args)
