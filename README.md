# Paragraph-Extracter

Document Paragraph Extractor
This Python program is a sophisticated tool designed to extract individual paragraphs from document images. Leveraging the OpenCV library, it automates the process of identifying, segmenting, and saving textual content while intelligently disregarding non-textual elements like figures and tables.

Features
Robust Preprocessing: Converts input images to grayscale and applies an inverted binary threshold to ensure text is white on a black background, optimizing it for subsequent analysis.

Intelligent Figure and Table Removal: Employs an advanced method to detect and remove tables based on their grid-like structure (horizontal and vertical lines) and removes figures by analyzing contour area and density. This ensures that only textual content remains for paragraph extraction.

Multi-Column Layout Detection: Accurately identifies and segments multiple text columns within a document image using vertical projection profiles, adapting to various document layouts.

Paragraph Grouping with Adaptive Logic: Groups individual text lines into coherent paragraphs based on vertical spacing. It utilizes the median line height to dynamically determine paragraph breaks, making the grouping more robust to variations in font size or line spacing.

Configurable Parameters: All key thresholds and kernel sizes are defined in a config dictionary, allowing for easy tuning and optimization for different document types.

Batch Processing: Processes multiple images from a specified input directory, organizing extracted paragraphs into subfolders for each original image.

Debug Mode: Includes an optional debug mode that saves intermediate processing steps (e.g., binary image, cleaned image, dilated column images), which is invaluable for understanding and refining the extraction process.

How It Works
The core functionality is encapsulated within the run_extraction function, which orchestrates the entire pipeline:

Image Loading & Binarization: Reads the input image, converts it to grayscale, and applies an Otsu's threshold to create a clean binary representation of the text.

Noise Reduction: The remove_figures_and_tables function takes the binary image and strategically eliminates large, dense contours (assumed figures) and grid-like structures (tables) using morphological operations and contour analysis.

Column Segmentation: detect_columns analyzes the vertical projection profile of the cleaned image to identify significant vertical white spaces, indicating boundaries between text columns.

Line and Paragraph Detection: For each detected column:

Line Extraction: Text lines are identified by dilating the column image and finding contours.

Paragraph Grouping: The group_lines_into_paragraphs function then groups these individual lines into logical paragraphs by comparing the vertical gap between lines against a factor of the median line height.

Output Saving: Each identified paragraph is cropped from the original image (to retain its full visual quality) and saved as a separate PNG file in a dedicated output directory structure.
