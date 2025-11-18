# OVERVIEW
# This file is a comprehensive script designed to automate the extraction 
# of measurement data from images of scuba diving equipment displays (specifically, "Petrel" and 
# "Gauge" screens). The script processes batches of images, applies image transformations to isolate 
# and enhance relevant display regions, performs OCR (Optical Character Recognition) to extract 
# numerical readings, and saves the results to a CSV file. It also provides interactive tools for 
# initial calibration (masking/cropping regions of interest) and visual feedback for the user. The 
# script is intended for both first-time setup (to define masks) and repeated batch processing of 
# new images.

# SUMMARY
# This file is a robust, semi-automated tool for extracting and analyzing measurement data from 
# images of scuba diving equipment displays. It combines interactive calibration, advanced image 
# processing, OCR, and data visualization, making it suitable for both initial setup and routine 
# data extraction tasks in a research or fieldwork context.

# PARAMETER AND PATH SETUP:
# - Defines directories for temporary data and images.
# - Sets up color codes for terminal output and parameters for cropping and image processing.

# HSV CALIBRATION AND MASKING:
# VariabiliHSV(Schermo): Returns HSV color thresholds for different screen types, used to isolate 
# display digits via color filtering.

# IMAGE PROCESSING FUNCTIONS:
# - order_points(pts), deskew_and_crop(image_path, pts): Utilities to deskew and crop images based on 
#   user-selected points, ensuring the region of interest is properly aligned.
# - HSVprocess(inputImage, ...): Applies HSV masking, inversion, grayscale conversion, and contrast 
#   enhancement to prepare images for OCR.

# OCR FUNCTIONS:
# - OCRPetrel(NomeImmagine): Uses Tesseract OCR to extract readings from Petrel images, with regex 
#   filtering for expected formats.
# - OCRGauge(NomeImmagine): Uses the external ssocr tool to extract readings from Gauge images, with 
#   error handling and regex extraction.

# INTERACTIVE MASKING AND CALIBRATION:
# - GrabPoints(Schermo, Immagine): Interactive OpenCV window for the user to select four points 
#   defining the region of interest for each screen type. Saves these points for future runs.
# - ClickWithMouse, ZoomTopLeft: Provide visual feedback and magnification to assist accurate point 
#   selection.

# BATCH PROCESSING AND CSV OUTPUT:
# - ProcessOneImage(): Orchestrates the full pipeline for a single image: cropping, processing, OCR, 
#   and appending results to a CSV.
# - Main script logic: Handles first-time setup (mask creation) and subsequent batch processing of all 
#   images in the directory, writing results to 20_Measurements.csv.

# VISUALIZATION AND PLOTTING:
# - VisualizeStep: Utility to display images at various processing stages.
# - At the end, uses pandas and matplotlib to load the CSV and plot the extracted measurements for 
#   analysis.

# FIRST-TIME SETUP AND REUSABILITY:
# The script distinguishes between first-time calibration (mask creation) and regular batch 
# processing, allowing for easy reuse once masks are defined.

# ERROR HANDLING AND USER GUIDANCE:
# Provides clear terminal messages, error reporting, and guidance for missing files or directories.



TempFolder = 'Temp data/' # not used yet
ImgDirectory = 'Images/'

# The size of the text is one of the most important parameters for OCR. Adjust these
# if the OCR is making mistakes
Petrel_max_height=100
Gauge_max_height=70

import os

# Delete all images in TempFolder before processing
for f in os.listdir(TempFolder):
    file_path = os.path.join(TempFolder, f)
    if os.path.isfile(file_path) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Could not delete {file_path}: {e}")


#from PIL import Image, ImageEnhance, ImageOps
import cv2, os, csv, glob

EXTRACT_SINGLE_IMAGE = os.environ.get("EXTRACT_SINGLE_IMAGE")

def resize_if_needed(image, max_height=500, max_width=500):
    h, w = image.shape[:2]
    if h > max_height or w > max_width:
        scale = min(max_height / h, max_width / w)
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

### PARAMETERS TO ADJUST WHEN YOU DO A NEW CALIBRATION ###################

import json
CROP_PARAMS_FILE = "10_Crop_Gauge.json"
if os.path.exists(CROP_PARAMS_FILE):
    with open(CROP_PARAMS_FILE, "r") as f:
        crop_params = json.load(f)
    crop_top_perc = crop_params.get("crop_top_perc", 0.00)
    crop_left_perc = crop_params.get("crop_left_perc", 0.01)
    crop_right_perc = crop_params.get("crop_right_perc", 0.01)
    crop_bottom_perc = crop_params.get("crop_bottom_perc", 0.25)
else:
    crop_top_perc = 0.00    # Remove 0% from the top
    crop_left_perc = 0.01   # Remove 1% from the left
    crop_right_perc = 0.01  # Remove 1% from the right
    crop_bottom_perc = 0.25 # Remove 25% from the bottom

# Font colors
DOMANDA = "\033[1;32m"  # Bold green
REGOLARE = "\033[0m" # Regular white
HIGHLIGHT = "\033[1;38;5;208m" #Bold orange
ERRORE = "\033[1;31m" #Bold red

def VariabiliHSV(Schermo):  # Adjust with 0AdjustHSVMask.py
    # Returns HSV min/max values as a tuple: (hMin, sMin, vMin, hMax, sMax, vMax)
    import os
    import re

    def load_hsv_from_file(filename):
        if not os.path.isfile(filename):
            return None
        try:
            with open(filename, "r") as f:
                lines = f.readlines()
                values = []
                for line in lines:
                    # Accepts lines like "hMin = 0"
                    match = re.match(r"\s*\w+\s*=\s*(\d+)", line)
                    if match:
                        values.append(int(match.group(1)))
                if len(values) == 6:
                    return tuple(values)
        except Exception:
            pass
        return None

    if Schermo == "Petrel":
        hsv = load_hsv_from_file("10_HSV_Values_Petrel.txt")
        if hsv:
            print(f"{HIGHLIGHT}Using Petrel HSV values from 10_HSV_Values_Petrel.txt: {hsv}{REGOLARE}")
            return hsv
        # Default values if file not found
        hMin = 0
        sMin = 122
        vMin = 179
        hMax = 90
        sMax = 255
        vMax = 255
        print(f"{HIGHLIGHT}Using default Petrel HSV values: {(hMin, sMin, vMin, hMax, sMax, vMax)}{REGOLARE}")
        return hMin, sMin, vMin, hMax, sMax, vMax
    elif Schermo == "Gauge":
        hsv = load_hsv_from_file("10_HSV_Values_Gauge.txt")
        if hsv:
            print(f"{HIGHLIGHT}Using Gauge HSV values from 10_HSV_Values_Gauge.txt: {hsv}{REGOLARE}")
            return hsv
        # Default values if file not found
        hMin = 0
        sMin = 0
        vMin = 225
        hMax = 179
        sMax = 255
        vMax = 255
        print(f"{HIGHLIGHT}Using default Gauge HSV values: {(hMin, sMin, vMin, hMax, sMax, vMax)}{REGOLARE}")
        return hMin, sMin, vMin, hMax, sMax, vMax
    else:
        raise ValueError("Wrong input for VariabiliHSV")



### FUNCTIONS ############################################################

def order_points(pts):
    import numpy as np
    # Sort points as: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def deskew_and_crop(image_path, pts):
    import numpy as np
    # Load image
    image = cv2.imread(image_path)

    # Convert points to numpy array and order them
    pts = np.array(pts, dtype="float32")
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate new image dimensions
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = max(int(height_left), int(height_right))

    # Destination points (perfect rectangle)
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def load_all_mask_params(filename):
    """Load all mask parameters from a JSON file into a dictionary."""
    import json
    if not os.path.isfile(filename):
        return {}
    with open(filename, "r") as f:
        return json.load(f)

def apply_mask(image, params, mask_type):
    import numpy as np
    import cv2

    # Morphological parameters (default if not present)
    morph_kernel = int(params.get("Morph_Kernel", 3))
    morph_iter = int(params.get("Morph_Iter", 1))
    if morph_kernel < 1:
        morph_kernel = 1
    if morph_iter < 0:
        morph_iter = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))

    def morph_open(mask):
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iter)

    if mask_type == "HSV":
        hMin = params.get("hMin", 0)
        sMin = params.get("sMin", 0)
        vMin = params.get("vMin", 0)
        hMax = params.get("hMax", 179)
        sMax = params.get("sMax", 255)
        vMax = params.get("vMax", 255)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])
        mask = cv2.inRange(hsv, lower, upper)
        # Do NOT apply morph_open to HSV mask
        result = cv2.bitwise_and(image, image, mask=mask)
        result = cv2.medianBlur(result, 5)
        return result

    elif mask_type == "LAB":
        lab_LMin = params.get("LAB_LMin", 0)
        lab_LMax = params.get("LAB_LMax", 255)
        lab_AMin = params.get("LAB_AMin", 0)
        lab_AMax = params.get("LAB_AMax", 255)
        lab_BMin = params.get("LAB_BMin", 0)
        lab_BMax = params.get("LAB_BMax", 255)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        lab_mask = (
            cv2.inRange(L, lab_LMin, lab_LMax)
            & cv2.inRange(A, lab_AMin, lab_AMax)
            & cv2.inRange(B, lab_BMin, lab_BMax)
        )
        # Do NOT apply morph_open to LAB mask
        result = cv2.bitwise_and(image, image, mask=lab_mask)
        return result

    elif mask_type == "RGB":
        rgb_RMin = params.get("RGB_RMin", 0)
        rgb_RMax = params.get("RGB_RMax", 255)
        rgb_GMin = params.get("RGB_GMin", 0)
        rgb_GMax = params.get("RGB_GMax", 255)
        rgb_BMin = params.get("RGB_BMin", 0)
        rgb_BMax = params.get("RGB_BMax", 255)
        rgb_mask = cv2.inRange(
            image,
            (rgb_BMin, rgb_GMin, rgb_RMin),
            (rgb_BMax, rgb_GMax, rgb_RMax)
        )
        # Do NOT apply morph_open to RGB mask
        result = cv2.bitwise_and(image, image, mask=rgb_mask)
        return result

    elif mask_type == "YCrCb":
        yMin = params.get("YMin", 0)
        yMax = params.get("YMax", 255)
        crMin = params.get("CrMin", 0)
        crMax = params.get("CrMax", 255)
        cbMin = params.get("CbMin", 0)
        cbMax = params.get("CbMax", 255)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)
        ycrcb_mask = (
            cv2.inRange(Y, yMin, yMax)
            & cv2.inRange(Cr, crMin, crMax)
            & cv2.inRange(Cb, cbMin, cbMax)
        )
        # Do NOT apply morph_open to YCrCb mask
        result = cv2.bitwise_and(image, image, mask=ycrcb_mask)
        return result

    elif mask_type == "SV":
        s_only_min = params.get("S_Only_Min", 0)
        s_only_max = params.get("S_Only_Max", 255)
        v_only_min = params.get("V_Only_Min", 0)
        v_only_max = params.get("V_Only_Max", 255)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]
        sv_mask = cv2.inRange(S, s_only_min, s_only_max) & cv2.inRange(V, v_only_min, v_only_max)
        # Do NOT apply morph_open to SV mask
        result = cv2.bitwise_and(image, image, mask=sv_mask)
        return result

    elif mask_type == "Adaptive":
        adaptive_block = params.get("Adaptive_Block", 21)
        adaptive_C = params.get("Adaptive_C", 10)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if adaptive_block % 2 == 0:
            adaptive_block += 1
        if adaptive_block < 3:
            adaptive_block = 3
        adaptive_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, adaptive_block, adaptive_C
        )
        adaptive_mask = morph_open(adaptive_mask)
        result = cv2.bitwise_and(image, image, mask=adaptive_mask)
        return result

    elif mask_type == "LAB_CLEAN":
        # LAB with morphological cleaning (example for green)
        lab_LMin = params.get("LAB_LMin", 0)
        lab_LMax = params.get("LAB_LMax", 255)
        lab_AMin = params.get("LAB_AMin", 0)
        lab_AMax = params.get("LAB_AMax", 255)
        lab_BMin = params.get("LAB_BMin", 0)
        lab_BMax = params.get("LAB_BMax", 255)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        lab_mask = (
            cv2.inRange(L, lab_LMin, lab_LMax)
            & cv2.inRange(A, lab_AMin, lab_AMax)
            & cv2.inRange(B, lab_BMin, lab_BMax)
        )
        lab_mask_clean = morph_open(lab_mask)
        result = cv2.bitwise_and(image, image, mask=lab_mask_clean)
        return result

    else:
        raise ValueError(f"Unknown mask type: {mask_type}")


def HSVprocess(inputImage, params, mask_type):
    from PIL import Image, ImageEnhance, ImageOps
    import numpy as np
    import os
    import re

    # Ensure TempFolder exists
    if not os.path.exists(TempFolder):
        os.makedirs(TempFolder)

    image = cv2.imread(inputImage)
    result = apply_mask(image, params, mask_type)

    # Save image
    base = os.path.splitext(os.path.basename(inputImage))[0]
    base = re.sub(r'_1_Cut$', '', base)
    ext = os.path.splitext(inputImage)[1]
    new_filename = os.path.join(TempFolder, f"{base}_2_HSV{ext}")
    cv2.imwrite(new_filename, result)

    # Open and further process the image
    from PIL import Image, ImageEnhance, ImageOps
    img = Image.open(new_filename)
    img = ImageOps.invert(img.convert('RGB')) #invert
    img = img.convert('L')  # Convert to grayscale
    img = ImageEnhance.Contrast(img).enhance(2.0)

    # Save the processed image to check it
    new_filename = os.path.join(TempFolder, f"{base}_3_BW{ext}")
    img.save(new_filename)
    return result


def OCRPetrel(NomeImmagine):
    from PIL import Image, ImageEnhance, ImageOps
    import pytesseract, re
    # OCR
    Image_Right = Image.open(NomeImmagine)
    # Restrict Tesseract to only digits, dot, and space
    custom_config = r'--psm 7'# -c tessedit_char_whitelist=0123456789. '
    text = pytesseract.image_to_string(Image_Right, config=custom_config)
    print("Petrel raw output: ", repr(text))

    # Use regex to extract numbers: 1-3 digits before decimal, 0 or 1 after
    numbers = re.findall(r'\d{1,3}(?:\.\d)?', text)
    if len(numbers) == 3:
        vector = numbers
        print("Petrel cleaned: ", repr(vector))
    else:
        vector = []
        print(f"{ERRORE}No match{REGOLARE}")
    return vector


def OCRGauge(NomeImmagine): # OCR Gauge
    import re, subprocess, os

    # Ensure the path uses the correct TempFolder
    NomeImmagine = os.path.normpath(NomeImmagine)

    # OCR
    command = f"ssocr -d -1 -c decimal \"{NomeImmagine}\""
    try:
        result = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as e:
        print(f"{ERRORE}ssocr failed for {NomeImmagine} with error code {e.returncode}{REGOLARE}")
        # Use e.output as the result, if available
        result = e.output.strip() if hasattr(e, "output") and e.output else ""
        print(f"{HIGHLIGHT}ssocr output:{REGOLARE}", result)

    print(f"ssocr raw output: {result}")  # Show exactly what ssocr is seeing

    # Remove underscores and keep only the rightmost decimal point (if multiple)
    cleaned = result.replace("_", "")
    # Replace multiple consecutive dots with a single dot
    cleaned = re.sub(r"\.+", ".", cleaned)
    # If more than one dot, keep only the rightmost as the decimal point
    if cleaned.count(".") > 1:
        # Remove all but the last dot
        parts = cleaned.split(".")
        cleaned = "".join(parts[:-2]) + parts[-2] + "." + parts[-1]

    # Now extract the rightmost 1 or 2 digits before the decimal and exactly 2 after
    match = re.search(r"(\d{1,2})\.(\d{2})$", cleaned)
    if match:
        integer_part = match.group(1)
        decimal_part = match.group(2)
        extracted_text = f"{int(integer_part)}.{decimal_part}"
        print("Gauge cleaned: ", repr(extracted_text))
        return extracted_text

    # If not found, try to find any such pattern in the string (rightmost)
    matches = re.findall(r"(\d{1,2})\.(\d{2})", cleaned)
    if matches:
        integer_part, decimal_part = matches[-1]
        extracted_text = f"{int(integer_part)}.{decimal_part}"
        print("Gauge cleaned: ", repr(extracted_text))
        return extracted_text

    print(f"{ERRORE}No match found.{REGOLARE}")
    return None


# --- Read mask type from 10_HSV_Values_X.txt for Petrel and Gauge and display in terminal ---
mask_types = [
    "HSV", "LAB", "RGB", "YCrCb", "SV", "Adaptive", "LAB_CLEAN"
]

def get_best_mask_type(filename, screen_name):
    params = load_all_mask_params(filename)
    best_mask = params.get("BestMask", None)
    if best_mask is None:
        print(f"{ERRORE}No BestMask found in {filename}. Defaulting to HSV for {screen_name}.{REGOLARE}")
        return "HSV"
    if best_mask not in mask_types:
        print(f"{ERRORE}BestMask '{best_mask}' in {filename} is not a recognized mask type. Defaulting to HSV for {screen_name}.{REGOLARE}")
        return "HSV"
    print(f"{HIGHLIGHT}Using mask type for {screen_name}: {best_mask} (from {filename}){REGOLARE}")
    return best_mask

petrel_mask_type = get_best_mask_type("10_HSV_Values_Petrel.json", "Petrel")
gauge_mask_type = get_best_mask_type("10_HSV_Values_Gauge.json", "Gauge")


def ProcessOneImage():
    from PIL import Image, ImageEnhance, ImageOps
    import json
    import os
    import re

    # Use the global filename patterns set at the top-level
    global PETREL_CUT_PATTERN, PETREL_BW_PATTERN, GAUGE_CUT_PATTERN, GAUGE_BW_PATTERN

    # Ensure TempFolder exists
    if not os.path.exists(TempFolder):
        os.makedirs(TempFolder)

    # Get a unique base name for this image (without extension)
    base_name = os.path.splitext(os.path.basename(Immagine))[0]

    # Generate unique filenames for this image
    gauge_cut = GAUGE_CUT_PATTERN.format(base=base_name)
    gauge_bw = GAUGE_BW_PATTERN.format(base=base_name)
    petrel_cut = PETREL_CUT_PATTERN.format(base=base_name)
    petrel_bw = PETREL_BW_PATTERN.format(base=base_name)

    # Extract measurements from Petrel
    print(f"\nFile:", repr(Immagine))
    with open('10_Mask_Petrel.json', 'r') as f: # Load points from the JSON file
        points = json.load(f)  # This will be a list of [x, y] lists
    input_points = [tuple(pt) for pt in points] # Convert points into tuples
    result_petrel = deskew_and_crop(Immagine, input_points) # Cut and deskew image
    temp_cut_petrel = os.path.join(TempFolder, petrel_cut)
    cv2.imwrite(temp_cut_petrel, result_petrel)

    # Load all mask parameters for Petrel and Gauge
    petrel_params = load_all_mask_params("10_HSV_Values_Petrel.json")
    gauge_params = load_all_mask_params("10_HSV_Values_Gauge.json")

    # Use the mask_type selected at the start of the script
    result_hsv_petrel = HSVprocess(temp_cut_petrel, petrel_params, petrel_mask_type)

    # Resize BW image before OCR (Petrel)
    temp_bw_petrel = os.path.join(TempFolder, petrel_bw)
    # Only resize the BW image before OCR
    img_bw_petrel = cv2.imread(temp_bw_petrel)
    img_bw_petrel = resize_if_needed(img_bw_petrel, max_height=Petrel_max_height, max_width=500)
    cv2.imwrite(temp_bw_petrel, img_bw_petrel)
    Petrel = OCRPetrel(temp_bw_petrel)

    # Extract measurements from Gauge
    with open('10_Mask_Gauge.json', 'r') as f: # Load points from the JSON file
        points = json.load(f)  # This will be a list of [x, y] lists
    input_points = [tuple(pt) for pt in points] # Convert points into tuples
    result_gauge = deskew_and_crop(Immagine, input_points) # Cut and deskew image
    temp_cut_gauge = os.path.join(TempFolder, gauge_cut)
    cv2.imwrite(temp_cut_gauge, result_gauge)

    result_hsv_gauge = HSVprocess(temp_cut_gauge, gauge_params, gauge_mask_type)

    temp_bw_gauge = os.path.join(TempFolder, gauge_bw)
    MoreProcessGauge(temp_bw_gauge)

     # Invert BW image for Gauge before OCR ONLY if not using Adaptive mask
    img_bw_gauge = cv2.imread(temp_bw_gauge, cv2.IMREAD_GRAYSCALE)
    if gauge_mask_type != "Adaptive":
        img_bw_gauge = cv2.bitwise_not(img_bw_gauge)
    img_bw_gauge = resize_if_needed(img_bw_gauge, max_height=Gauge_max_height, max_width=500)
    cv2.imwrite(temp_bw_gauge, img_bw_gauge)

    # OCR Gauge
    Gauge = OCRGauge(temp_bw_gauge) # OCR Gauge

    # Write new line with measurements
    csv_filename = '20_Measurements.csv'
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Immagine', 'Gauge'] + [f'Petrel_{i+1}' for i in range(len(Petrel))])
        writer.writerow([Immagine, Gauge] + Petrel) 
    return Gauge, Petrel


def GrabPoints(Schermo, Immagine):
    import json
    # Only read points from JSON, do not allow interactive selection
    filename = f'10_Mask_{Schermo}.json'
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Mask file '{filename}' not found. Please create it first using the mask selection tool.")
    with open(filename, 'r') as f:
        points = json.load(f)
    return points


def MoreProcessGauge(NomeImmagine, threshold=100):
    from PIL import Image, ImageEnhance, ImageOps

    img = Image.open(NomeImmagine)
    img = img.convert('L')
    img = img.point(lambda p: 255 if p > threshold else 0)
    img.save(NomeImmagine)

    img = Image.open(NomeImmagine)
    width, height = img.width, img.height

    left = int(width * crop_left_perc)
    right = width - int(width * crop_right_perc)
    top = int(height * crop_top_perc)
    bottom = height - int(height * crop_bottom_perc)

    # Ensure we don't go out of bounds
    right = min(right, width)
    bottom = min(bottom, height)

    crop_box = (left, top, right, bottom)
    img = img.crop(crop_box)
    img.save(NomeImmagine)


def RunHSVAndVisualize(screen, cut_filename, bw_filename, do_more_process=False):
    # RunHSV returns: result_path, result_img, hMin, sMin, vMin, hMax, sMax, vMax
    img_cut_path, result_img, hMin, sMin, vMin, hMax, sMax, vMax = RunHSV(screen, cut_filename)
    if do_more_process:
        MoreProcessGauge(os.path.join(TempFolder, bw_filename))
    image = VisualizeStep(img_cut_path, "Cut")
    image = VisualizeStep(os.path.join(TempFolder, bw_filename), "BW")
    return img_cut_path


def RunHSV(screen_type, cut_image_path):
    hMin, sMin, vMin, hMax, sMax, vMax = VariabiliHSV(screen_type)
    result_path = os.path.join(TempFolder, cut_image_path)
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Input image file does not exist: {result_path}")
    result_img = HSVprocess(result_path, hMin, sMin, vMin, hMax, sMax, vMax)
    return result_path, result_img, hMin, sMin, vMin, hMax, sMax, vMax


def VisualizeStep(arg0, arg1, wait_ms=800):
    # VisualizeStep(img_cut_path, "Cut", wait_ms=1000)  # Show for 1 second
    # VisualizeStep(img_cut_path, "Cut", wait_ms=0)     # Wait for key press
    # VisualizeStep(img_cut_path, "Cut", wait_ms=1)     # Show very briefly
    result = cv2.imread(arg0)
    cv2.imshow(arg1, result)
    cv2.waitKey(wait_ms)
    cv2.destroyAllWindows()
    return result


def ClickWithMouse(mouse_pos, image, original_image):
    # sourcery skip: move-assign
    x, y = mouse_pos
    crosshair_size = 120  # Crosshair length on main image
    cv2.line(image, (x - crosshair_size, y), (x + crosshair_size, y), (0, 0, 255), 7)
    cv2.line(image, (x, y - crosshair_size), (x, y + crosshair_size), (0, 0, 255), 7)
    # Magnify area around cursor
    mag_size = 200   # Size of area to magnify (sampled area)
    mag_factor = 5  # Magnification factor
    half = mag_size // 2
    h, w = image.shape[:2]
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)
    roi = original_image[y1:y2, x1:x2]
    if roi.size > 0:
        ZoomTopLeft(roi, mag_size, mag_factor, image)


def ZoomTopLeft(roi, mag_size, mag_factor, image):
    # Increase contrast of the ROI using CLAHE
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    contrast_roi = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    zoomed = cv2.resize(contrast_roi, (mag_size * mag_factor, mag_size * mag_factor), interpolation=cv2.INTER_NEAREST)
    # Draw crosshair in the magnification square
    zh, zw = zoomed.shape[:2]
    center_x, center_y = zw // 2, zh // 2
    crosshair_length = 300  # Crosshair length in magnification square
    crosshair_color = (0, 0, 255)
    thickness = 10
    cv2.line(zoomed, (center_x - crosshair_length, center_y), (center_x + crosshair_length, center_y), crosshair_color, thickness)
    cv2.line(zoomed, (center_x, center_y - crosshair_length), (center_x, center_y + crosshair_length), crosshair_color, thickness)
    # Draw a dot in the middle of the crosshair
    cv2.circle(zoomed, (center_x, center_y), 30, crosshair_color, -1)
    # Add a border to the magnified image
    border_color = (50, 0, 0)
    border_thickness = 30
    cv2.rectangle(zoomed, (0, 0), (zw-1, zh-1), border_color, border_thickness)
    # Overlay the zoomed area on the main image (top-left corner)
    img_h, img_w = image.shape[:2]
    overlay_x, overlay_y = 10, 10
    if overlay_y + zh < img_h and overlay_x + zw < img_w:
        image[overlay_y:overlay_y+zh, overlay_x:overlay_x+zw] = zoomed



### SCRIPT ###############################################################
os.system('cls' if os.name == 'nt' else 'clear') # Clear terminal
PETREL_CUT_PATTERN = "21_Temp_Petrel_{base}_1_Cut.jpg"
PETREL_HSV_PATTERN = "21_Temp_Petrel_{base}_2_HSV.jpg"
PETREL_BW_PATTERN = "21_Temp_Petrel_{base}_3_BW.jpg"
GAUGE_CUT_PATTERN = "22_Temp_Gauge_{base}_1_Cut.jpg"
GAUGE_HSV_PATTERN = "22_Temp_Gauge_{base}_2_HSV.jpg"
GAUGE_BW_PATTERN = "22_Temp_Gauge_{base}_3_BW.jpg"


# === SUBSEQUENT RUNS CODE ===
print(f"\n{HIGHLIGHT} ---- LOOP THROUGH IMAGES ---- {REGOLARE}\n")

# Delete all images in TempFolder before processing
for f in os.listdir(TempFolder):
    file_path = os.path.join(TempFolder, f)
    if os.path.isfile(file_path) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"{ERRORE}Could not delete {file_path}: {e}{REGOLARE}")

# Start CSV file
with open('20_Measurements.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'Gauge', 'Sensor 1', 'Sensor 2', 'Sensor 3']) # Write header

# Loop all images
input_directory = os.path.join(os.path.dirname(__file__), ImgDirectory)
if not os.path.exists(input_directory) or not os.path.isdir(input_directory):
    print(f"{ERRORE}Error: Image directory does not exist.\n  Constructed path: '{input_directory}'\n  ImgDirectory value: '{ImgDirectory}'{REGOLARE}")
else:
    all_files = os.listdir(input_directory)
    print(f"{HIGHLIGHT}All images found in folder:{REGOLARE}", all_files)
    print(f"{HIGHLIGHT}Mask type for Gauge: {REGOLARE}{gauge_mask_type}")
    print(f"{HIGHLIGHT}Mask type for Petrel: {REGOLARE}{petrel_mask_type}")
    if image_files := sorted(
        [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg'))]
    ):
        for filename in image_files:
            if EXTRACT_SINGLE_IMAGE and filename != EXTRACT_SINGLE_IMAGE:
                continue
            Immagine = os.path.join(ImgDirectory, filename)
            Gauge, Petrel = ProcessOneImage()
        print(f"{HIGHLIGHT}\nLoop finished!{REGOLARE}\n")
    else:
        print(f"{ERRORE}Warning: No image files found in '{input_directory}'.{REGOLARE}")



# Production: Load the CSV file
# Only visualize the plot if not running in "single image" mode (i.e., not called from 56_AdjustGaugeCrop)
if not EXTRACT_SINGLE_IMAGE:
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv('20_Measurements.csv')

# Demo: Create a sample DataFrame similar to Measurement.csv content
# data = {
#     #'Image': ['IMG_4136.jpg', 'IMG_4137.jpg', 'IMG_4138.jpg'],
#     'Gauge': [27.6, 25.92, 24.7, 23.88, 23.18, 22.5, 21.84, 21.16, 20.28, 19.28, 18.56, 18, 17.44, 17, 16.42, 15.9, 15.46, 14.96, 14.44, 13.9, 13.52, 13.02, 12.48, 11.96, 11.2, 10.52, 10.04, 9.52, 9.02, 8.48, 8, 7.48, 7.04, 6.48, 6, 5.52, 5, 4.52, 4.02, 3.5, 3, 2.5, 2.02, 1.46, 1, 0.72],
#     'Sensor 1': [35.7, 75.1, 83.1, 86.3, 86.9, 86.4, 85.6, 83.7, 82, 79.2, 77.3, 75.8, 74.3, 73.4, 72.1, 70.8, 69.4, 68.5, 67, 65.8, 64.6, 63.5, 62.4, 60.8, 58.9, 57.3, 56.1, 55, 53.8, 52.3, 51, 49.9, 48.6, 47.4, 46.2, 45, 43.7, 42.3, 41.2, 39.9, 38.5, 37.5, 36.1, 34.7, 33.4, 32.7],
#     'Sensor 2': [34.2, 67.8, 75, 78.6, 79.5, 79.8, 79.5, 78.4, 77, 74.6, 72.9, 71.5, 70.2, 69.3, 68, 66.7, 65.5, 64.5, 63.2, 62, 60.9, 59.8, 58.8, 57.4, 55.4, 53.8, 52.7, 51.6, 50.5, 49, 47.9, 46.8, 45.6, 44.4, 43.3, 42.1, 40.9, 39.6, 38.4, 37.2, 36, 34.7, 33.6, 32.3, 31.1, 30.4],
#     'Sensor 3': [27.5, 33.4, 32.6, 30.9, 34.6, 38.3, 40.2, 40.7, 40.8, 41.5, 41.5, 42.4, 42.6, 42.2, 41.7, 41.1, 40.4, 39.8, 39.6, 39, 38.4, 37.8, 37.2, 36.6, 35.5, 34.7, 33.8, 33.3, 32.7, 33, 34.3, 33.9, 33.3, 33.1, 32.4, 31.7, 31, 30.2, 29.5, 28.7, 28, 27.2, 26.5, 25.7, 25, 24.6]
# }
#df = pd.DataFrame(data)


# Plotting the results
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(df['Gauge'], df['Sensor 1'], label='Sensor 1')
    plt.plot(df['Gauge'], df['Sensor 2'], label='Sensor 2')
    plt.plot(df['Gauge'], df['Sensor 3'], label='Sensor 3')
    plt.xlabel('Gauge [psi]')
    plt.ylabel('Sensor [mV]')
    plt.title('Measurements')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

### END ##################################################################