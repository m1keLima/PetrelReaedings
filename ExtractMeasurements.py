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

#from PIL import Image, ImageEnhance, ImageOps
import cv2, os, csv, glob



### PARAMETERS TO ADJUST WHEN YOU DO A NEW CALIBRATION ###################

# Adjust if it doesnt cut perfect only the number of Gauge
wPerc = 0.84
hPerc = 0.80

# Font colors
DOMANDA = "\033[1;32m"  # Bold green
REGOLARE = "\033[0m" # Regular white
HIGHLIGHT = "\033[1;38;5;208m" #Bold orange
ERRORE = "\033[1;31m" #Bold red

def VariabiliHSV(Schermo):  # Adjust with 0AdjustHSVMask.py
    # Returns HSV min/max values as a tuple: (hMin, sMin, vMin, hMax, sMax, vMax)
    if Schermo == "Petrel":
        hMin = 0
        sMin = 122
        vMin = 179
        hMax = 90
        sMax = 255
        vMax = 255
        return hMin, sMin, vMin, hMax, sMax, vMax
    elif Schermo == "Gauge":
        hMin = 0
        sMin = 0
        vMin = 225
        hMax = 179
        sMax = 255
        vMax = 255
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


def HSVprocess(inputImage, hMin, sMin, vMin, hMax, sMax, vMax):
    from PIL import Image, ImageEnhance, ImageOps
    import numpy as np
    import os
    import re

    # Ensure TempFolder exists
    if not os.path.exists(TempFolder):
        os.makedirs(TempFolder)

    image = cv2.imread(inputImage)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Use the passed HSV values
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # HSV transformations
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    result = cv2.medianBlur(result, 5)

    # Save image
    base = os.path.splitext(os.path.basename(inputImage))[0]
    base = re.sub(r'_1_Cut$', '', base)
    ext = os.path.splitext(inputImage)[1]
    new_filename = os.path.join(TempFolder, f"{base}_2_HSV{ext}")
    cv2.imwrite(new_filename, result)

    # Open and further process the image
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
    #OCR
    Image_Right = Image.open(NomeImmagine)
    custom_config = r'--psm 7'
    text = pytesseract.image_to_string(Image_Right, config=custom_config)
    print("Petrel extracted: ", repr(text))

    #Filter with RegEx
    pattern = r'^(1\d{2}|\d{2}\.\d)\s(1\d{2}|\d{2}\.\d)\s(1\d{2}|\d{2}\.\d)$'

    if match := re.match(pattern, text.strip()):
        vector = text.strip().split()
        print("Petrel cleaned: ", repr(vector))
    else:
        vector = []
        print(f"{ERRORE}No match{REGOLARE}")
    return vector


def OCRGauge(NomeImmagine): # OCR Gauge
    import re, subprocess, os

    # Ensure the path uses the correct TempFolder
    NomeImmagine = os.path.normpath(NomeImmagine)

    #OCR
    command = f"ssocr -d -1 -c decimal \"{NomeImmagine}\""
    try:
        result = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT).strip()
        print("Gauge extracted: ", repr(result))

        if match := re.search(r'\d{1,2}\.\d{2}', result):
            extracted_text = match.group() if match else None
            print("Gauge cleaned: ", repr(extracted_text))
        else:
            extracted_text = None
            print(f"{ERRORE}No match found.{REGOLARE}")
        return extracted_text
    except subprocess.CalledProcessError as e:
        print(f"{ERRORE}ssocr failed for {NomeImmagine} with error code {e.returncode}{REGOLARE}")
        print(f"{HIGHLIGHT}ssocr output:{REGOLARE}", e.output)
        return None


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

    # Extract measurements from Petrel
    print(f"\nFile:", repr(Immagine))
    with open('10_Mask_Petrel.json', 'r') as f: # Load points from the JSON file
        points = json.load(f)  # This will be a list of [x, y] lists
    input_points = [tuple(pt) for pt in points] # Convert points into tuples
    result_petrel = deskew_and_crop(Immagine, input_points) # Cut and deskew image
    temp_cut_petrel = os.path.join(TempFolder, PETREL_CUT_PATTERN)
    cv2.imwrite(temp_cut_petrel, result_petrel)
    hMin, sMin, vMin, hMax, sMax, vMax = VariabiliHSV("Petrel")
    result_hsv_petrel = HSVprocess(temp_cut_petrel, hMin, sMin, vMin, hMax, sMax, vMax) # HSV process

    # Visualize image if we are doing the first setup run
    if answer in ['yes', 'y']:
        image = VisualizeStep(temp_cut_petrel, "BW", wait_ms=0)

    # OCR Petrel
    temp_bw_petrel = os.path.join(TempFolder, PETREL_BW_PATTERN)
    Petrel = OCRPetrel(temp_bw_petrel) 

    # Extract measurements from Gauge
    with open('10_Mask_Gauge.json', 'r') as f: # Load points from the JSON file
        points = json.load(f)  # This will be a list of [x, y] lists
    input_points = [tuple(pt) for pt in points] # Convert points into tuples
    result_gauge = deskew_and_crop(Immagine, input_points) # Cut and deskew image
    temp_cut_gauge = os.path.join(TempFolder, GAUGE_CUT_PATTERN)
    cv2.imwrite(temp_cut_gauge, result_gauge)
    hMin, sMin, vMin, hMax, sMax, vMax = VariabiliHSV("Gauge")
    result_hsv_gauge = HSVprocess(temp_cut_gauge, hMin, sMin, vMin, hMax, sMax, vMax) # HSV process
    temp_bw_gauge = os.path.join(TempFolder, GAUGE_BW_PATTERN)
    MoreProcessGauge(temp_bw_gauge)

    # Visualize image if we are doing the first setup run
    if answer in ['yes', 'y']:
        image = VisualizeStep(temp_bw_gauge, "BW", wait_ms=0)

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
    import os
    points = []
    image = cv2.imread(Immagine)
    original_image = image.copy()
    mouse_pos = None

    def select_point(event, x, y, flags, param):
        nonlocal points, image, mouse_pos
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_pos = (x, y)
            redraw(image, points, mouse_pos, original_image)
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                redraw(image, points, mouse_pos, original_image)
            if len(points) == 4:
                redraw(image, points, mouse_pos, original_image)  # Show final selection
                print(f"{HIGHLIGHT}\n4 points selected for {Schermo}.{REGOLARE}")
                print(f"Mask points ({Schermo}): {points}")
                print(f"{DOMANDA}Press any key.{REGOLARE}")
                cv2.setMouseCallback("Select 4 Points", lambda *a: None)  # Disable further mouse events
                cv2.destroyAllWindows()

    def redraw(image, points, mouse_pos, original_image):
        image = original_image.copy()
        for pt in points:
            cv2.circle(image, pt, 20, (0, 0, 255), -1)
        for i in range(1, len(points)):
            cv2.line(image, points[i - 1], points[i], (0, 0, 255), 4)
        if points and mouse_pos and len(points) < 4:
            cv2.line(image, points[-1], mouse_pos, (0, 0, 255), 4)
        if mouse_pos:
            ClickWithMouse(mouse_pos, image, original_image)
        
        # Overlay instruction text in top right, extra large font with padding and empty lines
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 5
        color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)     # Black background
        thickness = 10
        
        # Simulate empty lines by drawing three lines: empty, text, empty
        text_lines = ["", f"  Select 4 points for {Schermo}  ", ""]
        (w, h), _ = cv2.getTextSize(text_lines[1], font, font_scale, thickness)
        pad_x = 40
        pad_y = 30
        line_spacing = h + 20
        img_h, img_w = image.shape[:2]
        x = img_w - w - pad_x - 20
        y = 20
        total_height = line_spacing * len(text_lines)
        
        # Draw background rectangle with extra padding and space for empty lines
        cv2.rectangle(image, (x, y), (x + w + pad_x, y + total_height + pad_y), bg_color, -1)
        
        # Draw each line (empty lines will just add space)
        for i, line in enumerate(text_lines):
            if line.strip():
                cv2.putText(
                    image,
                    line,
                    (x + pad_x // 2, y + h + pad_y // 2 + i * line_spacing),
                    font,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
        cv2.imshow("Select 4 Points", image)

    cv2.namedWindow("Select 4 Points")
    cv2.setMouseCallback("Select 4 Points", select_point)
    redraw(image, points, mouse_pos, original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    input_points = points # Define your 4 points (replace with your coordinates)

    # Ensure TempFolder exists
    if not os.path.exists(TempFolder):
        os.makedirs(TempFolder)

    # Process and save result
    result = deskew_and_crop(Immagine, input_points)

    # Convert tuples to lists for JSON compatibility
    points_list = [list(pt) for pt in points]

    filename = f'10_Mask_{Schermo}.json'
    if Schermo == "Petrel":        
        with open(filename, 'w') as f: # Save to a JSON file
            json.dump(points_list, f)
        img_filename = os.path.join(TempFolder, PETREL_CUT_PATTERN)
        hsv_filename = os.path.join(TempFolder, PETREL_HSV_PATTERN)
        bw_filename = os.path.join(TempFolder, PETREL_BW_PATTERN)
    else:
        with open(filename, 'w') as f: # Save to a JSON file
            json.dump(points_list, f)
        img_filename = os.path.join(TempFolder, GAUGE_CUT_PATTERN)
        hsv_filename = os.path.join(TempFolder, GAUGE_HSV_PATTERN)
        bw_filename = os.path.join(TempFolder, GAUGE_BW_PATTERN)

    cv2.imwrite(img_filename, result)
    # If you want to immediately process and save HSV/BW, call your processing functions here:
    # result_hsv = HSVprocess(img_filename)  # This will save hsv_filename and bw_filename automatically
    return points


def MoreProcessGauge(NomeImmagine, threshold=100):
    #This is a processing step required for Gauge but not for Petrel.
    from PIL import Image, ImageEnhance, ImageOps

    img = Image.open(NomeImmagine)
    img = img.convert('L')  # Convert to grayscale for consistent binarization
    img = ImageOps.invert(img)  # Invert grayscale image
    img = img.point(lambda p: 255 if p > threshold else 0)
    img.save(NomeImmagine)
     
    img = Image.open(NomeImmagine)
    new_width_Gauge = int(img.width * wPerc)
    new_height_Gauge = int(img.height * hPerc)
    crop_box = (0, 0, new_width_Gauge, new_height_Gauge)
    img = img.crop(crop_box)
    img.save(NomeImmagine)


def RunFirstTime():
    import os
    global result, image, points, mouse_pos, original_image
    global PETREL_CUT_PATTERN, PETREL_BW_PATTERN, GAUGE_CUT_PATTERN, GAUGE_BW_PATTERN

    # Ensure TempFolder exists
    if not os.path.exists(TempFolder):
        os.makedirs(TempFolder)

    # Collect reference points
    points_petrel = GrabPoints("Petrel", Immagine) # Points for Petrel: Rectangle around last line
    points_gauge = GrabPoints("Gauge", Immagine)  # Points for Gauge: Rectangle around whole screen (then it will get cut) 

    def show_quadrilaterals_on_image(image_path, points_gauge, points_petrel):
        import numpy as np 

        image = cv2.imread(image_path)
        image_show = image.copy()
        overlay = image_show.copy()
        alpha = 0.25  # Transparency for fill

        # Define colors in RGB and convert to BGR for OpenCV
        GAUGE_RGB = (255, 0, 192)   # Color for Gauge
        PETREL_RGB = (255, 0, 0)  # Color for Petrel
        GAUGE_BGR = (GAUGE_RGB[2], GAUGE_RGB[1], GAUGE_RGB[0])
        PETREL_BGR = (PETREL_RGB[2], PETREL_RGB[1], PETREL_RGB[0])

        # Draw Gauge quadrilateral (thicker, with fill)
        if points_gauge and len(points_gauge) == 4:
            pts_gauge = np.array(points_gauge, np.int32).reshape(-1, 1, 2)
            cv2.polylines(image_show, [pts_gauge], isClosed=True, color=GAUGE_BGR, thickness=20)
            cv2.fillPoly(overlay, [pts_gauge], color=GAUGE_BGR)
        # Draw Petrel quadrilateral (thicker, with fill)
        if points_petrel and len(points_petrel) == 4:
            pts_petrel = np.array(points_petrel, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image_show, [pts_petrel], isClosed=True, color=PETREL_BGR, thickness=20)
            cv2.fillPoly(overlay, [pts_petrel], color=PETREL_BGR)
        # Blend the overlay with the original image for semi-transparent fill
        image_show = cv2.addWeighted(overlay, alpha, image_show, 1 - alpha, 0)


        # Overlay instruction text
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 3
        color = (255, 255, 255)
        bg_color = (0, 0, 0)
        thickness = 7
        text = "Check both masks, then press any key"
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        pad_x = 40
        pad_y = 30
        img_h, img_w = image_show.shape[:2]
        x = img_w - w - pad_x - 20
        y = 20
        cv2.rectangle(image_show, (x, y), (x + w + pad_x, y + h + pad_y), bg_color, -1)
        cv2.putText(image_show, text, (x + pad_x // 2, y + h + pad_y // 2), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.imshow("Check Quadrilaterals", image_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Visualize both quadrilaterals overlayed on the image and wait for user key
    show_quadrilaterals_on_image(Immagine, points_gauge, points_petrel)

    # Use the filename pattern variables instead of hardcoded names
    img_petrel_cut = RunHSVAndVisualize("Petrel", PETREL_CUT_PATTERN, PETREL_BW_PATTERN)
    img_gauge_cut = RunHSVAndVisualize("Gauge", GAUGE_CUT_PATTERN, GAUGE_BW_PATTERN, do_more_process=True)


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
answer = input(f"{DOMANDA}\nGet coordinates for mask? (yes/no):  {REGOLARE}").strip().lower()

if answer in ['yes', 'y']:
    PETREL_CUT_PATTERN = "11_Mask_Petrel_1_Cut.jpg"
    PETREL_HSV_PATTERN = "11_Mask_Petrel_2_HSV.jpg"
    PETREL_BW_PATTERN = "11_Mask_Petrel_3_BW.jpg"
    GAUGE_CUT_PATTERN = "12_Mask_Gauge_1_Cut.jpg"
    GAUGE_HSV_PATTERN = "12_Mask_Gauge_2_HSV.jpg"
    GAUGE_BW_PATTERN = "12_Mask_Gauge_3_BW.jpg"
else:
    PETREL_CUT_PATTERN = "21_Temp_Petrel_1_Cut.jpg"
    PETREL_HSV_PATTERN = "21_Temp_Petrel_2_HSV.jpg"
    PETREL_BW_PATTERN = "21_Temp_Petrel_3_BW.jpg"
    GAUGE_CUT_PATTERN = "22_Temp_Gauge_1_Cut.jpg"
    GAUGE_HSV_PATTERN = "22_Temp_Gauge_2_HSV.jpg"
    GAUGE_BW_PATTERN = "22_Temp_Gauge_3_BW.jpg"

if answer in ['yes', 'y']:
    # === FIRST TIME CODE ===
    print(f"{HIGHLIGHT}\n ---- FIRST RUN ---- \n{REGOLARE}")
    # Grab first image of the folder to do the masking
    files = sorted(glob.glob(f'{ImgDirectory}*.jp*g', recursive=False))
    Immagine = files[0] if files else None
    RunFirstTime()
    print(f"{HIGHLIGHT}\nMask acquisition completed!\n{REGOLARE}")
else:
    # === SUBSEQUENT RUNS CODE ===
    print(f"\n{HIGHLIGHT} ---- LOOP THROUGH IMAGES ---- {REGOLARE}\n")

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
        if image_files := sorted(
            [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg'))]
        ):
            for filename in image_files: # Loop the jpg sorted by alphatical order
                Immagine = os.path.join(ImgDirectory, filename)
                Gauge, Petrel = ProcessOneImage()
            print(f"{HIGHLIGHT}\nLoop finished!{REGOLARE}\n")
        else:
            print(f"{ERRORE}Warning: No image files found in '{input_directory}'.{REGOLARE}")
    
    
    # Production: Load the CSV file
    import pandas as pd
    df = pd.read_csv('20_Measurements.csv')

    # Demo: Create a sample DataFrame similar to Measurement.csv content
    data = {
        #'Image': ['IMG_4136.jpg', 'IMG_4137.jpg', 'IMG_4138.jpg'],
        'Gauge': [27.6, 25.92, 24.7, 23.88, 23.18, 22.5, 21.84, 21.16, 20.28, 19.28, 18.56, 18, 17.44, 17, 16.42, 15.9, 15.46, 14.96, 14.44, 13.9, 13.52, 13.02, 12.48, 11.96, 11.2, 10.52, 10.04, 9.52, 9.02, 8.48, 8, 7.48, 7.04, 6.48, 6, 5.52, 5, 4.52, 4.02, 3.5, 3, 2.5, 2.02, 1.46, 1, 0.72],
        'Sensor 1': [35.7, 75.1, 83.1, 86.3, 86.9, 86.4, 85.6, 83.7, 82, 79.2, 77.3, 75.8, 74.3, 73.4, 72.1, 70.8, 69.4, 68.5, 67, 65.8, 64.6, 63.5, 62.4, 60.8, 58.9, 57.3, 56.1, 55, 53.8, 52.3, 51, 49.9, 48.6, 47.4, 46.2, 45, 43.7, 42.3, 41.2, 39.9, 38.5, 37.5, 36.1, 34.7, 33.4, 32.7],
        'Sensor 2': [34.2, 67.8, 75, 78.6, 79.5, 79.8, 79.5, 78.4, 77, 74.6, 72.9, 71.5, 70.2, 69.3, 68, 66.7, 65.5, 64.5, 63.2, 62, 60.9, 59.8, 58.8, 57.4, 55.4, 53.8, 52.7, 51.6, 50.5, 49, 47.9, 46.8, 45.6, 44.4, 43.3, 42.1, 40.9, 39.6, 38.4, 37.2, 36, 34.7, 33.6, 32.3, 31.1, 30.4],
        'Sensor 3': [27.5, 33.4, 32.6, 30.9, 34.6, 38.3, 40.2, 40.7, 40.8, 41.5, 41.5, 42.4, 42.6, 42.2, 41.7, 41.1, 40.4, 39.8, 39.6, 39, 38.4, 37.8, 37.2, 36.6, 35.5, 34.7, 33.8, 33.3, 32.7, 33, 34.3, 33.9, 33.3, 33.1, 32.4, 31.7, 31, 30.2, 29.5, 28.7, 28, 27.2, 26.5, 25.7, 25, 24.6]
    }
    df = pd.DataFrame(data)


    # Plotting the results
    import matplotlib.pyplot as plt
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