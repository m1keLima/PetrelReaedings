import cv2
import os
import json
import subprocess
import sys

# Parameters file to save/load crop settings
CROP_PARAMS_FILE = "10_Crop_Gauge.json"
IMG_DIR = "Images"
TEMP_DIR = "Temp data"

# Find the first image in the Images folder (by filename, not extension)
files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG'))]
if not files:
    print(f"No image files found in '{IMG_DIR}'!")
    exit(1)
first_image = sorted(files, key=lambda x: x.lower())[0]
img_path = os.path.join(IMG_DIR, first_image)
print(f"Using image: {img_path}")

# Ensure the corresponding HSV Gauge image exists in Temp data, otherwise generate it
gauge_hsv_img = None
for f in os.listdir(TEMP_DIR):
    if f.lower().startswith("22_temp_gauge") and "_2_hsv" in f.lower() and f.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')):
        gauge_hsv_img = f
        break

if not gauge_hsv_img:
    # Generate the HSV image for the first image only
    print("Generating Gauge HSV image for crop adjustment...")
    # Set environment variable so 56_ExtractMeasurements.py only processes this image
    os.environ["EXTRACT_SINGLE_IMAGE"] = first_image
    subprocess.run([sys.executable, "56_ExtractMeasurements.py"])
    # Find the generated HSV image
    for f in os.listdir(TEMP_DIR):
        if f.lower().startswith("22_temp_gauge") and "_2_hsv" in f.lower() and f.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')):
            gauge_hsv_img = f
            break

if not gauge_hsv_img:
    print("Could not find or generate Gauge HSV image in Temp data.")
    exit(1)

img_path = os.path.join(TEMP_DIR, gauge_hsv_img)
print(f"Using Gauge HSV image: {img_path}")

# Load previous crop parameters if available
if os.path.exists(CROP_PARAMS_FILE):
    with open(CROP_PARAMS_FILE, "r") as f:
        crop_params = json.load(f)
    crop_top_perc = crop_params.get("crop_top_perc", 0.00)
    crop_left_perc = crop_params.get("crop_left_perc", 0.01)
    crop_right_perc = crop_params.get("crop_right_perc", 0.01)
    crop_bottom_perc = crop_params.get("crop_bottom_perc", 0.25)
else:
    crop_top_perc = 0.00
    crop_left_perc = 0.01
    crop_right_perc = 0.01
    crop_bottom_perc = 0.25

def apply_crop(img, crop_top_perc, crop_left_perc, crop_right_perc, crop_bottom_perc):
    h, w = img.shape[:2]
    left = int(w * crop_left_perc)
    right = w - int(w * crop_right_perc)
    top = int(h * crop_top_perc)
    bottom = h - int(h * crop_bottom_perc)
    # Ensure bounds
    right = min(right, w)
    bottom = min(bottom, h)
    cropped = img[top:bottom, left:right]
    return cropped

def nothing(x):
    pass

# Create window and trackbars
CROP_WINDOW_NAME = "Adjust crop for Gauge"
cv2.namedWindow(CROP_WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(CROP_WINDOW_NAME, 800, 600)
img_orig = cv2.imread(img_path)

# Trackbars use 0-1000 for fine control of percent
cv2.createTrackbar("Top %", CROP_WINDOW_NAME, int(crop_top_perc*1000), 200, nothing)
cv2.createTrackbar("Left %", CROP_WINDOW_NAME, int(crop_left_perc*1000), 200, nothing)
cv2.createTrackbar("Right %", CROP_WINDOW_NAME, int(crop_right_perc*1000), 200, nothing)
cv2.createTrackbar("Bottom %", CROP_WINDOW_NAME, int(crop_bottom_perc*1000), 400, nothing)

print("Adjust the crop using the sliders. Press 's' to save and quit, or 'q' to quit without saving.")

while True:
    t = cv2.getTrackbarPos("Top %", CROP_WINDOW_NAME) / 1000.0
    l = cv2.getTrackbarPos("Left %", CROP_WINDOW_NAME) / 1000.0
    r = cv2.getTrackbarPos("Right %", CROP_WINDOW_NAME) / 1000.0
    b = cv2.getTrackbarPos("Bottom %", CROP_WINDOW_NAME) / 1000.0

    img_cropped = apply_crop(img_orig, t, l, r, b)
    img_disp = img_cropped.copy()
    # Break the text over 4 lines
    overlay_lines = [
        f"Top:    {t:.3f}",
        f"Left:   {l:.3f}",
        f"Right:  {r:.3f}",
        f"Bottom: {b:.3f}"
    ]
    y0 = 25
    dy = 22
    for i, line in enumerate(overlay_lines):
        cv2.putText(
            img_disp,
            line,
            (10, y0 + i*dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,        # Smaller font scale
            (0,0,255),  # Red color
            1,          # Thinner line
            cv2.LINE_AA
        )
    cv2.imshow(CROP_WINDOW_NAME, img_disp)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('s'):
        # Save parameters and quit
        with open(CROP_PARAMS_FILE, "w") as f:
            json.dump({
                "crop_top_perc": t,
                "crop_left_perc": l,
                "crop_right_perc": r,
                "crop_bottom_perc": b
            }, f, indent=2)
        print(f"Saved crop parameters to {CROP_PARAMS_FILE}")
        break
    elif key == ord('q'):
        print("Quit without saving.")
        break

cv2.destroyAllWindows()