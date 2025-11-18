# This file provides an interactive tool for visually selecting and saving color and transparency 
# (alpha) parameters for a polygon overlay on an image. It uses OpenCV to display an image, draw 
# a polygon, and allow the user to adjust the polygon's color and transparency in real time using 
# sliders. When the user is satisfied, they can press any key to save the selected parameters 
# to a text file. The script is intended for use cases where precise color selection for overlays 
# is needed, such as in image annotation or UI prototyping.

import cv2
import numpy as np
import os

# Font colors
DOMANDA = "\033[1;32m"  # Bold green
REGOLARE = "\033[0m" # Regular white
HIGHLIGHT = "\033[1;38;5;208m" #Bold orange
ERRORE = "\033[1;31m" #Bold red

def nothing(x):
    pass

# Clear terminal
import contextlib
def clear_terminal():
    """Attempt to clear the terminal, but fail gracefully if not possible."""
    with contextlib.suppress(Exception):
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
clear_terminal()  # Clear terminal (optional, fails gracefully)

# Load image
# Load image: open first a Petrel image, then a Gauge image, and save both JSONs
import glob
import json

def adjust_and_save_color(img_path, param_json):
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    print(f"Loaded image: {img_path}")

    # Define polygon points (example: corners of the image, or replace with your own)
    h, w = image.shape[:2]
    polygon = np.array([
        [int(0.2*w), int(0.2*h)],
        [int(0.8*w), int(0.2*h)],
        [int(0.8*w), int(0.8*h)],
        [int(0.2*w), int(0.8*h)]
    ], np.int32).reshape((-1, 1, 2))

    print(f"{DOMANDA}Adjust the sliders. Press any key in the image window to quit and save parameters.{REGOLARE}")

    cv2.namedWindow('Polygon Overlay')

    # Load initial values from JSON if available
    if os.path.exists(param_json):
        with open(param_json, "r") as f:
            params = json.load(f)
        r_init = params.get("R", 0)
        g_init = params.get("G", 0)
        b_init = params.get("B", 255)
        alpha_init = int(params.get("Alpha", 0.5) * 100)
    else:
        r_init = 0
        g_init = 0
        b_init = 255
        alpha_init = 50

    # Create sliders for R, G, B, alpha
    cv2.createTrackbar('R', 'Polygon Overlay', r_init, 255, nothing)
    cv2.createTrackbar('G', 'Polygon Overlay', g_init, 255, nothing)
    cv2.createTrackbar('B', 'Polygon Overlay', b_init, 255, nothing)
    cv2.createTrackbar('Alpha', 'Polygon Overlay', alpha_init, 100, nothing)  # 0-100%

    while True:
        # Get slider values
        r = cv2.getTrackbarPos('R', 'Polygon Overlay')
        g = cv2.getTrackbarPos('G', 'Polygon Overlay')
        b = cv2.getTrackbarPos('B', 'Polygon Overlay')
        alpha = cv2.getTrackbarPos('Alpha', 'Polygon Overlay') / 100.0

        overlay = image.copy()
        cv2.fillPoly(overlay, [polygon], color=(b, g, r))
        # Blend overlay with original image
        blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Draw polygon border with the same color as the fill
        cv2.polylines(blended, [polygon], isClosed=True, color=(b, g, r), thickness=3)

        # Draw a text box to explain how to quit and save
        text = "Press any key to quit and save parameters"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.2
        thickness = 2
        text_color = (0, 0, 0)
        bg_color = (255, 255, 255)
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        pad_x, pad_y = 20, 15
        x, y = 20, 20
        cv2.rectangle(blended, (x, y), (x + tw + pad_x, y + th + pad_y), bg_color, -1)
        cv2.putText(blended, text, (x + pad_x // 2, y + th + pad_y // 2), font, font_scale, text_color, thickness, cv2.LINE_AA)

        cv2.imshow('Polygon Overlay', blended)
        key = cv2.waitKey(10) & 0xFF
        if key != 255:  # Any key pressed (not just q or esc)
            break

    cv2.destroyAllWindows()

    # Save the last slider values to a json file
    params = {
        "R": r,
        "G": g,
        "B": b,
        "Alpha": round(alpha, 2)
    }
    with open(param_json, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\n{HIGHLIGHT}Parameters saved to {param_json}{REGOLARE}\n")

# --- Main logic: open Petrel, then Gauge ---

img_files_petrel = sorted(glob.glob("Temp data/*Petrel*.jpg"), key=lambda x: x.lower())
img_files_gauge = sorted(glob.glob("Temp data/*Gauge*.jpg"), key=lambda x: x.lower())

if not img_files_petrel:
    raise FileNotFoundError("No file with 'Petrel' found in Temp data folder.")
if not img_files_gauge:
    raise FileNotFoundError("No file with 'Gauge' found in Temp data folder.")

# Petrel
adjust_and_save_color(img_files_petrel[0], "10_Mask_picker_color_Petrel.json")
# Gauge
adjust_and_save_color(img_files_gauge[0], "10_Mask_picker_color_Gauge.json")


# Define polygon points (example: corners of the image, or replace with your own)
# For demonstration, let's use a rectangle inside the image
h, w = image.shape[:2]
polygon = np.array([
    [int(0.2*w), int(0.2*h)],
    [int(0.8*w), int(0.2*h)],
    [int(0.8*w), int(0.8*h)],
    [int(0.2*w), int(0.8*h)]
], np.int32).reshape((-1, 1, 2))

def nothing(x):
    pass

print(f"{DOMANDA}Adjust the sliders. Press any key in the image window to quit and save parameters.{REGOLARE}")

cv2.namedWindow('Polygon Overlay')

# Load initial values from JSON if available
if os.path.exists(param_json):
    with open(param_json, "r") as f:
        params = json.load(f)
    r_init = params.get("R", 0)
    g_init = params.get("G", 0)
    b_init = params.get("B", 255)
    alpha_init = int(params.get("Alpha", 0.5) * 100)
else:
    r_init = 0
    g_init = 0
    b_init = 255
    alpha_init = 50

# Create sliders for R, G, B, alpha
cv2.createTrackbar('R', 'Polygon Overlay', r_init, 255, nothing)
cv2.createTrackbar('G', 'Polygon Overlay', g_init, 255, nothing)
cv2.createTrackbar('B', 'Polygon Overlay', b_init, 255, nothing)
cv2.createTrackbar('Alpha', 'Polygon Overlay', alpha_init, 100, nothing)  # 0-100%

while True:
    # Get slider values
    r = cv2.getTrackbarPos('R', 'Polygon Overlay')
    g = cv2.getTrackbarPos('G', 'Polygon Overlay')
    b = cv2.getTrackbarPos('B', 'Polygon Overlay')
    alpha = cv2.getTrackbarPos('Alpha', 'Polygon Overlay') / 100.0

    overlay = image.copy()
    cv2.fillPoly(overlay, [polygon], color=(b, g, r))
    # Blend overlay with original image
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw polygon border with the same color as the fill
    cv2.polylines(blended, [polygon], isClosed=True, color=(b, g, r), thickness=3)

    # Draw a text box to explain how to quit and save
    text = "Press any key to quit and save parameters"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.2
    thickness = 2
    text_color = (0, 0, 0)
    bg_color = (255, 255, 255)
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 20, 15
    x, y = 20, 20
    cv2.rectangle(blended, (x, y), (x + tw + pad_x, y + th + pad_y), bg_color, -1)
    cv2.putText(blended, text, (x + pad_x // 2, y + th + pad_y // 2), font, font_scale, text_color, thickness, cv2.LINE_AA)

    cv2.imshow('Polygon Overlay', blended)
    key = cv2.waitKey(10) & 0xFF
    if key != 255:  # Any key pressed (not just q or esc)
        break

cv2.destroyAllWindows()

# Save the last slider values to a json file
params = {
    "R": r,
    "G": g,
    "B": b,
    "Alpha": round(alpha, 2)
}
with open(param_json, "w") as f:
    json.dump(params, f, indent=2)
print(f"\n{HIGHLIGHT}Parameters saved to {param_json}{REGOLARE}\n")