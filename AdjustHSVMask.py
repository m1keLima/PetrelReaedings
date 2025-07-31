#############################################################################################
# Before running this file, launch the other one to cut out the Gauge and the Petrel that 
# will be then used to adjust the HSV masks.
#############################################################################################

# OVERVIEW
# This file provides an interactive tool for adjusting HSV (Hue, Saturation, Value) color masks 
# on pre-cropped images, specifically for images labeled as "Gauge" or "Petrel." The script allows 
# a user to select an image, interactively tune HSV mask parameters using OpenCV trackbars, and save 
# the selected HSV values for later use. It is intended to be run after another script that prepares 
# the relevant image crops.

# KEY COMPONENTS
# Terminal UI and File Selection:
# The script lists all files ending with _cut.jpg in the "Temp data" folder, prompts the user to 
# select one, and determines whether the file is a "Gauge" or "Petrel" image based on its filename.

# COLORFUL TERMINAL OUTPUT:
# ANSI escape codes are used to color terminal messages for better user experience and error 
# highlighting.

# OPENCV TRACKBAR INTERFACE:
# The script creates an OpenCV window with six trackbars (HMin, SMin, VMin, HMax, SMax, VMax) to let 
# the user interactively adjust the HSV mask parameters and see the effect in real time.

# HSV MASK APPLICATION:
# The selected image is converted to HSV color space, and a mask is applied based on the current 
# trackbar values. The masked result is displayed live.

# SAVING HSV VALUES:
# When the user presses 's', the current HSV min/max values are saved to a text file named according 
# to the image type (e.g., 10_HSV_Values_Gauge.txt).

# GRACEFUL TERMINAL CLEARING:
# The script attempts to clear the terminal at startup, but fails gracefully if not possible.

# EXIT MECHANISM:
# The user can quit the tool at any time by pressing 'q'.

# This file is a utility for visually tuning color segmentation parameters, which are likely used in 
# downstream image analysis tasks related to scuba diving equipment.


import sys
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

# List all *_cut.jpg files in Temp data folder
cut_folder = "Temp data"
if not os.path.exists(cut_folder):
    # Option 1: Create the directory if missing (uncomment next line if desired)
    # os.makedirs(cut_folder)
    raise FileNotFoundError(f"{ERRORE}Directory '{cut_folder}' does not exist.{REGOLARE}")
cut_files = [f for f in os.listdir(cut_folder) if f.lower().endswith('_cut.jpg')]
if not cut_files:
    raise FileNotFoundError(f"{ERRORE}No *_cut.jpg files found in '{cut_folder}' folder.{REGOLARE}")

print(f"Here are the available files\n")
for idx, fname in enumerate(cut_files):
    print(f"{idx}: {fname}")

while True:
    try:
        selection = int(input(f"{HIGHLIGHT}\nEnter the number of the file to use: {REGOLARE}"))
        if 0 <= selection < len(cut_files):
            break
        else:
            print(f"{ERRORE}Invalid selection. Try again.{REGOLARE}")
    except ValueError:
        print(f"{ERRORE}Please enter a valid number.{REGOLARE}")
    except (EOFError, KeyboardInterrupt):
        print(f"\n{ERRORE}Input cancelled by user. Exiting.{REGOLARE}")
        sys.exit(1)

selected_file = os.path.join(cut_folder, cut_files[selection])

# Determine if the file is Petrel or Gauge
if "petrel" in cut_files[selection].lower():
    selected_type = "Petrel"
elif "gauge" in cut_files[selection].lower():
    selected_type = "Gauge"
else:
    selected_type = "Unknown"

print(f"{HIGHLIGHT}\nSelected type: {REGOLARE}{selected_type}")

image = cv2.imread(selected_file)
if image is None:
    raise FileNotFoundError(f"{ERRORE}Could not load image '{selected_file}'. Please check the filename and path.{REGOLARE}")
cv2.namedWindow('image')

# Preload HSV values from file if available
hsv_defaults = {'hMin': 0, 'sMin': 0, 'vMin': 0, 'hMax': 179, 'sMax': 255, 'vMax': 255}
NomeFile = f"10_HSV_Values_{selected_type}.txt"
if os.path.exists(NomeFile):
    with open(NomeFile) as f:
        for line_num, line in enumerate(f, 1):
            if '=' in line:
                try:
                    k, v = line.strip().split('=')
                    k = k.strip()
                    v = int(v.strip())
                    # Only update if the key is in the defaults
                    if k in hsv_defaults:
                        hsv_defaults[k] = v
                except ValueError as e:
                    print(f"Warning: Skipping malformed line {line_num} in '{NomeFile}': {line.strip()} ({e})")
            elif line.strip():  # Only warn for non-empty lines
                print(f"Warning: Skipping malformed line {line_num} in '{NomeFile}': {line.strip()} (no '=')")

# Remove any existing trackbars before creating new ones (prevents OpenCV bug if rerun in same session)
trackbar_names = ['HMin', 'SMin', 'VMin', 'HMax', 'SMax', 'VMax']
window_name = 'image'
need_reset = False
for name in trackbar_names:
    with contextlib.suppress(cv2.error):
        cv2.getTrackbarPos(name, window_name)
        need_reset = True
        break
if need_reset:
    cv2.destroyWindow(window_name)
    cv2.namedWindow(window_name)

# Create trackbars for color change, using loaded defaults
cv2.createTrackbar('HMin', 'image', hsv_defaults['hMin'], 179, nothing)
cv2.createTrackbar('SMin', 'image', hsv_defaults['sMin'], 255, nothing)
cv2.createTrackbar('VMin', 'image', hsv_defaults['vMin'], 255, nothing)
cv2.createTrackbar('HMax', 'image', hsv_defaults['hMax'], 179, nothing)
cv2.createTrackbar('SMax', 'image', hsv_defaults['sMax'], 255, nothing)
cv2.createTrackbar('VMax', 'image', hsv_defaults['vMax'], 255, nothing)

print(f"{HIGHLIGHT}\nPress 's' to save the current values after adjusting the sliders or\npress 'q' to quit without saving.{REGOLARE}")

font_scale = 0.9
thickness = 1
text1 = " Press 's' to save and quit "
text2 = " Press 'q' to quit without saving "

pad = 6
extra_above = 12  # More space above text
extra_below = 6   # Less space below text

while True:
    hMin = cv2.getTrackbarPos('HMin', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    vMin = cv2.getTrackbarPos('VMin', 'image')
    hMax = cv2.getTrackbarPos('HMax', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')
    vMax = cv2.getTrackbarPos('VMax', 'image')

    # Ensure min does not exceed max for each channel; swap if needed
    if hMin > hMax:
        hMin, hMax = hMax, hMin
    if sMin > sMax:
        sMin, sMax = sMax, sMin
    if vMin > vMax:
        vMin, vMax = vMax, vMin

    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Overlay instructions on the image with white background rectangles
    result_with_text = result.copy()
    font = cv2.FONT_HERSHEY_DUPLEX
    color = (0, 120, 225)  # Orange in BGR
    # Calculate text size for background
    (w1, h1), _ = cv2.getTextSize(text1, font, font_scale, thickness)
    (w2, h2), _ = cv2.getTextSize(text2, font, font_scale, thickness)
    # Draw white rectangles as background (more above, less below)
    cv2.rectangle(result_with_text, (8, 10), (8 + w1 + pad, 10 + h1 + pad + extra_above + extra_below), (255, 255, 255), -1)
    cv2.rectangle(result_with_text, (8, 40), (8 + w2 + pad, 40 + h2 + pad + extra_above + extra_below), (255, 255, 255), -1)

    # Draw text on top of rectangles, shifted down by extra_above
    cv2.putText(result_with_text, text1, (10, 30 + extra_above), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(result_with_text, text2, (10, 60 + extra_above), font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imshow('image', result_with_text)
    key = cv2.waitKey(10) & 0xFF  # Ensure only the last 8 bits are used for key comparison
    if key == ord('s'):
        # Save the current HSV values in the requested format and close the window
        with open(NomeFile, "w") as f:
            f.write(f"hMin = {hMin}\n")
            f.write(f"sMin = {sMin}\n")
            f.write(f"vMin = {vMin}\n")
            f.write(f"hMax = {hMax}\n")
            f.write(f"sMax = {sMax}\n")
            f.write(f"vMax = {vMax}\n")
        print(f"\nHSV values saved to {NomeFile}")
        break  # Exit the loop and close the window after saving
    elif key == ord('q'):
        print(f"\n{HIGHLIGHT}Quit without saving.{REGOLARE}")
        break  # Exit the loop and close the window without saving

cv2.destroyAllWindows()