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
# List all *_cut.jpg files in Temp data folder, sorted alphabetically for consistent user experience
cut_files = sorted([f for f in os.listdir(cut_folder) if f.lower().endswith('_cut.jpg')])
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

# Print the file that will actually be loaded
print(f"{HIGHLIGHT}Selected file index: {selection} -> {cut_files[selection]}{REGOLARE}")
print(f"{HIGHLIGHT}Full path: {selected_file}{REGOLARE}")

# Determine if the file is Petrel or Gauge
if "petrel" in cut_files[selection].lower():
    selected_type = "Petrel"
    NomeFile = "10_HSV_Values_Petrel.json"
elif "gauge" in cut_files[selection].lower():
    selected_type = "Gauge"
    NomeFile = "10_HSV_Values_Gauge.json"
else:
    selected_type = "Unknown"
    NomeFile = "10_HSV_Values_Unknown.json"

print(f"{HIGHLIGHT}\nSelected type: {REGOLARE}{selected_type}")

image = cv2.imread(selected_file)
if image is None:
    raise FileNotFoundError(f"{ERRORE}Could not load image '{selected_file}'. Please check the filename and path.{REGOLARE}")

window_name1 = 'Parameter selection 1'
window_name2 = 'Parameter selection 2'
param1_w = 250
gap = 10

# Create and move window 2 first, then window 1, with a delay in between
cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name2, param1_w, 0)
cv2.moveWindow(window_name2, 0, 0)
cv2.waitKey(500)  # Let the OS register window 2

cv2.namedWindow(window_name1, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name1, param1_w, 0)
cv2.moveWindow(window_name1, param1_w + gap, 0)
cv2.waitKey(100)

font_scale = 0.5  # Make text smaller
thickness = 1     # Make text thinner
line_height = 18  # Make instruction lines more compact

# Preload all mask values from file if available
all_defaults = {
    'hMin': 0, 'sMin': 0, 'vMin': 0, 'hMax': 179, 'sMax': 255, 'vMax': 255,
    'LAB_LMin': 0, 'LAB_LMax': 255, 'LAB_AMin': 0, 'LAB_AMax': 255, 'LAB_BMin': 0, 'LAB_BMax': 255,
    'RGB_RMin': 0, 'RGB_RMax': 255, 'RGB_GMin': 0, 'RGB_GMax': 255, 'RGB_BMin': 0, 'RGB_BMax': 255,
    'YMin': 0, 'YMax': 255, 'CrMin': 0, 'CrMax': 255, 'CbMin': 0, 'CbMax': 255,
    'S_Only_Min': 0, 'S_Only_Max': 255, 'V_Only_Min': 0, 'V_Only_Max': 255,
    'Adaptive_Block': 21, 'Adaptive_C': 10,
    'Morph_Kernel': 3, 'Morph_Iter': 1
}

import json
if os.path.exists(NomeFile):
    with open(NomeFile, "r") as f:
        try:
            loaded = json.load(f)
            for k, v in loaded.items():
                if k in all_defaults:
                    all_defaults[k] = v
        except Exception as e:
            print(f"Warning: Could not load JSON from {NomeFile}: {e}")

# Remove any existing trackbars before creating new ones (prevents OpenCV bug if rerun in same session)
trackbar_names = [
    'hMin', 'hMax', 'sMin', 'sMax', 'vMin', 'vMax',
    'LAB_LMin', 'LAB_LMax', 'LAB_AMin', 'LAB_AMax', 'LAB_BMin', 'LAB_BMax',
    'RGB_RMin', 'RGB_RMax', 'RGB_GMin', 'RGB_GMax', 'RGB_BMin', 'RGB_BMax',
    'YMin', 'YMax', 'CrMin', 'CrMax', 'CbMin', 'CbMax',
    'S_Only_Min', 'S_Only_Max', 'V_Only_Min', 'V_Only_Max',
    'Adaptive_Block', 'Adaptive_C',
    'Morph_Kernel', 'Morph_Iter'
]
window_name1 = 'Parameter selection 1'
window_name2 = 'Parameter selection 2'
need_reset1 = False
need_reset2 = False
# Split sliders into two groups
sliders1 = [
    ('hMin', 179), ('hMax', 179),
    ('sMin', 255), ('sMax', 255),
    ('vMin', 255), ('vMax', 255),
    ('LAB_LMin', 255), ('LAB_LMax', 255),
    ('LAB_AMin', 255), ('LAB_AMax', 255),
    ('LAB_BMin', 255), ('LAB_BMax', 255),
    ('RGB_RMin', 255), ('RGB_RMax', 255),
    ('RGB_GMin', 255), ('RGB_GMax', 255),
]
sliders2 = [
    ('RGB_BMin', 255), ('RGB_BMax', 255),
    ('YMin', 255), ('YMax', 255),
    ('CrMin', 255), ('CrMax', 255),
    ('CbMin', 255), ('CbMax', 255),
    ('S_Only_Min', 255), ('S_Only_Max', 255),
    ('V_Only_Min', 255), ('V_Only_Max', 255),
    ('Adaptive_Block', 51), ('Adaptive_C', 50),
    ('Morph_Kernel', 15), ('Morph_Iter', 10)
]
for name, maxval in sliders1:
    cv2.createTrackbar(name, window_name1, all_defaults.get(name, 0), maxval, nothing)
for name, maxval in sliders2:
    cv2.createTrackbar(name, window_name2, all_defaults.get(name, 0), maxval, nothing)

print(f"{HIGHLIGHT}\nPress 's' to save the current values after adjusting the sliders or\npress 'q' to quit without saving.{REGOLARE}")

font_scale = 0.9
thickness = 1
text1 = " Press 's' to save and quit "
text2 = " Press 'q' to quit without saving "

pad = 6
extra_above = 12  # More space above text
extra_below = 6   # Less space below text

while True:
    # --- HSV sliders ---
    hMin = cv2.getTrackbarPos('hMin', window_name1)
    hMax = cv2.getTrackbarPos('hMax', window_name1)
    sMin = cv2.getTrackbarPos('sMin', window_name1)
    sMax = cv2.getTrackbarPos('sMax', window_name1)
    vMin = cv2.getTrackbarPos('vMin', window_name1)
    vMax = cv2.getTrackbarPos('vMax', window_name1)

    # --- LAB sliders ---
    lab_LMin = cv2.getTrackbarPos('LAB_LMin', window_name1)
    lab_LMax = cv2.getTrackbarPos('LAB_LMax', window_name1)
    lab_AMin = cv2.getTrackbarPos('LAB_AMin', window_name1)
    lab_AMax = cv2.getTrackbarPos('LAB_AMax', window_name1)
    lab_BMin = cv2.getTrackbarPos('LAB_BMin', window_name1)
    lab_BMax = cv2.getTrackbarPos('LAB_BMax', window_name1)

    # --- RGB sliders ---
    rgb_RMin = cv2.getTrackbarPos('RGB_RMin', window_name1)
    rgb_RMax = cv2.getTrackbarPos('RGB_RMax', window_name1)
    rgb_GMin = cv2.getTrackbarPos('RGB_GMin', window_name1)
    rgb_GMax = cv2.getTrackbarPos('RGB_GMax', window_name1)
    rgb_BMin = cv2.getTrackbarPos('RGB_BMin', window_name2)
    rgb_BMax = cv2.getTrackbarPos('RGB_BMax', window_name2)

    # --- YCrCb sliders ---
    yMin = cv2.getTrackbarPos('YMin', window_name2)
    yMax = cv2.getTrackbarPos('YMax', window_name2)
    crMin = cv2.getTrackbarPos('CrMin', window_name2)
    crMax = cv2.getTrackbarPos('CrMax', window_name2)
    cbMin = cv2.getTrackbarPos('CbMin', window_name2)
    cbMax = cv2.getTrackbarPos('CbMax', window_name2)

    # --- S/V only sliders ---
    s_only_min = cv2.getTrackbarPos('S_Only_Min', window_name2)
    s_only_max = cv2.getTrackbarPos('S_Only_Max', window_name2)
    v_only_min = cv2.getTrackbarPos('V_Only_Min', window_name2)
    v_only_max = cv2.getTrackbarPos('V_Only_Max', window_name2)

    # --- Adaptive threshold sliders ---
    adaptive_block = cv2.getTrackbarPos('Adaptive_Block', window_name2)
    if adaptive_block % 2 == 0:
        adaptive_block += 1
    if adaptive_block < 3:
        adaptive_block = 3
    adaptive_C = cv2.getTrackbarPos('Adaptive_C', window_name2)

    # --- Morphological opening sliders ---
    morph_kernel = cv2.getTrackbarPos('Morph_Kernel', window_name2)
    morph_iter = cv2.getTrackbarPos('Morph_Iter', window_name2)
    if morph_kernel < 1:
        morph_kernel = 1
    if morph_iter < 0:
        morph_iter = 0

    # Ensure min does not exceed max for each channel; swap if needed
    if hMin > hMax:
        hMin, hMax = hMax, hMin
    if sMin > sMax:
        sMin, sMax = sMax, s_min
    if vMin > vMax:
        vMin, vMax = vMax, vMin
    if lab_LMin > lab_LMax:
        lab_LMin, lab_LMax = lab_LMax, lab_LMin
    if lab_AMin > lab_AMax:
        lab_AMin, lab_AMax = lab_AMax, lab_AMin
    if lab_BMin > lab_BMax:
        lab_BMin, lab_BMax = lab_BMax, lab_BMin
    if rgb_RMin > rgb_RMax:
        rgb_RMin, rgb_RMax = rgb_RMax, rgb_RMin
    if rgb_GMin > rgb_GMax:
        rgb_GMin, rgb_GMax = rgb_GMax, rgb_GMin
    if rgb_BMin > rgb_BMax:
        rgb_BMin, rgb_BMax = rgb_BMax, rgb_BMin
    if yMin > yMax:
        yMin, yMax = yMax, yMin
    if crMin > crMax:
        crMin, crMax = crMax, crMin
    if cbMin > cbMax:
        cbMin, cbMax = cbMax, cbMin
    if s_only_min > s_only_max:
        s_only_min, s_only_max = s_only_max, s_only_min
    if v_only_min > v_only_max:
        v_only_min, v_only_max = v_only_max, v_only_min

    # --- HSV Mask ---
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, lower, upper)
    hsv_result = cv2.bitwise_and(image, image, mask=hsv_mask)

    # --- LAB Mask ---
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    lab_mask = (
        cv2.inRange(L, lab_LMin, lab_LMax)
        & cv2.inRange(A, lab_AMin, lab_AMax)
        & cv2.inRange(B, lab_BMin, lab_BMax)
    )
    lab_result = cv2.bitwise_and(image, image, mask=lab_mask)

    # --- RGB Mask ---
    rgb_mask = cv2.inRange(
        image,
        (rgb_BMin, rgb_GMin, rgb_RMin),
        (rgb_BMax, rgb_GMax, rgb_RMax)
    )
    rgb_result = cv2.bitwise_and(image, image, mask=rgb_mask)

    # --- YCrCb Mask ---
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    ycrcb_mask = (
        cv2.inRange(Y, yMin, yMax)
        & cv2.inRange(Cr, crMin, crMax)
        & cv2.inRange(Cb, cbMin, cbMax)
    )
    ycrcb_result = cv2.bitwise_and(image, image, mask=ycrcb_mask)

    # --- S/V only Mask ---
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    sv_mask = cv2.inRange(S, s_only_min, s_only_max) & cv2.inRange(V, v_only_min, v_only_max)
    sv_result = cv2.bitwise_and(image, image, mask=sv_mask)

    # --- Adaptive Thresholding (for black) ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_mask_raw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, adaptive_block, adaptive_C
    )
    adaptive_mask = cv2.bitwise_not(adaptive_mask_raw)  # Invert so white=background, black=foreground

    adaptive_result = cv2.bitwise_and(image, image, mask=adaptive_mask)

    # Arrange mask windows vertically on the right, sliders/instructions on the left
    # If Petrel, invert all masks for display
    invert_all = selected_type.lower() == "petrel"

    # Apply morphological opening ONLY to Adaptive mask
    def clean_mask(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iter)

    # Only apply to Adaptive
    adaptive_mask = clean_mask(adaptive_mask)
    # All other masks remain unchanged

    mask_windows = [
        ('HSV', cv2.bitwise_not(hsv_mask) if invert_all else hsv_mask),
        ('LAB', cv2.bitwise_not(lab_mask) if invert_all else lab_mask),
        ('RGB', cv2.bitwise_not(rgb_mask) if invert_all else rgb_mask),
        ('YCrCb', cv2.bitwise_not(ycrcb_mask) if invert_all else ycrcb_mask),
        ('S/V only', cv2.bitwise_not(sv_mask) if invert_all else sv_mask),
        ('Adaptive', cv2.bitwise_not(adaptive_mask) if invert_all else adaptive_mask)
    ]


    # Dynamically resize mask windows to fit vertically on the screen, preserving aspect ratio
    screen_h = 800  # Set this to your screen height
    n_masks = len(mask_windows)
    mask_window_spacing = 30  # Space between mask windows (pixels)
    total_spacing = mask_window_spacing * (n_masks - 1)
    available_h = screen_h - total_spacing

    # Compute the aspect ratio for each mask and the max width needed
    orig_sizes = [mask_img.shape for _, mask_img in mask_windows]
    orig_heights = [s[0] for s in orig_sizes]
    orig_widths = [s[1] for s in orig_sizes]
    aspect_ratios = [w / h for w, h in zip(orig_widths, orig_heights)]

    # Compute the height for each mask so that all fit and aspect ratio is preserved
    # Distribute available_h proportionally to original heights
    total_orig_h = sum(orig_heights)
    mask_heights = [max(1, int(available_h * (h / total_orig_h))) for h in orig_heights]
    mask_widths = [max(1, int(h * ar)) for h, ar in zip(mask_heights, aspect_ratios)]

    left_x = 0
    left_y = 0
    mask_x = 520  # leave space for sliders/instructions
    mask_y = 0

    for i, (win_name, mask_img) in enumerate(mask_windows):
        # For Adaptive, show the mask as-is (already white=foreground)
        resized_mask = cv2.resize(mask_img, (mask_widths[i], mask_heights[i]), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(win_name, resized_mask)
        cv2.moveWindow(win_name, mask_x, mask_y)
        mask_y += mask_heights[i] + mask_window_spacing

    # Show the instructions in a separate window to the right of the mask windows
    instr_lines = [
        " 's': save and quit ",
        " 'q': quit without saving ",
        " After saving, select best mask:",
        " 0=HSV 1=LAB 2=RGB 3=YCrCb",
        " 4=SV 5=Adaptive "
    ]
    font = cv2.FONT_HERSHEY_DUPLEX
    color = (0, 120, 225)  # Orange in BGR
    # font_scale, thickness, and line_height are now set above for smaller text
    pad = 6
    extra_above = 12
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in instr_lines]
    # Use a fixed width for the instruction window to avoid text cutoff
    instr_width = max(text_sizes) + 30  # 30px padding
       # Increase space between lines
    line_spacing = 16  # Additional pixels between lines
    instr_height = line_height * len(instr_lines) + (len(instr_lines)-1)*line_spacing + 20
    instr_img = np.ones((instr_height, instr_width, 3), dtype=np.uint8) * 255
    for i, line in enumerate(instr_lines):
        y = 30 + i * (line_height + line_spacing)
        cv2.putText(instr_img, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Instructions', instr_img)
    # Place instructions to the right of all mask windows
    mask_windows_right = mask_x + max(mask_widths) + 10  # 10px gap after mask windows
    cv2.moveWindow('Instructions', mask_windows_right, 0)



    key = cv2.waitKey(10) & 0xFF  # Ensure only the last 8 bits are used for key comparison
    if key == ord('s'):
        # Ask which mask is best before saving, using numbers for selection
        valid_masks = ["HSV", "LAB", "RGB", "YCrCb", "SV", "Adaptive"]
        print("\nWhich mask is best? Choose the number:")
        for idx, name in enumerate(valid_masks):
            print(f"{idx}: {name}")
        best_mask = ""
        while True:
            try:
                best_mask_idx = int(input("Best mask number: ").strip())
                if 0 <= best_mask_idx < len(valid_masks):
                    best_mask = valid_masks[best_mask_idx]
                    break
                else:
                    print(f"{ERRORE}Invalid number. Please select a number from 0 to {len(valid_masks)-1}.{REGOLARE}")
            except ValueError:
                print(f"{ERRORE}Please enter a valid number.{REGOLARE}")

        # Save all current slider values as JSON and close the window
        to_save = {
            "hMin": hMin,
            "hMax": hMax,
            "sMin": sMin,
            "sMax": sMax,
            "vMin": vMin,
            "vMax": vMax,
            "LAB_LMin": lab_LMin,
            "LAB_LMax": lab_LMax,
            "LAB_AMin": lab_AMin,
            "LAB_AMax": lab_AMax,
            "LAB_BMin": lab_BMin,
            "LAB_BMax": lab_BMax,
            "RGB_RMin": rgb_RMin,
            "RGB_RMax": rgb_RMax,
            "RGB_GMin": rgb_GMin,
            "RGB_GMax": rgb_GMax,
            "RGB_BMin": rgb_BMin,
            "RGB_BMax": rgb_BMax,
            "YMin": yMin,
            "YMax": yMax,
            "CrMin": crMin,
            "CrMax": crMax,
            "CbMin": cbMin,
            "CbMax": cbMax,
            "S_Only_Min": s_only_min,
            "S_Only_Max": s_only_max,
            "V_Only_Min": v_only_min,
            "V_Only_Max": v_only_max,
            "Adaptive_Block": adaptive_block,
            "Adaptive_C": adaptive_C,
            "Morph_Kernel": morph_kernel,
            "Morph_Iter": morph_iter,
            "BestMask": best_mask
        }
        with open(NomeFile, "w") as f:
            json.dump(to_save, f, indent=2)
        print(f"\nAll mask values saved to {NomeFile} (BestMask = {best_mask})")
        break  # Exit the loop and close the window after saving
    elif key == ord('q'):
        print(f"\n{HIGHLIGHT}Quit without saving.{REGOLARE}")
        break  # Exit the loop and close the window without saving

cv2.destroyAllWindows()