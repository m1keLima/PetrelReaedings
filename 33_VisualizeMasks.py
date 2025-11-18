import os
import cv2
import json
import numpy as np

# Font colors for terminal output
DOMANDA = "\033[1;32m"  # Bold green
REGOLARE = "\033[0m" # Regular white
HIGHLIGHT = "\033[1;38;5;208m" #Bold orange
ERRORE = "\033[1;31m" #Bold red

TempFolder = 'Temp data/'
ImgDirectory = 'Images/'

def GrabPoints(Schermo, Immagine):
    # Only read points from JSON, do not allow interactive selection
    filename = f'10_Mask_{Schermo}.json'
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Mask file '{filename}' not found. Please create it first using the mask selection tool.")
    with open(filename, 'r') as f:
        points = json.load(f)
    return points

def show_quadrilaterals_on_image(image_path, points_gauge, points_petrel):
    image = cv2.imread(image_path)
    image_show = image.copy()
    overlay = image_show.copy()

    # Load colors and alpha from JSON files if available
    def load_color_json(json_file, default_rgb, default_alpha):
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                params = json.load(f)
            r = params.get("R", default_rgb[0])
            g = params.get("G", default_rgb[1])
            b = params.get("B", default_rgb[2])
            alpha = params.get("Alpha", default_alpha)
            return (r, g, b), alpha
        else:
            return default_rgb, default_alpha

    GAUGE_RGB, gauge_alpha = load_color_json("10_Mask_picker_color_Gauge.json", (255, 0, 192), 0.25)
    PETREL_RGB, petrel_alpha = load_color_json("10_Mask_picker_color_Petrel.json", (255, 0, 0), 0.25)
    GAUGE_BGR = (GAUGE_RGB[2], GAUGE_RGB[1], GAUGE_RGB[0])
    PETREL_BGR = (PETREL_RGB[2], PETREL_RGB[1], PETREL_RGB[0])


    # Draw Gauge quadrilateral (thicker, with fill)
    if points_gauge and len(points_gauge) == 4:
        pts_gauge = np.array(points_gauge, np.int32).reshape(-1, 1, 2)
        cv2.polylines(image_show, [pts_gauge], isClosed=True, color=GAUGE_BGR, thickness=20)
        cv2.fillPoly(overlay, [pts_gauge], color=GAUGE_BGR)
        # Draw text "Gauge" below the quadrilateral
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 3.5  # Larger text
        thickness = 10
        text = "Gauge"
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        pt = tuple(pts_gauge[np.argmax(pts_gauge[:,0,1])][0])
        text_pos = (pt[0] + 10, pt[1] + th + 30)
        cv2.putText(image_show, text, text_pos, font, font_scale, GAUGE_BGR, thickness, cv2.LINE_AA)
    # Draw Petrel quadrilateral (thicker, with fill)
    if points_petrel and len(points_petrel) == 4:
        pts_petrel = np.array(points_petrel, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image_show, [pts_petrel], isClosed=True, color=PETREL_BGR, thickness=20)
        cv2.fillPoly(overlay, [pts_petrel], color=PETREL_BGR)
        # Draw text "Petrel" below the quadrilateral
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 3.5  # Larger text
        thickness = 10
        text = "Petrel"
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        pt = tuple(pts_petrel[np.argmax(pts_petrel[:,0,1])][0])
        text_pos = (pt[0] + 10, pt[1] + th + 30)
        cv2.putText(image_show, text, text_pos, font, font_scale, PETREL_BGR, thickness, cv2.LINE_AA)

    # Blend the overlay with the original image for semi-transparent fill
    # Use the max alpha of the two overlays for blending
    alpha = max(gauge_alpha, petrel_alpha)
    image_show = cv2.addWeighted(overlay, alpha, image_show, 1 - alpha, 0)


    # Overlay file name at the top left
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.5
    thickness = 3
    file_text = f"File: {os.path.basename(image_path)}"
    (fw, fh), _ = cv2.getTextSize(file_text, font, font_scale, thickness)
    cv2.rectangle(image_show, (10, 10), (20 + fw, 30 + fh), (0, 0, 0), -1)
    cv2.putText(image_show, file_text, (15, 30 + fh // 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

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

def find_first_image(images_dir):
    # Get all jpg/jpeg files, then sort by filename (not by extension)
    files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG'))]
    if not files:
        return None
    files_sorted = sorted(files, key=lambda x: x.lower())
    return os.path.join(images_dir, files_sorted[0])

def RunFirstTime():
    # Ensure TempFolder exists
    if not os.path.exists(TempFolder):
        os.makedirs(TempFolder)

    # Find the first image in the Images folder
    if not os.path.exists(ImgDirectory) or not os.path.isdir(ImgDirectory):
        print(f"{ERRORE}Error: Image directory does not exist.\n  ImgDirectory value: '{ImgDirectory}'{REGOLARE}")
        exit(1)
    Immagine = find_first_image(ImgDirectory)
    if not Immagine or not os.path.isfile(Immagine):
        print(f"{ERRORE}Error: No valid image found for mask acquisition. Please ensure there is at least one image in the '{ImgDirectory}' directory.{REGOLARE}")
        exit(1)

    # Collect reference points
    points_petrel = GrabPoints("Petrel", Immagine) # Points for Petrel: Rectangle around last line
    points_gauge = GrabPoints("Gauge", Immagine)  # Points for Gauge: Rectangle around whole screen (then it will get cut) 

    # Visualize both quadrilaterals overlayed on the image and wait for user key
    show_quadrilaterals_on_image(Immagine, points_gauge, points_petrel)

if __name__ == "__main__":
    RunFirstTime()