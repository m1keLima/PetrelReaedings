import cv2
import json
import os
import numpy as np

TempFolder = 'Temp data/'

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def ClickWithMouse(mouse_pos, image, original_image):
    x, y = mouse_pos
    crosshair_size = 120
    cv2.line(image, (x - crosshair_size, y), (x + crosshair_size, y), (0, 0, 255), 7)
    cv2.line(image, (x, y - crosshair_size), (x, y + crosshair_size), (0, 0, 255), 7)
    mag_size = 200
    mag_factor = 5
    half = mag_size // 2
    h, w = image.shape[:2]
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)
    roi = original_image[y1:y2, x1:x2]
    if roi.size > 0:
        ZoomTopLeft(roi, mag_size, mag_factor, image)

def ZoomTopLeft(roi, mag_size, mag_factor, image):
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    contrast_roi = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    zoomed = cv2.resize(contrast_roi, (mag_size * mag_factor, mag_size * mag_factor), interpolation=cv2.INTER_NEAREST)
    zh, zw = zoomed.shape[:2]
    center_x, center_y = zw // 2, zh // 2
    crosshair_length = 300
    crosshair_color = (0, 0, 255)
    thickness = 10
    cv2.line(zoomed, (center_x - crosshair_length, center_y), (center_x + crosshair_length, center_y), crosshair_color, thickness)
    cv2.line(zoomed, (center_x, center_y - crosshair_length), (center_x, center_y + crosshair_length), crosshair_color, thickness)
    cv2.circle(zoomed, (center_x, center_y), 30, crosshair_color, -1)
    border_color = (50, 0, 0)
    border_thickness = 30
    cv2.rectangle(zoomed, (0, 0), (zw-1, zh-1), border_color, border_thickness)
    img_h, img_w = image.shape[:2]
    overlay_x, overlay_y = 10, 10
    if overlay_y + zh < img_h and overlay_x + zw < img_w:
        image[overlay_y:overlay_y+zh, overlay_x:overlay_x+zw] = zoomed

def GrabPoints(Schermo, Immagine):
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
                redraw(image, points, mouse_pos, original_image)
                print(f"\n4 points selected for {Schermo}.")
                print(f"Mask points ({Schermo}): {points}")
                print("Press any key.")
                cv2.setMouseCallback("Select 4 Points", lambda *a: None)
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
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 2
        color = (255, 255, 255)
        thickness = 4
        text = f"Select 4 points for {Schermo}"
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        img_h, img_w = image.shape[:2]
        x = img_w - w - 40
        y = 40
        cv2.rectangle(image, (x, y), (x + w + 40, y + h + 30), (0, 0, 0), -1)
        cv2.putText(image, text, (x + 20, y + h + 15), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.imshow("Select 4 Points", image)

    cv2.namedWindow("Select 4 Points")
    cv2.setMouseCallback("Select 4 Points", select_point)
    redraw(image, points, mouse_pos, original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    input_points = points
    if not os.path.exists(TempFolder):
        os.makedirs(TempFolder)
    points_list = [list(pt) for pt in points]
    filename = f'10_Mask_{Schermo}.json'
    with open(filename, 'w') as f:
        json.dump(points_list, f)
    print(f"Saved {Schermo} mask points to {filename}")
    return points

if __name__ == "__main__":
    # Find the first image (by filename, not extension) in the "Images" folder
    images_dir = "Images"
    image_path = None
    if os.path.isdir(images_dir):
        files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG'))]
        if files:
            files_sorted = sorted(files, key=lambda x: x.lower())
            image_path = os.path.join(images_dir, files_sorted[0])
    if not image_path or not os.path.isfile(image_path):
        print(f"No JPG/JPEG image file found in the '{images_dir}' folder!")
        exit(1)
    print(f"Using image: {image_path}")

    # Show the file name on the image during point selection
    def GrabPointsWithFileName(Schermo, Immagine):
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
                    redraw(image, points, mouse_pos, original_image)
                    print(f"\n4 points selected for {Schermo}.")
                    print(f"Mask points ({Schermo}): {points}")
                    print("Press any key.")
                    cv2.setMouseCallback("Select 4 Points", lambda *a: None)
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
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 2
            color = (255, 255, 255)
            thickness = 4
            text = f"Select 4 points for {Schermo}"
            (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            img_h, img_w = image.shape[:2]
            x = img_w - w - 40
            y = 40
            cv2.rectangle(image, (x, y), (x + w + 40, y + h + 30), (0, 0, 0), -1)
            cv2.putText(image, text, (x + 20, y + h + 15), font, font_scale, color, thickness, cv2.LINE_AA)
            # Show file name at top left
            file_text = f"File: {os.path.basename(Immagine)}"
            font_scale_file = 1.2
            thickness_file = 2
            (fw, fh), _ = cv2.getTextSize(file_text, font, font_scale_file, thickness_file)
            cv2.rectangle(image, (10, 10), (20 + fw, 30 + fh), (0, 0, 0), -1)
            cv2.putText(image, file_text, (15, 30 + fh // 2), font, font_scale_file, (255, 255, 255), thickness_file, cv2.LINE_AA)
            cv2.imshow("Select 4 Points", image)

        cv2.namedWindow("Select 4 Points")
        cv2.setMouseCallback("Select 4 Points", select_point)
        redraw(image, points, mouse_pos, original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        input_points = points
        if not os.path.exists(TempFolder):
            os.makedirs(TempFolder)
        points_list = [list(pt) for pt in points]
        filename = f'10_Mask_{Schermo}.json'
        with open(filename, 'w') as f:
            json.dump(points_list, f)
        print(f"Saved {Schermo} mask points to {filename}")
        return points

    print("Select Petrel region:")
    GrabPointsWithFileName("Petrel", image_path)
    print("Select Gauge region:")
    GrabPointsWithFileName("Gauge", image_path)
    print("Done. Coordinates saved to 10_Mask_Petrel.json and 10_Mask_Gauge.json.")

