import cv2 as cv
import numpy as np
import os

def kirmizi_alani_ayikla(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    avg_value = np.mean(hsv[:, :, 2])

    if avg_value < 90:
        lower_red1 = np.array([0, 30, 30])
        lower_red2 = np.array([160, 30, 30])
    else:
        lower_red1 = np.array([0, 60, 60])
        lower_red2 = np.array([170, 60, 60])

    upper_red1 = np.array([10, 255, 255])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    combined_mask = mask1 + mask2

    return cv.bitwise_and(image, image, mask=combined_mask)

def sekizgeni_tespit_et(masked_img, original_img):
    centers = []
    contours, _ = cv.findContours(masked_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 8000:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 8:
                x, y, w, h = cv.boundingRect(approx)
                cv.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv.putText(original_img, "Dur Tabelasi", (x + w + 20, y + 20), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                cv.putText(original_img, f"Nokta: {len(approx)}", (x + w + 20, y + 45), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                centers.append([x + w // 2, y + h // 2])
    return centers

def dur_tabelasini_bul(image, positions_list):
    img_copy = image.copy()
    red_filtered = kirmizi_alani_ayikla(image)

    blurred = cv.GaussianBlur(red_filtered, (7, 7), 2)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

    kernel = np.ones((7, 7), np.uint8)
    edges = cv.Canny(gray, 30, 40)
    dilated = cv.dilate(edges, kernel, iterations=1)

    new_positions = sekizgeni_tespit_et(dilated, img_copy)
    positions_list.extend(new_positions)

    img_copy = cv.resize(img_copy, (960, 720))
    return img_copy

def dur_tabelasi_tespit():
    folder_path = "./stop_sign_dataset"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or -1))

    all_positions = []
    processed_images = []

    for index, filename in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, filename)
        image = cv.imread(image_path)
        if image is None:
            print(f"Resim okunamadi: {image_path}")
            continue

        result = dur_tabelasini_bul(image, all_positions)
        processed_images.append(result)

        output_path = os.path.join(output_dir, f"output_{index}.jpg")
        cv.imwrite(output_path, result)

    for i, center in enumerate(all_positions):
        print(f"{i + 1}. Dur tabelasi koordinatlari: {center}")

    key = cv.waitKey(0) & 0xFF
    if key == ord('q') or key == 27:
        cv.destroyAllWindows()

if __name__ == "__main__":
    dur_tabelasi_tespit()
