import cv2
import os

# Cesty k obrázkům a složce pro uložení anotací
image_folder = '/home/husix/BP_2024/data/images/train'
output_folder = '/home/husix/BP_2024/data/labels/train'
os.makedirs(output_folder, exist_ok=True)

# Procházení obrázků ve složce
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Načti obrázek
        img = cv2.imread(os.path.join(image_folder, filename))
        detector = cv2.QRCodeDetector()

        # Detekuj QR kód
        retval, points = detector.detect(img)
        if retval:
            # Výpočet středu a velikosti bounding boxu pro YOLO formát
            x_center = (points[0][0][0] + points[0][2][0]) / 2 / img.shape[1]
            y_center = (points[0][0][1] + points[0][2][1]) / 2 / img.shape[0]
            width = (points[0][2][0] - points[0][0][0]) / img.shape[1]
            height = (points[0][2][1] - points[0][0][1]) / img.shape[0]

            # Ulož anotaci ve formátu YOLO
            with open(os.path.join(output_folder, filename.replace('.jpg', '.txt')), 'w') as f:
                f.write(f"0 {x_center} {y_center} {width} {height}\n")
