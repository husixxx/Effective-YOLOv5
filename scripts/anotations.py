import cv2
import os

image_folder = '/home/husix/BP_2024/data/images/train'
output_folder = '/home/husix/BP_2024/data/labels/train'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):

        img = cv2.imread(os.path.join(image_folder, filename))
        detector = cv2.QRCodeDetector()


        retval, points = detector.detect(img)
        if retval and points is not None:

            x_min = max(0, min(points[0][0][0], points[0][1][0], points[0][2][0], points[0][3][0]))
            x_max = min(img.shape[1], max(points[0][0][0], points[0][1][0], points[0][2][0], points[0][3][0]))
            y_min = max(0, min(points[0][0][1], points[0][1][1], points[0][2][1], points[0][3][1]))
            y_max = min(img.shape[0], max(points[0][0][1], points[0][1][1], points[0][2][1], points[0][3][1]))

            # normalization
            x_center = ((x_min + x_max) / 2) / img.shape[1]
            y_center = ((y_min + y_max) / 2) / img.shape[0]
            width = (x_max - x_min) / img.shape[1]
            height = (y_max - y_min) / img.shape[0]

            if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                with open(os.path.join(output_folder, filename.replace('.jpg', '.txt')), 'w') as f:
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")
            else:
                print(f"Warning: Bounding box out of range for file {filename}")
