import torch
import cv2
import numpy as np
from yolov5 import YOLOv5


weights_path = '../models/500epochs/weights/best.pt'


model = YOLOv5(weights_path, torch.device('cpu'))




model.conf = 0.5  # Confidence threshold
model.iou = 0.45  

image_path = 'test1.jpg'
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Obrázok na ceste '{image_path}' sa nenašiel.")


results = model.predict(img)


detections = results.pred[0]  


for *xyxy, conf, cls in detections:
    x1, y1, x2, y2 = map(int, xyxy)
    conf = float(conf)
    cls = int(cls)

    
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f'QR {conf:.2f}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"Detection: x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf:.2f}, class={cls}")


output_path = 'output_detected.jpg'
cv2.imwrite(output_path, img)
cv2.imshow('Detected QR Codes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
