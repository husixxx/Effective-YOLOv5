import sys
import os
import torch
import cv2
import numpy as np
import yaml
from PIL import Image
from torchvision import transforms


YOLOV5_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov5'))


if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)
    
    
from models.yolo import Model

yaml_path = '../yolov5/data/qr_code.yaml' 
with open(yaml_path, 'r') as f:
    data_dict = yaml.safe_load(f)

class_names = data_dict['names']
print(f"Načítané triedy: {class_names}")


model_config = '../yolov5/models/yolov5s.yaml'


model = Model(model_config)


weights_path = '../models/500epochs/weights/best.pt' 
checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))


print("Checkpoint keys:", checkpoint.keys() if isinstance(checkpoint, dict) else "Checkpoint is a full model")


if isinstance(checkpoint['model'], torch.nn.Module):
    model = checkpoint['model'] 
else:
    model.load_state_dict(checkpoint['model'])  


model.eval()
print("Model bol úspešne načítaný manuálne.")


image_path = 'test1.jpg'  
img = Image.open(image_path)  


img = img.convert('RGB')
img = img.resize((640, 640)) 
img = np.array(img).astype(np.float32) / 255.0 
img = np.transpose(img, (2, 0, 1))  
img = torch.tensor(img).unsqueeze(0).float()
img = img.half()


with torch.no_grad():
    detections = model(img)


results = detections[0].cpu().numpy() 


confidence_threshold = 0.5
img = img.squeeze().cpu().numpy()
img = (img * 255).astype(np.uint8)
img = np.transpose(img, (1, 2, 0))


for i, detection in enumerate(results):
    detection = detection.flatten()  
    x1, y1, x2, y2, conf, cls = detection[:6] 

 
    if conf < confidence_threshold:
        continue  

    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    conf = float(conf)
    cls = int(cls)

    print(f"Detection {i}: x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf:.6f}, class={class_names[cls]}")

   
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f'{class_names[cls]} {conf:.2f}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


output_path = 'output_detected.jpg'
cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.imshow('Detected QR Codes', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
