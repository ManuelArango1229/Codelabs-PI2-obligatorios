from ultralytics import YOLO  
import time
model = YOLO("yolov8n.pt")  
image_path = "imagen1.png"
t2 = time.time()  
results = model(image_path)  
t3 = time.time()
for r in results:  
    print(r.names)
    print(r.boxes)  
results[0].save(filename="resultadoYolo.jpg")
