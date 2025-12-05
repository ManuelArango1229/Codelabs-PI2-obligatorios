from ultralytics import YOLO

model_yolo = YOLO("yolov8n.pt")
results = model_yolo("imagen1.png")

# results es una lista, así que tomamos el primer elemento
r = results[0]

# Mostrar en ventana
r.show()

# Guardar la imagen con las detecciones en disco
r.save(filename="imagen1.png")

# Ver las cajas detectadas (x1, y1, x2, y2, confianza, clase)
print(r.boxes)

# Ver las coordenadas y confianza de cada detección
for box in r.boxes:
    print(f"Clase: {int(box.cls)}, Confianza: {float(box.conf):.2f}, Coordenadas: {box.xyxy.tolist()}")

# Ver nombres de las clases
print("Clases detectadas:", [model_yolo.names[int(box.cls)] for box in r.boxes])
