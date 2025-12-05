import cv2
import mediapipe as mp
mp_face = mp.solutions.face_detection.FaceDetection()
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(frame_rgb)
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, bw, bh = int(bbox.xmin*w), int(bbox.ymin*h), int(bbox.width*w), int(bbox.height*h)
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
    cv2.imshow("Detección de Rostro", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
