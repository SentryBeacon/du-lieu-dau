import cv2
import time
from ultralytics import YOLO
import  os, urllib.request

# Load models
vehicle_model = YOLO("yolov8n.pt")
if not os.path.exists("yolov8n-license-plate.pt"):
    urllib.request.urlretrieve(
        "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt",
        "yolov8n-license-plate.pt"
    )
plate_model = YOLO("yolov8n-license-plate.pt")

def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter == 0:
        return 0
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter)
cap = cv2.VideoCapture("test.mp4")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

VEHICLE_CLASSES = [2,3,5,7]
frame_id = 0
VEHICLE_INTERVAL = 3
last = time.time()

tracks = {}
next_id = 0

cv2.namedWindow("YOLO REALTIME", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    detections = []

    if frame_id % VEHICLE_INTERVAL == 0:
        res = vehicle_model(frame, imgsz=640, conf=0.4)[0]
        for box, cls in zip(res.boxes.xyxy, res.boxes.cls):
            if int(cls) in VEHICLE_CLASSES:
                detections.append(tuple(map(int, box)))

    new_tracks = {}
    for det in detections:
        best_iou, best_id = 0, None
        for tid, tr in tracks.items():
            val = iou(det, tr)
            if val > best_iou:
                best_iou, best_id = val, tid
        if best_iou > 0.3:
            new_tracks[best_id] = det
        else:
            new_tracks[next_id] = det
            next_id += 1

    tracks = new_tracks

    for tid, (x1,y1,x2,y2) in tracks.items():
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(frame,f"ID {tid}",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

    fps = int(1/(time.time()-last))
    last = time.time()
    cv2.putText(frame,f"FPS {fps}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("YOLO REALTIME", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
