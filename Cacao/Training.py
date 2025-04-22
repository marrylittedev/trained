import cv2
import torch
import random

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/cacao_varieties8/weights/best.pt', force_reload=True)

# Set confidence threshold
model.conf = 0.4

# Assign a unique color for each class
colors = {}
names = model.names  # {class_id: class_name}
for cls_id, cls_name in names.items():
    colors[cls_name] = [random.randint(0, 255) for _ in range(3)]

# Connect to default webcam (usually index 0)
cap = cv2.VideoCapture(1)

print("Starting cacao variety detection from webcam. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    # Run detection
    results = model(frame)
    detections = results.xyxy[0]  # tensor of detections

    # Print to terminal
    print("Detected beans in frame:")
    if len(detections) == 0:
        print("  None")

    for *box, conf, cls in detections:
        cls = int(cls)
        label = names[cls]
        confidence = float(conf)

        # Extract box coordinates
        x1, y1, x2, y2 = map(int, box)

        # Get color
        color = colors[label]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Create label text
        label_text = f"{label} ({confidence:.2f})"

        # Draw filled rectangle behind label text for visibility
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)

        # Put label text
        cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Terminal output
        print(f"  - {label} (Confidence: {confidence:.2f})")

    # Show video with annotations
    cv2.imshow("Cacao Variety Detection - Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")
