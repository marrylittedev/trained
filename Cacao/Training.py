import cv2
import torch
import random
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='yolov5/runs/train/cacao_varieties8/weights/best.pt',
                       force_reload=True)
model.conf = 0.4

# Assign random colors for class labels
colors = {}
names = model.names
for cls_id, cls_name in names.items():
    colors[cls_name] = [random.randint(0, 255) for _ in range(3)]

# Define HSV color ranges
color_ranges = {
    "Criollo": ((10, 50, 80), (30, 255, 255)),
    "Forastero": ((0, 0, 0), (20, 255, 120)),
    "Trinitario": ((15, 100, 100), (40, 255, 200))
}

# Create dashboard panel
def create_dashboard(info_list, width=300, height=480):
    panel = np.zeros((height, width, 3), dtype=np.uint8) + 50  # dark background
    y = 30
    cv2.putText(panel, "📊 Dashboard", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 30

    if not info_list:
        cv2.putText(panel, "No beans detected", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return panel

    for i, info in enumerate(info_list):
        text = f"{info['label']} - {info['variety']}"
        conf = f"Conf: {info['confidence']:.2f}"
        color_text = f"HSV Match: {info['variety']}"

        cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        y += 20
        cv2.putText(panel, conf, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
        y += 20
        cv2.putText(panel, color_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)
        y += 30

    return panel

# Start webcam
cap = cv2.VideoCapture(1)
print("Starting detection with dashboard. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run detection
    results = model(frame)
    detections = results.xyxy[0]
    dashboard_data = []

    for *box, conf, cls in detections:
        cls = int(cls)
        label = names[cls]
        confidence = float(conf)
        x1, y1, x2, y2 = map(int, box)

        # Extract ROI and convert to HSV
        roi = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Color classification
        bean_color = "Unknown"
        for variety, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
            match_ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1]) if roi.size > 0 else 0

            if match_ratio > 0.3:
                bean_color = variety
                break

        combined_label = f"{label} ({bean_color}) ({confidence:.2f})"
        color = colors[label]

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(combined_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, combined_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add to dashboard info
        dashboard_data.append({
            "label": label,
            "variety": bean_color,
            "confidence": confidence
        })

    # Create dashboard panel
    dashboard = create_dashboard(dashboard_data, height=frame.shape[0])

    # Combine both views
    combined = np.hstack((frame, dashboard))

    cv2.imshow("Cacao Detection Dashboard", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")
