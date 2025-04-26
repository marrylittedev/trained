import cv2
import numpy as np
from roboflow import Roboflow
import tkinter as tk
from PIL import Image, ImageTk
import datetime
import time

# Initialize Roboflow
rf = Roboflow(api_key="f4UBb9Y1BqAaVoiasTC1")
project = rf.workspace("cacaotrain").project("trained-q5iwo")
model = project.version(2).model

# HSV color thresholds
criollo_lower = np.array([0, 70, 50])
criollo_upper = np.array([10, 255, 255])
forastero_lower = np.array([130, 50, 50])
forastero_upper = np.array([170, 255, 255])
trinitario_lower = np.array([10, 100, 20])
trinitario_upper = np.array([25, 255, 200])
min_match_threshold = 10.0

# Setup Tkinter window
root = tk.Tk()
root.title("Cacao Detection Dashboard")

# Left: video panel
video_label = tk.Label(root)
video_label.grid(row=0, column=0, rowspan=2)

# Right: dashboard panel
dashboard = tk.Frame(root)
dashboard.grid(row=0, column=1, padx=10, sticky="n")

criollo_var = tk.StringVar()
forastero_var = tk.StringVar()
trinitario_var = tk.StringVar()
unknown_var = tk.StringVar()
confidence_var = tk.StringVar()
match_var = tk.StringVar()
current_bin_var = tk.StringVar()

tk.Label(dashboard, text="🧠 Detection Summary", font=("Arial", 14, "bold")).pack()
tk.Label(dashboard, textvariable=criollo_var).pack()
tk.Label(dashboard, textvariable=forastero_var).pack()
tk.Label(dashboard, textvariable=trinitario_var).pack()
tk.Label(dashboard, textvariable=unknown_var).pack()
tk.Label(dashboard, textvariable=confidence_var).pack()
tk.Label(dashboard, textvariable=match_var).pack()

# Detection history viewer
history_label = tk.Label(dashboard, text="\n📜 Detection History", font=("Arial", 14, "bold"))
history_label.pack(pady=(10, 0))

history_box = tk.Listbox(dashboard, width=40, height=10)
history_box.pack()

# Exit button
def close_app():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

exit_button = tk.Button(dashboard, text="❌ Exit", font=("Arial", 12), command=close_app, bg="red", fg="white")
exit_button.pack(pady=10)

# Detection counts
counts = {
    "Criollo": 0,
    "Forastero": 0,
    "Trinitario": 0,
    "Unknown": 0
}

# Start camera
cap = cv2.VideoCapture(1)
time.sleep(0.2)

def update():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update)
        return

    frame = cv2.resize(frame, (640, 480))
    cv2.imwrite("frame.jpg", frame)

    predictions = model.predict("frame.jpg", confidence=40, overlap=30).json()

    # Reset counts
    for key in counts:
        counts[key] = 0

    last_confidence = 0
    last_match = 0
    detected_type = "Unknown"

    best_pred = None
    best_conf = 0

    # Find most confident detection
    for pred in predictions['predictions']:
        if pred['confidence'] > best_conf:
            best_pred = pred
            best_conf = pred['confidence']

    if best_pred:
        x, y, w, h = int(best_pred['x']), int(best_pred['y']), int(best_pred['width']), int(best_pred['height'])
        label = best_pred['class']
        confidence = best_pred['confidence']
        last_confidence = confidence

        x1, y1 = max(x - w // 2, 0), max(y - h // 2, 0)
        x2, y2 = min(x + w // 2, frame.shape[1]), min(y + h // 2, frame.shape[0])
        crop = frame[y1:y2, x1:x2]
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        criollo_mask = cv2.inRange(hsv_crop, criollo_lower, criollo_upper)
        forastero_mask = cv2.inRange(hsv_crop, forastero_lower, forastero_upper)
        trinitario_mask = cv2.inRange(hsv_crop, trinitario_lower, trinitario_upper)

        criollo_ratio = (cv2.countNonZero(criollo_mask) / (crop.size / 3)) * 100
        forastero_ratio = (cv2.countNonZero(forastero_mask) / (crop.size / 3)) * 100
        trinitario_ratio = (cv2.countNonZero(trinitario_mask) / (crop.size / 3)) * 100

        color_ratios = {
            "Criollo": criollo_ratio,
            "Forastero": forastero_ratio,
            "Trinitario": trinitario_ratio
        }

        detected_type = "Unknown"
        match_percent = 0.0

        for variety, ratio in color_ratios.items():
            if ratio > match_percent and ratio >= min_match_threshold:
                detected_type = variety
                match_percent = ratio

        counts[detected_type] += 1
        last_match = match_percent

        # Draw detection box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} | {detected_type} ({match_percent:.1f}%)",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Add to history
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        history_box.insert(0, f"[{timestamp}] {label} → {detected_type} ({match_percent:.1f}%)")
        if history_box.size() > 10:
            history_box.delete(10)

    # Update UI
    criollo_var.set(f"Criollo: {counts['Criollo']}")
    forastero_var.set(f"Forastero: {counts['Forastero']}")
    trinitario_var.set(f"Trinitario: {counts['Trinitario']}")
    unknown_var.set(f"Unknown: {counts['Unknown']}")
    confidence_var.set(f"Last Confidence: {last_confidence:.1f}%")
    match_var.set(f"Last Color Match: {last_match:.1f}%")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update)

# Start loop
update()
root.mainloop()
