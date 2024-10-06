import cv2
from ultralytics import YOLO
import torch
import numpy as np
from collections import deque, defaultdict
import matplotlib.pyplot as plt

# Load model
model = YOLO(r'D:\@Code\Python\ProjectASL2\runs\detect\train9\weights\best.pt')

# innitialize camera
cap = cv2.VideoCapture(0) 

# Parameters for frame
buffer_size = 7 
detection_buffer = deque(maxlen=buffer_size)

# statistic store
class_stats = defaultdict(lambda: {"total": 0, "correct": 0})

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert frame to tensor
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Predict the bounding boxes
    results = model(frame_tensor)

    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            label = result.names[int(box.cls)]
            confidence = box.conf[0].cpu().numpy()  # accuracy

            # Add buffer detection
            detection_buffer.append((label, confidence, (x1, y1, x2, y2)))

            # Update class statistics
            class_stats[label]["total"] += 1
            if confidence > 0.5:  # Consider it correct if confidence > 0.5
                class_stats[label]["correct"] += 1

    # Find the highest confidence detection in the buffer
    if detection_buffer:
        highest_confidence_detection = max(detection_buffer, key=lambda x: x[1])
        label, confidence, (x1, y1, x2, y2) = highest_confidence_detection

        # Draw rectangle and label with accuracy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f'{label}: {confidence:.2f}'
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the result with the highest confidence in the top corner
    if detection_buffer:
        label, confidence, _ = highest_confidence_detection
        cv2.putText(frame, f'Highest: {label} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Sign Language Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# Calculate overall accuracy for each class
class_names = []
accuracies = []
total_detections = 0
correct_detections = 0

for class_name, stats in class_stats.items():
    total = stats["total"]
    correct = stats["correct"]
    accuracy = correct / total if total > 0 else 0
    class_names.append(class_name)
    accuracies.append(accuracy * 100)  # Convert to percentage

    # Update overall stats
    total_detections += total
    correct_detections += correct

# Calculate overall accuracy
overall_accuracy = correct_detections / total_detections * 100 if total_detections > 0 else 0
print(f"Overall Accuracy: {overall_accuracy:.2f}%")

# Plot the accuracies
plt.figure(figsize=(10, 6))
plt.barh(class_names, accuracies, color='skyblue')
plt.xlabel('Accuracy (%)')
plt.title('Class-wise Accuracy')
plt.grid(True)
plt.show()

# Display the overall accuracy
plt.figure(figsize=(6, 4))
plt.bar(['Overall'], [overall_accuracy], color='salmon')
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('Overall Accuracy')
plt.grid(True)
plt.show()