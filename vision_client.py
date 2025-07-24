# === client.py ===
import cv2
import time
import requests
import json
import logging
from datetime import datetime  # Import datetime module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SERVER_URL = " https://api-visionserver-dev-wus-001.azurewebsites.net/upload/"
SERVER_URL = " http://localhost:7000/upload/"
CAPTURE_INTERVAL = 2  # seconds
robot_id = "robot_1"

cap = cv2.VideoCapture(0)
last_capture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show live video feed
    cv2.imshow("Live Feed", frame)

    current_time = time.time()
    if current_time - last_capture_time >= CAPTURE_INTERVAL:
        last_capture_time = current_time

        # Save captured frame to memory
        _, img_encoded = cv2.imencode(".jpg", frame)

        # Generate a unique filename using the current date and time
        unique_filename = datetime.now().strftime("image%Y%m%d%H%M%S%f.jpg")
        files = {"file": (unique_filename, img_encoded.tobytes(), "image/jpeg")}

        try:
            local_time_vision = datetime.now().isoformat()
            dt_vision = datetime.fromisoformat(local_time_vision)
            formatted_time_vision = dt_vision.strftime("%H")
            hour = int(formatted_time_vision)  # Convert to integer

            response = requests.post(SERVER_URL, files=files, data={"robot_id": robot_id, "local_time_vision": hour})
            result = response.json()
            # print("Server Response:", json.dumps(result, indent=2))

            # Draw bounding boxes from handup_result
            handup_boxes = result.get("handup_result", {}).get("bounding_boxes", [])
            for box in handup_boxes:
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                label = box.get("label", "")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            logger.info(f"\n----------------\n✅ handup_boxes: {handup_boxes}\n----------------")

            # Draw bounding boxes from face_recognition_result
            face_boxes = result.get("face_recognition_result", {}).get("bounding_boxes", [])
            for box in face_boxes:
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                label = box.get("label", "Face")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            logger.info(f"\n----------------\n✅ face_boxes: {face_boxes}\n----------------")

            # Show captured frame with bounding boxes
            cv2.imshow("Captured", frame)

        except Exception as e:
            # print("Failed to send image:", e)
            logger.error(f"\n----------------\n❌ Failed to send image: {e}\n----------------")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
