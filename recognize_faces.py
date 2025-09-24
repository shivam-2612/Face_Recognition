import cv2
import face_recognition
import numpy as np
from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb://localhost:27017")
db = client["face_recognition_db"]
faces_col = db["faces"]
att_col = db["attendance"]

# Load known encodings
data = list(faces_col.find())
known_encodings = [np.array(d["encoding"]) for d in data]
known_names = [d["name"] for d in data]
known_ids = [d["_id"] for d in data]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    for (top, right, bottom, left), enc in zip(boxes, encodings):
        name = "Unknown"

        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, enc)
            min_idx = np.argmin(distances)
            if distances[min_idx] < 0.45:  # threshold
                name = known_names[min_idx]
                user_id = known_ids[min_idx]

                # Mark attendance if not already marked today
                today = datetime.now().strftime("%Y-%m-%d")
                existing = att_col.find_one({"user_id": user_id, "date": today})
                if not existing:
                    att_col.insert_one({
                        "user_id": user_id,
                        "name": name,
                        "date": today,
                        "time": datetime.now().strftime("%H:%M:%S")
                    })
                    print(f"Attendance marked for {name} on {today}")

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
