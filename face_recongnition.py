import cv2
import face_recognition
import numpy as np
from pymongo import MongoClient

# Load encodings from MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["face_recognition_db"]
collection = db["faces"]

data = list(collection.find({}))
known_encodings = [np.array(d["encodings"]) for d in data]
known_names = [d["name"] for d in data]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, boxes)

    for (top, right, bottom, left), enc in zip(boxes, encodings):
        matches = face_recognition.compare_faces(known_encodings, enc)
        name = "Unknown"

        if True in matches:
            match_idx = matches.index(True)
            name = known_names[match_idx]

        # Draw result
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
