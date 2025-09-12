import cv2
import face_recognition
import numpy as np
from pymongo import MongoClient

# MongoDB setup
client = MongoClient("mongodb://localhost:27017")
db = client["face_recognition_db"]
collection = db["faces"]

cap = cv2.VideoCapture(0)
person_name = input("Enter the name of person: ")
encodings = []
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype('uint8')

    # Detect face
    boxes = face_recognition.face_locations(rgb_frame)
    if len(boxes) == 0:
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    

    # Get encoding
    enc = face_recognition.face_encodings(rgb_frame, boxes)[0]
    encodings.append(enc)
    print(f"Captured image: {len(encodings)}")

    # Draw rectangle
    top, right, bottom, left = boxes[0]
    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to MongoDB
face_record = {
    "name": person_name,
    "encodings": np.mean(encodings, axis=0).tolist()  # store average encoding
}
collection.insert_one(face_record)
print("Face data saved to MongoDB!")