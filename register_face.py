import cv2
import face_recognition
import numpy as np
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["face_recognition_db"]
faces_col = db["faces"]

cap = cv2.VideoCapture(0)
person_name = input("Enter the name of person: ")

captured_encodings = []

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)

    if boxes:
        enc = face_recognition.face_encodings(rgb, boxes)[0]
        captured_encodings.append(enc)

        top, right, bottom, left = boxes[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, f"Capturing {len(captured_encodings)}", (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Register", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or len(captured_encodings) >= 5:
        break

cap.release()
cv2.destroyAllWindows()

if not captured_encodings:
    print("No face captured.")
    exit()

# Pick the *first clear encoding* instead of averaging
new_encoding = captured_encodings[0]

# Check if already exists
all_faces = list(faces_col.find())
for face in all_faces:
    known_enc = np.array(face["encoding"])
    distance = face_recognition.face_distance([known_enc], new_encoding)[0]
    if distance < 0.45:
        print(f"Already registered as: {face['name']} (ID: {face['_id']})")
        exit()

# Save new user
faces_col.insert_one({
    "name": person_name,
    "encoding": new_encoding.tolist()
})
print(f"New person registered: {person_name}")
