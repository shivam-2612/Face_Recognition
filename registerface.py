import cv2
import face_recognition
import numpy as np
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["face_recognition_db"]
faces_col = db["faces"]

person_name = input("Enter the name of the person to register: ")

cap = cv2.VideoCapture(0)
captured_encodings = []

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    
    for i, box in enumerate(boxes):
        top, right, bottom, left = box
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, f"Face {i+1}", (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Select Face to Register", frame)
    key = cv2.waitKey(1) & 0xFF

    # Press 1,2,3,... to select which face to register
    if key >= ord('1') and key < ord('1') + len(boxes):
        index = key - ord('1')
        enc = face_recognition.face_encodings(rgb, boxes)[index]
        captured_encodings.append(enc)
        print(f"Selected Face {index+1} for registration.")
        break

    # Press q to quit without registering
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if not captured_encodings:
    print("No face selected for registration.")
    exit()

new_encoding = captured_encodings[0]

# Check if already exists
exists = False
for face in faces_col.find():
    known_enc = np.array(face["encoding"])
    if face_recognition.face_distance([known_enc], new_encoding)[0] < 0.45:
        print(f"{face['name']} is already registered.")
        exists = True
        break

if not exists:
    faces_col.insert_one({"name": person_name, "encoding": new_encoding.tolist()})
    print(f"New person registered: {person_name}")
