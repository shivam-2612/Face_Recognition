import cv2
import face_recognition
import numpy as np
from pymongo import MongoClient
from bson.objectid import ObjectId

client = MongoClient("mongodb://localhost:27017")
db = client["face_recognition_db"]
collection = db["faces"]

def load_known_faces():
    data = list(collection.find({}))
    known_encodings = [np.array(d["encodings"]) for d in data]
    known_names = [d["name"] for d in data]
    known_ids = [str(d["_id"]) for d in data]
    return known_encodings, known_names, known_ids

known_encodings, known_names, known_ids = load_known_faces()
new_faces = {}  # temp storage for new faces in this session

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, boxes)

    for (top, right, bottom, left), enc in zip(boxes, encodings):
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
        name = "Unknown"
        user_id = None

        if True in matches:
            match_idx = matches.index(True)
            name = known_names[match_idx]
            user_id = known_ids[match_idx]
        else:
            found = False
            for k, v in new_faces.items():
                if face_recognition.compare_faces([v["encoding"]], enc, tolerance=0.5)[0]:
                    name = v["name"]
                    user_id = v["id"]
                    found = True
                    break
            if not found:
                cv2.imshow("Recognition", frame)
                print("New face detected! Please enter your name in the terminal:")
                input_name = input("Enter name for new face: ")
                face_record = {
                    "name": input_name,
                    "encodings": enc.tolist()
                }
                result = collection.insert_one(face_record)
                user_id = str(result.inserted_id)
                name = input_name
                known_encodings.append(enc)
                known_names.append(name)
                known_ids.append(user_id)
                new_faces[user_id] = {"name": name, "encoding": enc, "id": user_id}
                print(f"Saved new user '{name}' with id {user_id}")

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, f"{name}", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()