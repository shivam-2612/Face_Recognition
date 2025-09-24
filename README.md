# Face Recognition Based Attendance System

This project is an **Attendance Marking System** using **OpenCV**, **face_recognition**, **dlib**, and **MongoDB**.  
It allows registering faces, recognizing them in real time, and marking attendance automatically.  

## 📂 Project Structure
- `recognize_face.py` → Runs the recognition model and marks attendance.  
  - When a registered user appears for the **first time in a day**, attendance is saved in MongoDB.  
  - If the same user comes again on the same day, the system **only displays their name** (no duplicate attendance).
    
- `register_face.py` → Registers a **single detected face** with a given name.  
  - If already saved in the database, it shows: *"Already saved with this name"*.
    
- `registerface.py` → Supports **multi-face detection** in the frame.  
  - All faces are detected with rectangles and temporary labels (*face1, face2, …*).  
  - User can press the respective number key to select and register that face with a desired name.  

## ⚙️ Setup Instructions

1. **Install Python**  
   - Tested with **Python 3.9.5**.  
   - Ensure `dlib` is installed manually (since prebuilt wheels may not be available for all platforms).  

2. **Clone the Repository**
   ```bash
   git clone <https://github.com/shivam-2612/Face_Recognition.git>
   cd face-attendance
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Setup MongoDB**

   * Install and run MongoDB locally or use MongoDB Atlas.
   * Update the MongoDB connection string in scripts if needed.

5. **Run Scripts**

   * Start attendance recognition:

     ```bash
     python recognize_face.py
     ```
   * Register a single face:

     ```bash
     python register_face.py
     ```
   * Register multiple faces:

     ```bash
     python registerface.py
     ```

## 📚 Libraries Used

* OpenCV (cv2)
* face\_recognition
* numpy
* dlib
* pymongo

## 🗄 MongoDB Schema Example

### **Faces Collection**

Stores registered face encodings.

```json
{
  "name": "Shivam",
  "encoding": [0.123, -0.456, ...],
  "created_at": "2025-09-23T12:34:56"
}
```

### **Attendance Collection**

Stores daily attendance records.

```json
{
  "name": "Shivam",
  "date": "2025-09-23",
  "time": "09:15:32"
}
```

### 🔑 Attendance Logic

* **First detection of the day** → Attendance marked with date and time.
* **Subsequent detections same day** → Only the name is displayed (no duplicate entry).

## 📝 Notes

* Ensure **dlib** is properly compiled and installed for Python 3.9.5.
* Face encodings are stored in MongoDB for persistence.
* Attendance accuracy depends on camera quality and lighting.
* Best results occur with front-facing faces and stable lighting.
