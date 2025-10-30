# Ex.No.6 Development of Python Code Compatible with Multiple AI Tools

# Date: 30-10-2025
# Register no: 212223060224
# Aim: Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights with Multiple AI Tools

# AI Tools Required:
 ### OpenAI (ChatGPT / GPT-4 / GPT-5)
 ### Google Gemini (Generative AI)
 

# Explanation:
Experiment the persona pattern as a programmer for any specific applications related with your interesting area. 
Generate the outoput using more than one AI tool and based on the code generation analyse and discussing that. 
Experiment Steps with Example Prompts and Code


---



# Prompt :

🗣️ “Generate Python code to detect and recognize student faces using OpenCV and the face_recognition library. Store the attendance in a CSV file.”
## Project code:
```
import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)
known_faces = ["reshmi.jpg", "rahul.jpg"]
known_names = ["Reshmi", "Rahul"]

known_encodings = [face_recognition.face_encodings(face_recognition.load_image_file(img))[0] for img in known_faces]

with open("attendance.csv", "a", newline="") as file:
    writer = csv.writer(file)
    while True:
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, faces)
        for encoding, location in zip(encodings, faces):
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = "Unknown"
            if True in matches:
                name = known_names[matches.index(True)]
                writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                print(f"{name} marked present.")
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
```
## Explanation :
  This Python program uses OpenCV and face_recognition to take attendance automatically.
It opens the webcam, detects faces, compares them with saved student photos, and if a match is found, it writes the student’s name and time into a CSV file.
Press ‘q’ to stop the camera.

# GEMINI:
## Prompt:
  Real-Time Recognition and Attendance Marking
This script uses the saved encodings, opens the webcam, recognizes faces, and updates a CSV file (Attendance.csv).

Python Code: attendance_system.py
```
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os
import pickle

# --- Setup ---

# Load saved encodings and names
try:
    with open("encodings.pkl", "rb") as f:
        data = pickle.load(f)
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]
    print(f"Loaded {len(known_face_names)} known faces.")
except FileNotFoundError:
    print("Error: encodings.pkl not found. Run 'encode_faces.py' first.")
    exit()

# Open the default webcam
video_capture = cv2.VideoCapture(0)

# List to keep track of students whose attendance has been marked in this session
students_present = set()

# --- Attendance Functions ---

def mark_attendance(name):
    """Marks attendance in a CSV file only if the student hasn't been marked yet today."""
    
    # Check if student is already marked present
    if name in students_present:
        return

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # Define the attendance file name
    filename = 'Attendance.csv'
    
    # Check if file exists to determine if headers need to be written
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write headers if the file is newly created
        if not file_exists:
            writer.writerow(['Name', 'Date', 'Time'])
        
        # Write the attendance record
        writer.writerow([name, date_str, time_str])
        
        # Add the student to the 'present' set to avoid duplicate entries
        students_present.add(name)
        print(f"Attendance marked for {name} at {time_str}")


# --- Main Loop ---

print("Starting video stream. Press 'q' to quit.")

while True:
    # 1. Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Safety check
    if not ret:
        print("Failed to grab frame.")
        break

    # 2. Resize frame for faster processing (optional, but highly recommended)
    # The default 1/4 size is used (fx=0.25, fy=0.25)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # 3. Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # 4. Find all faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # 5. Loop through each face found in the current frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        # Compare the face with known encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the distance to each known face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        # Get the best match (smallest distance)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            # The recognized name
            name = known_face_names[best_match_index]
            # Mark the attendance
            mark_attendance(name)

        # Rescale the face location coordinates back to the original frame size (x4)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 6. Draw the box and label on the frame
        
        # Choose color based on recognition status
        box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) # Green for known, Red for unknown
        
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        
        # Show 'Present' if marked, otherwise show the name/Unknown
        display_text = f"{name} (Present)" if name in students_present else name
        
        cv2.putText(frame, display_text, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # 7. Display the resulting frame
    cv2.imshow('Student Attendance System', frame)

    # 8. Hit 'q' on the keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()

```
## Comparison table:
| **Aspect**                   | **ChatGPT Output**                                                          | **Gemini Output**                                                                 | **Comparison Summary**                                                                 |
| ---------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Code Length & Complexity** | Short and simple (~40 lines). Easy for beginners.                           | Longer and modular (~120 lines). Uses functions and structured flow.              | Gemini code is more detailed and production-ready. ChatGPT’s is simpler for quick use. |
| **Face Data Handling**       | Uses static images (`reshmi.jpg`, `rahul.jpg`). Loads and encodes directly. | Loads pre-saved encodings from a `pickle` file (`encodings.pkl`).                 | Gemini approach allows scalability (can handle many students easily).                  |
| **Attendance Management**    | Writes every detection directly to CSV (duplicates possible).               | Uses a function `mark_attendance()` to avoid duplicate entries and add date/time. | Gemini handles attendance more efficiently and avoids duplication.                     |
| **Performance Optimization** | Processes full-sized frames, which may slow down recognition.               | Resizes frames (`fx=0.25`, `fy=0.25`) for faster processing.                      | Gemini is optimized for real-time recognition.                                         |
| **Error Handling**           | No error handling.                                                          | Includes error handling (`FileNotFoundError`, frame checks).                      | Gemini has better reliability and robustness.                                          |
| **User Feedback**            | Prints when a student is marked present.                                    | Displays labeled bounding boxes and “Present” tag on video.                       | Gemini gives better visual feedback to users.                                          |
| **Readability & Comments**   | Few comments, straightforward.                                              | Well-commented, professional documentation style.                                 | Gemini is more readable and educational.                                               |

## 💡 Actionable Insights

## Combine Best Features from Both Tools:
- Use ChatGPT’s simple structure for quick setup.
- Integrate Gemini’s attendance management function to avoid duplicate entries.
- Improve Real-Time Performance:
- Apply frame resizing (as Gemini does) to speed up recognition.
- Use GPU acceleration if available (via dlib or CUDA).

## Enhance Scalability:
- Store face encodings in a database or pickle file (Gemini approach).
- Add a new student registration module to automatically encode and save new faces.
  
## Add Cloud Integration:
- Push attendance data to Google Sheets or Firebase for live tracking.
- Automate attendance reports generation weekly or monthly.

## Security & Accuracy:
- Encrypt stored encodings to protect identity data.
- Train with multiple images per student under different lighting conditions.

## Prompt Refinement Tip:
- Instead of just “Generate Python code,” you could refine your prompt as:
- “Generate modular Python code using OpenCV and face_recognition that performs real-time attendance marking, prevents duplicates, and optimizes processing speed.”
- This will push AI tools to produce more accurate and optimized results.

# Conclusion:
* ChatGPT focused on simplicity and ease of understanding.
* Gemini focused on structure, scalability, and optimization.
* Combining both approaches gives a powerful and efficient face recognition attendance system.


# Result: The corresponding Prompt is executed successfully.
