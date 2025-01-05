import cv2
import dlib
import numpy as np
import mysql.connector

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    port=3308,
    database="face_recognition"
)
cursor = conn.cursor()

known_face_descriptors = []
known_labels = []

cursor.execute("SELECT id, face_descriptor FROM user_faces")
rows = cursor.fetchall()

for row in rows:
    label = row[0]
    descriptor_blob = row[1]

    descriptor = np.frombuffer(descriptor_blob, dtype=np.float64)
    known_face_descriptors.append(descriptor)
    known_labels.append(label)


def get_username(label):
    try:
        cursor.execute("SELECT username FROM user_faces WHERE id = %s", (label,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            return "Unknown User"
    except Exception as e:
        print(f"Database error: {e}")
        return "Error"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(known_face_descriptors) == 0:
        username = "No training data"
        access_message = "No Faces Available"
        access_color = (0, 0, 255)
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), access_color, 2)
            cv2.putText(frame, f"{username}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.putText(frame, access_message, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, access_color, 2)
    else:
        for face in faces:
            landmarks = sp(frame, face)
            face_descriptor = face_recognizer.compute_face_descriptor(frame, landmarks)

            distances = np.linalg.norm(np.array(known_face_descriptors) - face_descriptor, axis=1)
            min_distance_index = np.argmin(distances)
            label = known_labels[min_distance_index]
            confidence = distances[min_distance_index]

            username = get_username(label)

            if confidence < 0.30:
                access_message = "Allow Access"
                access_color = (0, 255, 0)
            else:
                access_message = "Access Denied"
                access_color = (0, 0, 255)

            if access_message == "Access Denied":
                username = "Unknown User"

            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), access_color, 2)
            cv2.putText(frame, f"{username}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.putText(frame, access_message, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, access_color, 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

cursor.close()
conn.close()
