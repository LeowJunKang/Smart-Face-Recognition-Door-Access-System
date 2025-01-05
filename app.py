from flask import Flask, render_template, Response, jsonify, request
import cv2
import dlib
import numpy as np
import mysql.connector

app = Flask(__name__)

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

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

face_capture_count = 0
MAX_FACES = 50
username = None
is_capturing = False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/status')
def status():
    global face_capture_count
    progress = int((face_capture_count / MAX_FACES) * 100)
    if face_capture_count >= MAX_FACES:
        return jsonify({"message": "Done capturing faces.", "progress": 100})
    else:
        return jsonify({"message": f"Capturing faces progress: {progress}%", "progress": progress})


@app.route('/start_capture', methods=['POST'])
def start_capture():
    global face_capture_count, username, is_capturing

    data = request.get_json()
    username = data.get('username') if data else None

    if not username:
        return jsonify({"message": "Username is required!"}), 400

    face_capture_count = 0
    is_capturing = True
    print(f"Starting capture for username: {username}")
    return jsonify({"message": "Capture started successfully."})


@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    global is_capturing
    is_capturing = False
    print("Capture stopped.")
    return jsonify({"message": "Capture stopped."})


def generate_frames():
    global face_capture_count, username, is_capturing
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if is_capturing:
            for face in faces:
                if face_capture_count >= MAX_FACES:
                    is_capturing = False
                    break

                landmarks = sp(frame, face)
                face_descriptor = face_recognizer.compute_face_descriptor(frame, landmarks)
                face_descriptor_array = np.array(face_descriptor)
                face_descriptor_bytes = face_descriptor_array.tobytes()

                try:
                    cursor.execute(
                        "INSERT INTO user_faces (username, face_descriptor) VALUES (%s, %s)",
                        (username, face_descriptor_bytes)
                    )
                    conn.commit()
                    face_capture_count += 1
                except Exception as e:
                    print(f"Error saving to database: {e}")

                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Could not encode frame to JPEG.")
            break
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/reset_capture', methods=['POST'])
def reset_capture():
    global face_capture_count
    face_capture_count = 0
    return jsonify({"message": "Capture process has been reset."})


if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        cap.release()
        cursor.close()
        conn.close()
