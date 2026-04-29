import cv2
import mediapipe as mp
import math
from datetime import datetime

# ---------------- Mediapipe Setup ----------------
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ---------------- Webcam Setup ----------------
cap = cv2.VideoCapture(0)

# ---------------- Eye Aspect Ratio ----------------
def eye_aspect_ratio(landmarks, eye_indices):
    left = landmarks[eye_indices[0]]
    right = landmarks[eye_indices[3]]
    top = landmarks[eye_indices[1]]
    bottom = landmarks[eye_indices[2]]
    horizontal = math.dist((left.x, left.y), (right.x, right.y))
    vertical = math.dist((top.x, top.y), (bottom.x, bottom.y))
    return vertical / horizontal

# Mediapipe eye landmarks
LEFT_EYE = [33, 159, 145, 133]
RIGHT_EYE = [362, 386, 374, 263]

# ---------------- Logs ----------------
events_log = []

# Blink cooldown (frames to ignore after a blink)
blink_cooldown = 0
BLINK_THRESHOLD = 0.27  # Adjust if EAR too sensitive

# ---------------- Run Detection ----------------
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process face and mesh
        face_results = face_detection.process(rgb_frame)
        mesh_results = face_mesh.process(rgb_frame)

        alert_text = ""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ---------------- Two Faces Detection ----------------
        if face_results.detections and len(face_results.detections) >= 2:
            alert_text += "ALERT: Two faces detected! "
            events_log.append(f"{timestamp} - Two faces detected")

        # ---------------- Eye Blink Detection ----------------
        if blink_cooldown > 0:
            blink_cooldown -= 1
        elif mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE)
                right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE)

                # Print EAR values for debugging
                print(f"Left EAR: {left_ear:.3f}, Right EAR: {right_ear:.3f}")

                if left_ear < BLINK_THRESHOLD or right_ear < BLINK_THRESHOLD:
                    alert_text += "Blink detected! "
                    events_log.append(f"{timestamp} - Blink detected")
                    blink_cooldown = 5  # wait 5 frames before next blink
                    break

        # ---------------- Draw Faces ----------------
        if face_results.detections:
            for detection in face_results.detections:
                mp_drawing.draw_detection(frame, detection)

        # ---------------- Overlay Alert ----------------
        if alert_text:
            cv2.putText(frame, alert_text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Face & Blink Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ---------------- Cleanup ----------------
cap.release()
cv2.destroyAllWindows()

# Save logs
with open("events_log.txt", "w") as f:
    for event in events_log:
        f.write(event + "\n")

print("Event log saved to events_log.txt")
