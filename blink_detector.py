# blink_detector.py
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh

# landmark indices for eye region (MediaPipe face mesh)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]   # approximate
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380] # approximate

def eye_aspect_ratio(landmarks, eye_idx, image_w, image_h):
    # landmarks: list of mp_landmark objects
    coords = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in eye_idx]
    # Using vertical/horizontal distances: (p2,p6), (p3,p5), (p1,p4)
    # Here index mapping based on eye_idx list (adapted for medipipe indices)
    A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]) + 1e-6)
    ear = (A + B) / (2.0 * C)
    return ear

class BlinkDetector:
    def __init__(self, ear_threshold=0.20, closed_time_threshold=0.3):
        self.ear_threshold = ear_threshold
        self.closed_time_threshold = closed_time_threshold
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.closed_since = None
        self.blink_count = 0
        self.prev_closed = False

    def update(self, frame_rgb):
        img_h, img_w = frame_rgb.shape[:2]
        results = self.face_mesh.process(frame_rgb)
        is_closed = False
        closed_duration = 0.0
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left_ear = eye_aspect_ratio(lm, LEFT_EYE_IDX, img_w, img_h)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE_IDX, img_w, img_h)
            ear = (left_ear + right_ear) / 2.0

            # closed if ear below threshold
            if ear < self.ear_threshold:
                if self.closed_since is None:
                    self.closed_since = time.time()
                is_closed = True
                closed_duration = time.time() - self.closed_since
            else:
                if self.closed_since is not None:
                    # eye opened now -> count as blink if duration short
                    dur = time.time() - self.closed_since
                    if dur >= 0.05 and dur <= 1.0:
                        self.blink_count += 1
                    self.closed_since = None
                is_closed = False
                closed_duration = 0.0
        else:
            # no face: treat as open (or you might want to flag)
            is_closed = False
            closed_duration = 0.0

        return is_closed, closed_duration, self.blink_count
