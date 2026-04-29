# gaze_detector.py
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# iris indices for MediaPipe FaceMesh
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

class GazeDetector:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    def estimate(self, frame_rgb):
        h, w = frame_rgb.shape[:2]
        res = self.face_mesh.process(frame_rgb)
        if not res.multi_face_landmarks:
            return "no_face"
        lm = res.multi_face_landmarks[0].landmark

        # Left iris center
        lxs = [lm[i].x * w for i in LEFT_IRIS]
        lys = [lm[i].y * h for i in LEFT_IRIS]
        lcx, lcy = np.mean(lxs), np.mean(lys)

        # Right iris center
        rxs = [lm[i].x * w for i in RIGHT_IRIS]
        rys = [lm[i].y * h for i in RIGHT_IRIS]
        rcx, rcy = np.mean(rxs), np.mean(rys)

        # Eye corner positions for normalization
        left_outer = np.array([lm[33].x * w, lm[33].y * h])
        left_inner = np.array([lm[133].x * w, lm[133].y * h])
        right_outer = np.array([lm[362].x * w, lm[362].y * h])
        right_inner = np.array([lm[263].x * w, lm[263].y * h])

        # relative position across eye width (0=left, 1=right)
        left_rel = (lcx - left_outer[0]) / (left_inner[0] - left_outer[0] + 1e-6)
        right_rel = (rcx - right_inner[0]) / (right_outer[0] - right_inner[0] + 1e-6)
        # average and clamp
        rel = np.clip((left_rel + (1 - right_rel)) / 2.0, 0.0, 1.0)

        # thresholds
        if rel < 0.35:
            return "left"
        elif rel > 0.65:
            return "right"
        else:
            return "center"
