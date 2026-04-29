import cv2
import mediapipe as mp
import numpy as np
import time
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

# ============================
#  MediaPipe Setup (Face Mesh for Blink)
# ============================
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# Eye landmark indices for left and right eyes (for EAR calculation)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_points, w, h):
    pts = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_points]
    # Vertical distances
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # Horizontal distance
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# ============================
#  GUI Application
# ============================
class ProctorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Exam Proctoring System – GUI")
        self.root.geometry("1000x700")
        self.root.configure(bg="#121212")

        # States
        self.cap = None
        self.running = False

        self.blink_text = StringVar(value="Blink: -")
        self.gaze_text = StringVar(value="Gaze: -")
        self.face_text = StringVar(value="Face: -")
        self.fps_text = StringVar(value="FPS: -")

        self.blink_counter = 0
        self.close_frames = 0
        self.prev_time = 0

        # cheating
        self.multi_face_violation = 0

        # Face Mesh
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5)

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ============================
    #  Build GUI
    # ============================
    def build_ui(self):
        title = Label(
            self.root,
            text="AI-Powered Online Exam Proctoring",
            font=("Segoe UI", 20, "bold"),
            bg="#121212",
            fg="#00E676"
        )
        title.pack(pady=10)

        control_frame = Frame(self.root, bg="#1E1E1E", bd=2, relief=RIDGE)
        control_frame.pack(side=TOP, fill=X, padx=10, pady=5)

        self.start_btn = Button(
            control_frame, text="▶ Start Proctoring",
            font=("Segoe UI", 11, "bold"),
            bg="#00C853", fg="white",
            command=self.start_proctoring
        )
        self.start_btn.pack(side=LEFT, padx=10, pady=5)

        self.stop_btn = Button(
            control_frame, text="■ Stop",
            font=("Segoe UI", 11, "bold"),
            bg="#D50000", fg="white",
            state=DISABLED,
            command=self.stop_proctoring
        )
        self.stop_btn.pack(side=LEFT, padx=5, pady=5)

        self.exam_btn = Button(
            control_frame, text="📝 Demo Online Exam",
            font=("Segoe UI", 11, "bold"),
            bg="#2962FF", fg="white",
            command=self.open_exam
        )
        self.exam_btn.pack(side=LEFT, padx=15, pady=5)

        exit_btn = Button(
            control_frame, text="✖ Exit",
            font=("Segoe UI", 11, "bold"),
            bg="#424242", fg="white",
            command=self.on_close
        )
        exit_btn.pack(side=RIGHT, padx=10, pady=5)

        video_frame = Frame(self.root, bg="#121212")
        video_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.video_label = Label(video_frame, bg="#000000")
        self.video_label.pack(fill=BOTH, expand=True)

        info_frame = Frame(self.root, bg="#1E1E1E", bd=2, relief=RIDGE)
        info_frame.pack(side=BOTTOM, fill=X, padx=10, pady=5)

        Label(info_frame, textvariable=self.face_text,
              bg="#1E1E1E", fg="white", font=("Segoe UI", 11)).pack(side=LEFT, padx=15)
        Label(info_frame, textvariable=self.blink_text,
              bg="#1E1E1E", fg="white", font=("Segoe UI", 11)).pack(side=LEFT, padx=15)
        Label(info_frame, textvariable=self.gaze_text,
              bg="#1E1E1E", fg="white", font=("Segoe UI", 11)).pack(side=LEFT, padx=15)
        Label(info_frame, textvariable=self.fps_text,
              bg="#1E1E1E", fg="white", font=("Segoe UI", 11)).pack(side=LEFT, padx=15)

    # ============================
    #  Start Proctoring
    # ============================
    def start_proctoring(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Webcam not found.")
            return

        self.running = True
        self.start_btn.config(state=DISABLED)
        self.stop_btn.config(state=NORMAL)

        self.blink_counter = 0
        self.close_frames = 0
        self.multi_face_violation = 0

        self.update_frame()

    # ============================
    #  Stop Proctoring
    # ============================
    def stop_proctoring(self):
        self.running = False
        self.start_btn.config(state=NORMAL)
        self.stop_btn.config(state=DISABLED)

        if self.cap:
            self.cap.release()

        self.video_label.config(image="")
        self.face_text.set("Face: -")
        self.blink_text.set("Blink: -")
        self.gaze_text.set("Gaze: -")
        self.fps_text.set("FPS: -")

    # ============================
    #  Update Frame
    # ============================
    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_proctoring()
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        face_count = 0
        gaze = "Center"

        if results.multi_face_landmarks:
            face_count = len(results.multi_face_landmarks)

            # MULTI FACE ALERT
            if face_count > 1:
                self.multi_face_violation += 1
                cv2.putText(frame, "⚠ Multiple Faces Detected!", (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                if self.multi_face_violation >= 3:
                    messagebox.showerror("Cheating Alert", "Multiple faces detected repeatedly. Exam terminated!")
                    self.on_close()
                    return
            else:
                self.multi_face_violation = 0

            # BLINK DETECTION
            for face_landmarks in results.multi_face_landmarks:
                left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_IDX, w, h)
                right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_IDX, w, h)
                ear = (left_ear + right_ear) / 2.0

                # Blink threshold
                if ear < 0.23:
                    self.close_frames += 1
                else:
                    if self.close_frames >= 2:
                        self.blink_counter += 1
                    self.close_frames = 0

                # Draw rectangle around face
                mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                       mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                       mp_draw.DrawingSpec(color=(0,0,255), thickness=1))

        # Update labels
        self.face_text.set(f"Face: {face_count}")
        self.blink_text.set(f"Blink: {self.blink_counter}")
        self.gaze_text.set(f"Gaze: {gaze}")

        # FPS
        curr = time.time()
        fps = 1 / (curr - self.prev_time) if self.prev_time else 0
        self.prev_time = curr
        self.fps_text.set(f"FPS: {int(fps)}")

        # Tkinter update
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((900, 500))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    # ============================
    #  Demo Exam Window
    # ============================
    def open_exam(self):
        exam = Toplevel(self.root)
        exam.title("Demo Online Exam")
        exam.geometry("600x400")
        Label(exam, text="Sample Online Exam", font=("Segoe UI", 18, "bold")).pack(pady=20)

    # ============================
    #  Close App
    # ============================
    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


# ============================
#  Run App
# ============================
if __name__ == "__main__":
    root = Tk()
    app = ProctorGUI(root)
    root.mainloop()
