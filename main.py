# main.py
import cv2
import time
from blink_detector import BlinkDetector
from gaze_detector import GazeDetector
from utils import Logger

def main():
    cam_id = 0  # change if needed
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    blink_detector = BlinkDetector()
    gaze_detector = GazeDetector()
    logger = Logger("results/session_log.csv")

    suspicious_start = None
    ALERT_BLINK_THRESHOLD = 2.0  # seconds eyes closed -> alert
    ALERT_LOOK_AWAY_COUNT = 5    # number of consecutive away frames -> alert

    look_away_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Blink detection (returns: is_closed(bool), closed_duration_sec(float), blink_count(int))
            is_closed, closed_duration, blink_count = blink_detector.update(frame_rgb)

            # Gaze detection (returns 'center'|'left'|'right'|'down' etc.)
            gaze_dir = gaze_detector.estimate(frame_rgb)

            # Display overlay
            h, w = frame.shape[:2]
            cv2.putText(frame, f"Blink Count: {blink_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
            cv2.putText(frame, f"Eyes Closed: {is_closed} ({closed_duration:.2f}s)", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)
            cv2.putText(frame, f"Gaze: {gaze_dir}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)

            # Determine suspiciousness
            suspicious = False
            if is_closed and closed_duration >= ALERT_BLINK_THRESHOLD:
                suspicious = True
                logger.log_event("LongEyeClosure", f"duration={closed_duration:.2f}s")
            if gaze_dir != "center":
                look_away_count += 1
            else:
                look_away_count = 0

            if look_away_count >= ALERT_LOOK_AWAY_COUNT:
                suspicious = True
                logger.log_event("LookAway", f"consecutive={look_away_count}")

            if suspicious:
                cv2.putText(frame, "ALERT: Suspicious Behavior", (10,h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),3)

            # Show frame
            cv2.imshow("Proctoring", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                # save screenshot
                ts = int(time.time())
                cv2.imwrite(f"results/screenshot_{ts}.jpg", frame)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.close()
        print("Session ended. Logs saved to results/session_log.csv")

if __name__ == "__main__":
    main()
