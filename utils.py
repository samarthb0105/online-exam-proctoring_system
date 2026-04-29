# utils.py
import csv
import os
from datetime import datetime

class Logger:
    def __init__(self, path="results/session_log.csv"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "a", newline="")
        self.writer = csv.writer(self.file)
        # write header if file empty
        if os.path.getsize(path) == 0:
            self.writer.writerow(["timestamp", "event", "details"])

    def log_event(self, event, details=""):
        ts = datetime.utcnow().isoformat()
        self.writer.writerow([ts, event, details])
        self.file.flush()
        print(f"[LOG] {ts} {event} {details}")

    def close(self):
        self.file.close()
