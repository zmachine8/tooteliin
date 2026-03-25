import threading
import cv2
import time
import numpy as np

class RTSPStreamReader:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not self.running: break
            with self.lock:
                self.ret = ret
                self.frame = frame
            if not ret: break

    def read(self):
        with self.lock:
            if self.frame is None: return self.ret, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()
        
def is_green_screen(frame):
    if frame is None: return False
    small = cv2.resize(frame, (64, 64))
    avg_color = np.mean(small, axis=(0, 1))
    return avg_color[1] > 200 and avg_color[0] < 50 and avg_color[2] < 50

def measure_global_change(f1, f2):
    # Resize to 350x250 (perfectly divisible by 7x5 grid)
    # This resolution is high enough to detect objects but small enough to fit in cache.
    w, h = 350, 250
    g1 = cv2.cvtColor(cv2.resize(f1, (w, h)), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(f2, (w, h)), cv2.COLOR_BGR2GRAY)
    
    # Compute difference once for the whole image
    diff_map = cv2.absdiff(g1, g2)
    
    uw, uh = w // 7, h // 5  # Unit width (50px) and height (50px)
    maes = []

    # Sample 6 areas at (row, col) indices: (0,0), (0,3), (0,6), (4,0), (4,3), (4,6)
    for r in [0, 4]:
        for c in [0, 3, 6]:
            roi_diff = diff_map[r*uh : (r+1)*uh, c*uw : (c+1)*uw]
            maes.append(np.mean(roi_diff))

    # Return minimum: Only trigger if ALL regions show significant change (Global Motion)
    return min(maes)
