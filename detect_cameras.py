import cv2

def detect_cameras(max_tested=5):
    print("==========================================")
    print("        CAMERA DIAGNOSTIC TOOL            ")
    print("==========================================")
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera index {i}: SUCCESS (Opened & grabbed frame: {frame.shape[1]}x{frame.shape[0]})")
            else:
                print(f"Camera index {i}: PARTIAL (Opened, but FAILED to grab frame. Likely in use by another app.)")
            cap.release()
        else:
            print(f"Camera index {i}: CLOSED (Could not open/device not found.)")
    print("==========================================")

if __name__ == "__main__":
    detect_cameras()
