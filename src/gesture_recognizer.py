import os
import math
import urllib.request
import cv2
import mediapipe as mp

def download_model_if_missing(model_path="hand_landmarker.task"):
    """
    Downloads the official MediaPipe Hand Landmarker model file if it does not exist locally.
    """
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Downloading from Google Edge APIs...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        try:
            # Show a simple download status
            def report_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, int(downloaded * 100 / total_size))
                print(f"\rDownloading: {percent}% ({downloaded / (1024*1024):.2f}MB / {total_size / (1024*1024):.2f}MB)", end="")
            
            urllib.request.urlretrieve(url, model_path, reporthook=report_hook)
            print("\nDownload completed successfully!")
        except Exception as e:
            print(f"\nError downloading model file: {e}")
            raise e

class GestureRecognizer:
    """
    Encapsulates MediaPipe Tasks HandLandmarker initialization and gesture recognition logic.
    Identifies gestures: Open Palm, Fist, Thumbs Up, Peace Sign, and OK Sign.
    """
    def __init__(self, model_path="hand_landmarker.task"):
        # Make sure model is downloaded
        download_model_if_missing(model_path)
        
        # Configure MediaPipe Task HandLandmarker
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame_rgb):
        """
        Processes a single RGB frame to detect hands and extract landmarks.
        """
        # Convert RGB numpy frame to mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb.copy())
        return self.detector.detect(mp_image)

    def draw_landmarks(self, frame_bgr, results):
        """
        Draws custom hand landmarks and connection lines on the BGR frame.
        """
        if not results.hand_landmarks:
            return
            
        h, w, _ = frame_bgr.shape
        
        # 21 Hand Landmarks connected in a specific sequence
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (9, 10), (10, 11), (11, 12),         # Middle
            (13, 14), (14, 15), (15, 16),        # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17)            # Palm base joins
        ]
        
        for hand_landmarks in results.hand_landmarks:
            # 1. Draw connection lines in sleek green
            for start_idx, end_idx in connections:
                lm_start = hand_landmarks[start_idx]
                lm_end = hand_landmarks[end_idx]
                
                pt1 = (int(lm_start.x * w), int(lm_start.y * h))
                pt2 = (int(lm_end.x * w), int(lm_end.y * h))
                
                cv2.line(frame_bgr, pt1, pt2, (0, 220, 100), 2, cv2.LINE_AA)
            
            # 2. Draw joint dots (red for tips, orange for joints)
            for idx, lm in enumerate(hand_landmarks):
                pt = (int(lm.x * w), int(lm.y * h))
                if idx in [4, 8, 12, 16, 20]: # Tips
                    cv2.circle(frame_bgr, pt, 6, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame_bgr, pt, 8, (255, 255, 255), 1, cv2.LINE_AA)
                else: # Joints
                    cv2.circle(frame_bgr, pt, 4, (255, 150, 0), -1, cv2.LINE_AA)
                    cv2.circle(frame_bgr, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)

    def _get_distance(self, p1, p2):
        """
        Calculates the 2D Euclidean distance between two landmarks.
        """
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def recognize_gesture(self, hand_landmarks, handedness_label):
        """
        Recognizes the hand gesture based on landmark positions.
        Returns:
            str: Name of the recognized gesture ("Open Palm", "Fist", "Thumbs Up", "Peace Sign", "OK Sign", "Unknown")
        """
        # hand_landmarks is a list of 21 NormalizedLandmark objects
        landmarks = hand_landmarks

        # Use Wrist (0) to Middle Finger MCP (9) distance as the hand scale normalization factor
        hand_scale = self._get_distance(landmarks[0], landmarks[9])
        if hand_scale == 0:
            return "Unknown"

        # Check if four main fingers are extended (TIP y-coordinate is smaller than PIP y-coordinate)
        is_index_extended = landmarks[8].y < landmarks[6].y
        is_middle_extended = landmarks[12].y < landmarks[10].y
        is_ring_extended = landmarks[16].y < landmarks[14].y
        is_pinky_extended = landmarks[20].y < landmarks[18].y

        # Check if thumb is extended
        # For thumb, we use the distance between Thumb Tip (4) and Index MCP (5) normalized by hand_scale
        thumb_index_mcp_dist = self._get_distance(landmarks[4], landmarks[5]) / hand_scale
        is_thumb_extended = thumb_index_mcp_dist > 0.4

        # Distance between Thumb Tip (4) and Index Tip (8) normalized
        thumb_index_tip_dist = self._get_distance(landmarks[4], landmarks[8]) / hand_scale

        # Gesture Recognition Logic
        
        # 1. OK Sign: Thumb and Index tips touch, other fingers are extended
        if thumb_index_tip_dist < 0.25 and is_middle_extended and is_ring_extended and is_pinky_extended:
            return "OK Sign"

        # 2. Open Palm: All fingers (including thumb) extended
        if is_index_extended and is_middle_extended and is_ring_extended and is_pinky_extended and is_thumb_extended:
            return "Open Palm"

        # 3. Fist: All main fingers folded, and thumb folded or close to fingers
        if not is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended:
            if thumb_index_mcp_dist < 0.4:
                return "Fist"

        # 4. Thumbs Up: Thumb extended upward, other fingers folded
        if is_thumb_extended and not is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended:
            # Ensure thumb is pointing upwards relative to the wrist and IP joint
            if landmarks[4].y < landmarks[3].y and landmarks[4].y < landmarks[5].y:
                return "Thumbs Up"

        # 5. Peace Sign: Index and Middle extended, Ring and Pinky folded
        if is_index_extended and is_middle_extended and not is_ring_extended and not is_pinky_extended:
            return "Peace Sign"

        return "Unknown"

    def close(self):
        """
        Releases detector resources.
        """
        self.detector.close()
