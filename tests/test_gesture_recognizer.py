import unittest
import math
from src.gesture_recognizer import GestureRecognizer

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class TestGestureRecognizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the recognizer once for all tests
        cls.recognizer = GestureRecognizer()

    def _create_base_landmarks(self):
        # Create a base template for 21 landmarks representing an upright open hand
        # Index: MCP = 5, PIP = 6, DIP = 7, TIP = 8
        # Middle: MCP = 9, PIP = 10, DIP = 11, TIP = 12
        # Ring: MCP = 13, PIP = 14, DIP = 15, TIP = 16
        # Pinky: MCP = 17, PIP = 18, DIP = 19, TIP = 20
        landmarks = [(0.5, 0.9, 0.0)] * 21  # Default everything to wrist pos initially
        
        # Wrist
        landmarks[0] = (0.5, 0.9, 0.0)
        
        # Thumb: 1, 2, 3, 4
        landmarks[1] = (0.45, 0.85, 0.0)
        landmarks[2] = (0.4, 0.8, 0.0)
        landmarks[3] = (0.35, 0.75, 0.0)
        landmarks[4] = (0.3, 0.7, 0.0)
        
        # Index
        landmarks[5] = (0.45, 0.7, 0.0)
        landmarks[6] = (0.45, 0.6, 0.0)
        landmarks[7] = (0.45, 0.5, 0.0)
        landmarks[8] = (0.45, 0.4, 0.0)
        
        # Middle
        landmarks[9] = (0.5, 0.68, 0.0)
        landmarks[10] = (0.5, 0.58, 0.0)
        landmarks[11] = (0.5, 0.48, 0.0)
        landmarks[12] = (0.5, 0.38, 0.0)
        
        # Ring
        landmarks[13] = (0.55, 0.7, 0.0)
        landmarks[14] = (0.55, 0.6, 0.0)
        landmarks[15] = (0.55, 0.5, 0.0)
        landmarks[16] = (0.55, 0.4, 0.0)
        
        # Pinky
        landmarks[17] = (0.6, 0.73, 0.0)
        landmarks[18] = (0.6, 0.65, 0.0)
        landmarks[19] = (0.6, 0.58, 0.0)
        landmarks[20] = (0.6, 0.5, 0.0)
        
        return landmarks

    def _to_mock_landmarks(self, landmarks_list):
        return [MockLandmark(x, y, z) for x, y, z in landmarks_list]

    def test_open_palm(self):
        landmarks_list = self._create_base_landmarks()
        mock_landmarks = self._to_mock_landmarks(landmarks_list)
        gesture = self.recognizer.recognize_gesture(mock_landmarks, "Right")
        self.assertEqual(gesture, "Open Palm")

    def test_fist(self):
        landmarks_list = self._create_base_landmarks()
        
        # Make all fingers folded by raising the tip y coordinates (lower down on screen)
        # Index
        landmarks_list[8] = (0.45, 0.75, 0.0) # below PIP (6) at 0.6
        # Middle
        landmarks_list[12] = (0.5, 0.73, 0.0) # below PIP (10) at 0.58
        # Ring
        landmarks_list[16] = (0.55, 0.75, 0.0) # below PIP (14) at 0.6
        # Pinky
        landmarks_list[20] = (0.6, 0.78, 0.0) # below PIP (18) at 0.65
        
        # Fold thumb: move TIP (4) close to index MCP (5)
        landmarks_list[4] = (0.44, 0.71, 0.0)
        
        mock_landmarks = self._to_mock_landmarks(landmarks_list)
        gesture = self.recognizer.recognize_gesture(mock_landmarks, "Right")
        self.assertEqual(gesture, "Fist")

    def test_thumbs_up(self):
        landmarks_list = self._create_base_landmarks()
        
        # Fold fingers:
        landmarks_list[8] = (0.45, 0.75, 0.0)
        landmarks_list[12] = (0.5, 0.73, 0.0)
        landmarks_list[16] = (0.55, 0.75, 0.0)
        landmarks_list[20] = (0.6, 0.78, 0.0)
        
        # Extend thumb upwards
        landmarks_list[3] = (0.35, 0.5, 0.0)
        landmarks_list[4] = (0.3, 0.4, 0.0) # y=0.4 is above 3 (0.5), 5 (0.7), and 17 (0.73)
        
        mock_landmarks = self._to_mock_landmarks(landmarks_list)
        gesture = self.recognizer.recognize_gesture(mock_landmarks, "Right")
        self.assertEqual(gesture, "Thumbs Up")

    def test_peace_sign(self):
        landmarks_list = self._create_base_landmarks()
        
        # Fold Ring and Pinky:
        landmarks_list[16] = (0.55, 0.75, 0.0)
        landmarks_list[20] = (0.6, 0.78, 0.0)
        
        # Extend Index and Middle (already extended in template)
        # Fold/close thumb
        landmarks_list[4] = (0.44, 0.71, 0.0)
        
        mock_landmarks = self._to_mock_landmarks(landmarks_list)
        gesture = self.recognizer.recognize_gesture(mock_landmarks, "Right")
        self.assertEqual(gesture, "Peace Sign")

    def test_ok_sign(self):
        landmarks_list = self._create_base_landmarks()
        
        # Extend Middle, Ring, Pinky (already extended in template)
        # Touch Thumb TIP (4) and Index TIP (8)
        landmarks_list[4] = (0.4, 0.45, 0.0)
        landmarks_list[8] = (0.4, 0.45, 0.0)
        
        mock_landmarks = self._to_mock_landmarks(landmarks_list)
        gesture = self.recognizer.recognize_gesture(mock_landmarks, "Right")
        self.assertEqual(gesture, "OK Sign")

if __name__ == "__main__":
    unittest.main()
