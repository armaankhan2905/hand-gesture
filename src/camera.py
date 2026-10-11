import cv2
import time
from src.gesture_recognizer import GestureRecognizer

class GestureAppCamera:
    """
    Manages the webcam feed, frame capture, HUD overlays (gesture status, FPS), 
    and links to the gesture recognition engine.
    """
    def __init__(self, camera_index=0, width=1280, height=720):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # FPS calculation variables
        self.prev_time = 0
        self.fps_list = []
        self.max_fps_samples = 15

    def _draw_hud(self, frame, gesture, fps):
        """
        Draws a modern, semi-transparent HUD card with status and FPS.
        """
        h, w, _ = frame.shape
        
        # Define HUD card dimensions (Top-Left)
        card_x1, card_y1 = 20, 20
        card_x2, card_y2 = 380, 140
        
        # Create an overlay layer for transparency
        overlay = frame.copy()
        
        # Draw background card (dark grey / blue glassmorphism effect)
        cv2.rectangle(overlay, (card_x1, card_y1), (card_x2, card_y2), (30, 25, 24), -1)
        
        # Blend the overlay with the original frame (alpha blending)
        alpha = 0.75
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw a sleek neon border on the card
        border_color = (0, 230, 115) if gesture != "Unknown" else (100, 100, 100) # Green for active gesture, gray otherwise
        cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), border_color, 2)
        
        # Write text
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Title/Heading
        cv2.putText(frame, "HAND GESTURE RECOGNIZER", (card_x1 + 15, card_y1 + 30), 
                    font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        
        # Gesture Name
        gesture_text = f"Gesture: {gesture}"
        cv2.putText(frame, gesture_text, (card_x1 + 15, card_y1 + 70), 
                    font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # FPS status
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (card_x1 + 15, card_y1 + 105), 
                    font, 0.6, (200, 255, 200) if fps > 20 else (150, 150, 255), 1, cv2.LINE_AA)

    def run(self):
        """
        Runs the main loop: captures frames, processes gestures, and updates the display.
        """
        if not self.cap.isOpened():
            print(f"Error: Could not open video source at index {self.camera_index}.")
            return

        recognizer = GestureRecognizer()
        print("Application started. Press 'q' in the window to quit.")
        
        try:
            failed_frames_count = 0
            max_failed_frames = 30
            
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    failed_frames_count += 1
                    if failed_frames_count >= max_failed_frames:
                        print(f"Error: Failed to grab frame after {max_failed_frames} consecutive attempts.")
                        break
                    time.sleep(0.03)  # wait 30ms before retrying
                    continue
                
                # Reset counter on successful frame read
                failed_frames_count = 0
                
                # Flip the frame horizontally for a natural selfie-view mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert the BGR image to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame and find landmarks
                results = recognizer.process_frame(frame_rgb)
                
                # Draw hand landmarks on the frame
                recognizer.draw_landmarks(frame, results)
                
                # Recognize gesture if a hand is detected
                detected_gesture = "Unknown"
                if results.hand_landmarks and results.handedness:
                    # Detect only the first hand
                    hand_landmarks = results.hand_landmarks[0]
                    handedness_list = results.handedness[0]
                    if handedness_list:
                        cat = handedness_list[0]
                        handedness_label = getattr(cat, "category_name", getattr(cat, "categoryName", "Right"))
                        detected_gesture = recognizer.recognize_gesture(hand_landmarks, handedness_label)
                
                # Calculate FPS (smoothed using simple moving average)
                current_time = time.time()
                time_delta = current_time - self.prev_time
                self.prev_time = current_time
                
                if time_delta > 0:
                    current_fps = 1.0 / time_delta
                    self.fps_list.append(current_fps)
                    if len(self.fps_list) > self.max_fps_samples:
                        self.fps_list.pop(0)
                    avg_fps = sum(self.fps_list) / len(self.fps_list)
                else:
                    avg_fps = 0.0
                
                # Render HUD
                self._draw_hud(frame, detected_gesture, avg_fps)
                
                # Display the frame
                cv2.imshow("Hand Gesture Recognition", frame)
                
                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nApplication interrupted by user.")
        finally:
            # Clean up resources
            self.cap.release()
            cv2.destroyAllWindows()
            recognizer.close()
            print("Resources released cleanly. Exiting.")

if __name__ == "__main__":
    app = GestureAppCamera()
    app.run()
