"""
Hand Detector Module using MediaPipe
Detects and tracks hand landmarks in real-time video stream
Simple and efficient for sign language recognition
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional


class HandDetector:
    """
    Hand detection and tracking using MediaPipe Hands solution
    """
    
    def __init__(
        self,
        mode: bool = False,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5
    ):
        """
        Initialize HandDetector
        
        Args:
            mode: Static image mode if True, video stream if False
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum confidence for detection
            tracking_confidence: Minimum confidence for tracking
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def find_hands(self, img: np.ndarray, draw: bool = True) -> np.ndarray:
        """
        Detect hands in image and optionally draw landmarks
        
        Args:
            img: Input image (BGR format)
            draw: Whether to draw landmarks on image
            
        Returns:
            Image with landmarks drawn (if draw=True)
        """
        # Convert BGR to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image
        self.results = self.hands.process(img_rgb)
        
        # Draw hand landmarks
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return img
    
    def find_position(
        self, 
        img: np.ndarray, 
        hand_no: int = 0,
        draw: bool = True
    ) -> List[Tuple[int, int, int]]:
        """
        Get positions of all hand landmarks
        
        Args:
            img: Input image
            hand_no: Hand index (0 for first hand, 1 for second)
            draw: Whether to draw circles on landmarks
            
        Returns:
            List of (id, x, y) tuples for each landmark
        """
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_no]
                
                h, w, c = img.shape
                
                for id, landmark in enumerate(hand.landmark):
                    # Convert normalized coordinates to pixel coordinates
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmark_list.append((id, cx, cy))
                    
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        return landmark_list
    
    def get_hand_label(self, hand_no: int = 0) -> Optional[str]:
        """
        Get hand label (Left or Right)
        
        Args:
            hand_no: Hand index
            
        Returns:
            'Left' or 'Right' or None
        """
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_handedness):
                return self.results.multi_handedness[hand_no].classification[0].label
        return None
    
    def fingers_up(self, landmark_list: List[Tuple[int, int, int]]) -> List[int]:
        """
        Detect which fingers are up
        
        Args:
            landmark_list: List of landmarks from find_position()
            
        Returns:
            List of 5 integers (0 or 1) for [thumb, index, middle, ring, pinky]
        """
        if len(landmark_list) == 0:
            return []
        
        fingers = []
        
        # Thumb (check if tip is to the right of IP joint for right hand)
        if landmark_list[4][1] > landmark_list[3][1]:  # Simplified check
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other 4 fingers
        tip_ids = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        pip_ids = [6, 10, 14, 18]  # PIP joints
        
        for tip, pip in zip(tip_ids, pip_ids):
            if landmark_list[tip][2] < landmark_list[pip][2]:  # Tip is above PIP
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def get_bounding_box(self, landmark_list: List[Tuple[int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box around hand
        
        Args:
            landmark_list: List of landmarks
            
        Returns:
            (x, y, w, h) of bounding box or None
        """
        if len(landmark_list) == 0:
            return None
        
        x_list = [lm[1] for lm in landmark_list]
        y_list = [lm[2] for lm in landmark_list]
        
        xmin, xmax = min(x_list), max(x_list)
        ymin, ymax = min(y_list), max(y_list)
        
        bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
        
        return bbox
    
    def close(self):
        """Clean up resources"""
        self.hands.close()


def main():
    """
    Demo function to test HandDetector
    """
    # Initialize webcam - try multiple indices
    cap = None
    for idx in [0, 1, 2]:
        print(f"Trying camera index {idx}...")
        cap_test = cv2.VideoCapture(idx)
        if cap_test.isOpened():
            ret, frame = cap_test.read()
            if ret and frame is not None:
                cap = cap_test
                print(f"✓ Camera {idx} opened successfully!")
                break
            else:
                cap_test.release()
        else:
            cap_test.release()
    
    if cap is None or not cap.isOpened():
        print("\n✗ Error: Could not open any camera")
        print("Make sure you're running in Windows PowerShell (not WSL)")
        return
    
    # Set resolution - try lower resolution first
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Verify settings
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera resolution: {int(actual_width)}x{int(actual_height)}")
    
    # Initialize detector
    detector = HandDetector(max_hands=2, detection_confidence=0.7)
    
    print("Hand Detector Demo")
    print("Press 'q' to quit")
    
    # Wait a bit for camera to warm up
    import time
    time.sleep(1.0)
    
    while True:
        success, img = cap.read()
        if not success or img is None:
            print("Failed to read frame, retrying...")
            time.sleep(0.1)
            continue
        
        # Detect hands
        img = detector.find_hands(img)
        
        # Get landmarks
        landmark_list = detector.find_position(img, draw=False)
        
        if len(landmark_list) != 0:
            # Get hand label
            hand_label = detector.get_hand_label()
            
            # Get fingers up
            fingers = detector.fingers_up(landmark_list)
            
            # Get bounding box
            bbox = detector.get_bounding_box(landmark_list)
            
            # Display info
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(img, (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 2)
            
            # Display hand info
            cv2.putText(img, f"Hand: {hand_label}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(img, f"Fingers up: {sum(fingers)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Display
        cv2.imshow("Hand Detector Demo", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
