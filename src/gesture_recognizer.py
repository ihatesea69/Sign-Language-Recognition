"""
Simple Rule-Based Gesture Recognition
Using MediaPipe hand landmarks without ML training
"""

import numpy as np
from typing import List, Tuple, Optional
import math


class SimpleGestureRecognizer:
    """
    Rule-based gesture recognition using hand landmarks
    No training data or ML model required
    
    Recognizes basic hand shapes:
    - Numbers: 0-9
    - Letters: A, B, C, L, O, V, W, Y (easy static gestures)
    - Commands: thumbs_up, thumbs_down, peace, ok, fist
    """
    
    def __init__(self):
        """Initialize gesture recognizer"""
        self.gesture_history = []
        self.history_size = 5
        
    def recognize(self, landmarks: List[Tuple[float, float]]) -> Tuple[str, float]:
        """
        Recognize gesture from hand landmarks
        
        Args:
            landmarks: List of 21 (x, y) coordinates
            
        Returns:
            (gesture_name, confidence)
        """
        if not landmarks or len(landmarks) != 21:
            return ("NONE", 0.0)
        
        # Extract features
        fingers_up = self._get_fingers_up(landmarks)
        thumb_up = fingers_up[0]
        index_up = fingers_up[1]
        middle_up = fingers_up[2]
        ring_up = fingers_up[3]
        pinky_up = fingers_up[4]
        
        fingers_up_count = sum(fingers_up)
        
        # Get angles and distances (normalized by palm size for robustness)
        thumb_index_distance = self._get_distance(landmarks[4], landmarks[8])
        # Use wrist (0) to middle MCP (9) as palm scale reference
        palm_scale = max(self._get_distance(landmarks[0], landmarks[9]), 1e-6)
        thumb_index_touching = (thumb_index_distance / palm_scale) < 0.35

        # Angle-based extension checks (more robust than simple y-threshold)
        index_ext = self._is_finger_extended(landmarks, mcp=5, pip=6, tip=8)
        middle_ext = self._is_finger_extended(landmarks, mcp=9, pip=10, tip=12)
        ring_ext = self._is_finger_extended(landmarks, mcp=13, pip=14, tip=16)
        pinky_ext = self._is_finger_extended(landmarks, mcp=17, pip=18, tip=20)

        # Separation between index and middle tips (for V sign)
        v_sep = self._get_distance(landmarks[8], landmarks[12]) / palm_scale
        
        # Recognize gestures
        gesture, confidence = self._match_gesture(
            fingers_up, fingers_up_count, thumb_index_touching, landmarks,
            palm_scale, index_ext, middle_ext, ring_ext, pinky_ext, v_sep
        )
        
        # Add to history for smoothing
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
        
        # Return most common gesture in history
        if self.gesture_history:
            most_common = max(set(self.gesture_history), key=self.gesture_history.count)
            return (most_common, confidence)
        
        return (gesture, confidence)
    
    def _match_gesture(self, fingers_up, fingers_up_count, thumb_index_touching, landmarks,
                       palm_scale: float,
                       index_ext: bool, middle_ext: bool, ring_ext: bool, pinky_ext: bool,
                       v_sep: float):
        """Match gesture based on finger states"""
        thumb_up, index_up, middle_up, ring_up, pinky_up = fingers_up
        
        # PRIORITY: OK sign (thumb and index forming a circle) should win early
        # This avoids misclassifying as Y (thumb+pinky up) or O
        if thumb_index_touching:
            # If other three fingers mostly up, it's the classic OK; else still accept OK
            confidence = 0.9 if (middle_up or ring_up or pinky_up) else 0.85
            return ("OK", confidence)

        # FIVE / OPEN HAND - All fingers extended with adequate separation
        if index_ext and middle_ext and ring_ext and pinky_ext and (thumb_up or not thumb_up):
            sep_im = self._get_distance(landmarks[8], landmarks[12]) / palm_scale
            sep_mr = self._get_distance(landmarks[12], landmarks[16]) / palm_scale
            sep_rp = self._get_distance(landmarks[16], landmarks[20]) / palm_scale
            if min(sep_im, sep_mr, sep_rp) > 0.2:
                return ("FIVE", 0.9)

        # V / PEACE (two fingers up and clearly separated)
        ring_bent = self._is_finger_bent(landmarks, mcp=13, pip=14, tip=16)
        pinky_bent = self._is_finger_bent(landmarks, mcp=17, pip=18, tip=20)
        if index_ext and middle_ext and v_sep > 0.30:
            # Allow slight extension on ring/pinky, but prefer when they are bent
            if (not ring_ext or ring_bent) and (not pinky_ext or pinky_bent):
                return ("PEACE" if thumb_up else "V", 0.9)

        # L shape: thumb + index extended, remaining fingers bent, angle ~ 90°
        if (index_ext and thumb_up) and (not middle_ext and not ring_ext and not pinky_ext):
            angle = self._get_thumb_index_angle(landmarks)
            if 70 < angle < 110:
                return ("L", 0.9)

        # FIST - All fingers down
        if fingers_up_count == 0:
            return ("FIST", 0.95)
        
        # THUMBS UP - Only thumb up
        if thumb_up and fingers_up_count == 1:
            return ("THUMBS_UP", 0.9)
        
        # (Handled above) PEACE / V
        
        # THREE - Index, middle, ring up
        if index_up and middle_up and ring_up and not pinky_up:
            return ("THREE", 0.85)
        
        # (Handled above) FIVE
        
        # ONE / POINTING - Only index extended
        if index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_up:
            return ("ONE", 0.85)
        
        # (Handled above) OK SIGN
        
        # TWO - index and middle extended (fallback when separation small)
        if index_ext and middle_ext and v_sep <= 0.30:
            return ("TWO", 0.8)
        
        # ROCK (ROCK AND ROLL) - Index and pinky up
        if index_up and pinky_up and not middle_up and not ring_up:
            return ("ROCK", 0.85)
        
        # W - Three middle fingers up
        if index_ext and middle_ext and ring_ext and not thumb_up and not pinky_ext:
            return ("W", 0.8)
        
        # Y - Thumb and pinky up
        if thumb_up and pinky_ext and not index_ext and not middle_ext and not ring_ext:
            return ("Y", 0.85)
        
        # A - Fist with thumb to the side
        if not index_up and not middle_up and not ring_up and not pinky_up:
            thumb_pos = landmarks[4][1]  # Thumb tip y position
            if thumb_pos < landmarks[2][1]:  # Thumb higher than thumb IP
                return ("A", 0.75)
        
        # B - Four fingers extended, thumb folded across palm, roughly parallel fingers
        if index_ext and middle_ext and ring_ext and pinky_ext and not thumb_up:
            dirs = [
                self._finger_dir(landmarks, 5, 8),
                self._finger_dir(landmarks, 9, 12),
                self._finger_dir(landmarks, 13, 16),
                self._finger_dir(landmarks, 17, 20),
            ]
            max_pair_angle = self._max_pairwise_angle(dirs)
            if max_pair_angle < 25:
                return ("B", 0.88)
        
        # C - Curved hand (check curvature of fingertips toward palm center)
        if (index_ext or middle_ext or ring_ext or pinky_ext) and self._is_hand_curved(landmarks):
            return ("C", 0.7)
        
   
        return (f"{fingers_up_count}_FINGERS", 0.5)

    def _finger_dir(self, lm, mcp_idx: int, tip_idx: int) -> Tuple[float, float]:
        v = (lm[tip_idx][0] - lm[mcp_idx][0], lm[tip_idx][1] - lm[mcp_idx][1])
        n = math.hypot(v[0], v[1])
        if n == 0:
            return (0.0, -1.0)
        return (v[0]/n, v[1]/n)

    def _angle_at_joint(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        v1 = (a[0]-b[0], a[1]-b[1])
        v2 = (c[0]-b[0], c[1]-b[1])
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 == 0 or n2 == 0:
            return 180.0
        cosang = (v1[0]*v2[0] + v1[1]*v2[1]) / (n1*n2)
        cosang = max(-1.0, min(1.0, cosang))
        return math.degrees(math.acos(cosang))

    def _is_finger_extended(self, landmarks: List[Tuple[float, float]], mcp: int, pip: int, tip: int) -> bool:
        angle = self._angle_at_joint(landmarks[mcp], landmarks[pip], landmarks[tip])
        return angle > 160.0

    def _is_finger_bent(self, landmarks: List[Tuple[float, float]], mcp: int, pip: int, tip: int) -> bool:
        angle = self._angle_at_joint(landmarks[mcp], landmarks[pip], landmarks[tip])
        return angle < 140.0

    def _max_pairwise_angle(self, dirs: List[Tuple[float, float]]) -> float:
        m = 0.0
        for i in range(len(dirs)):
            for j in range(i+1, len(dirs)):
                d = dirs[i][0]*dirs[j][0] + dirs[i][1]*dirs[j][1]
                d = max(-1.0, min(1.0, d))
                ang = math.degrees(math.acos(d))
                if ang > m:
                    m = ang
        return m
    
    def _get_fingers_up(self, landmarks: List[Tuple[float, float]]) -> List[bool]:
        """
        Determine which fingers are up
        Returns: [thumb, index, middle, ring, pinky]
        """
        fingers = []
        
        # Thumb - check if tip is to the right of IP joint (for right hand)
        # This is simplified - works best when hand is facing camera
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        # Check horizontal distance for thumb
        thumb_up = abs(thumb_tip[0] - thumb_mcp[0]) > abs(thumb_ip[0] - thumb_mcp[0])
        fingers.append(thumb_up)
        
        # Other fingers - check if tip is above PIP joint
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        finger_pips = [6, 10, 14, 18]  # Corresponding PIP joints
        
        for tip, pip in zip(finger_tips, finger_pips):
            # Finger is up if tip is higher (lower y value) than PIP
            fingers.append(landmarks[tip][1] < landmarks[pip][1])
        
        return fingers
    
    def _get_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _get_thumb_index_angle(self, landmarks: List[Tuple[float, float]]) -> float:
        """Calculate angle between thumb and index finger"""
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Vectors from wrist
        v1 = (thumb_tip[0] - wrist[0], thumb_tip[1] - wrist[1])
        v2 = (index_tip[0] - wrist[0], index_tip[1] - wrist[1])
        
        # Calculate angle
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
        
        angle = math.degrees(math.acos(cos_angle))
        return angle
    
    def _is_hand_curved(self, landmarks: List[Tuple[float, float]]) -> bool:
        """Check if hand is in curved position (for C gesture)"""
        # Check if fingertips are curved inward
        finger_tips = [8, 12, 16, 20]
        palm_center_x = landmarks[0][0]  # Wrist x position
        
        curved_count = 0
        for tip_idx in finger_tips:
            # If fingertip is closer to palm center than expected, it's curved
            tip_x = landmarks[tip_idx][0]
            pip_x = landmarks[tip_idx - 2][0]
            
            # Check if tip is bent toward palm
            if abs(tip_x - palm_center_x) < abs(pip_x - palm_center_x):
                curved_count += 1
        
        return curved_count >= 3
    
    def reset_history(self):
        """Reset gesture history"""
        self.gesture_history = []
    
    def get_supported_gestures(self) -> List[str]:
        """Get list of supported gestures"""
        return [
            # Numbers
            "FIST", "ONE", "TWO", "THREE", "FOUR", "FIVE",
            
            # Letters
            "A", "B", "C", "L", "O", "V", "W", "Y",
            
            # Commands
            "THUMBS_UP", "PEACE", "OK", "ROCK",
            
            # Finger counts
            "0_FINGERS", "1_FINGERS", "2_FINGERS", "3_FINGERS", "4_FINGERS", "5_FINGERS"
        ]


# Demo code
if __name__ == "__main__":
    print("Simple Gesture Recognizer Demo")
    print("=" * 50)
    
    recognizer = SimpleGestureRecognizer()
    
    print("\nSupported Gestures:")
    for gesture in recognizer.get_supported_gestures():
        print(f"  - {gesture}")
    
    print("\nThis recognizer works with MediaPipe hand landmarks")
    print("No training data or ML model required!")
    print("\nTo use:")
    print("  1. Get 21 hand landmarks from MediaPipe")
    print("  2. Pass them to recognizer.recognize(landmarks)")
    print("  3. Get gesture name and confidence")
