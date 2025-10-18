"""
Main Application - Sign Language Recognition with Text-to-Speech
Integrates hand detection, gesture classification, and voice output
"""

import cv2
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from hand_detector import HandDetector
from gesture_recognizer import SimpleGestureRecognizer
from text_to_speech import TextToSpeech, SpeechBuffer
from utils.config import Config
from utils.helpers import FPSCounter, TextRenderer, UIComponents


class SignLanguageApp:
    """
    Main application class for Sign Language Recognition
    """
    
    def __init__(self):
        """Initialize the application"""
        print("Initializing Sign Language Recognition App...")
        
        # Load and validate configuration
        if not Config.validate():
            raise ValueError("Invalid configuration. Please check your .env file")
        
        Config.print_config()
        
        # Initialize components
        self.detector = HandDetector(
            max_hands=Config.MAX_HANDS,
            detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
        )
        
        # Use rule-based recognizer (no training needed!)
        self.recognizer = SimpleGestureRecognizer()
        
        # Initialize TTS (only if API key is available)
        self.tts = None
        self.speech_buffer = None
        if Config.OPENAI_API_KEY:
            try:
                self.tts = TextToSpeech(
                    api_key=Config.OPENAI_API_KEY,
                    model=Config.TTS_MODEL,
                    voice=Config.TTS_VOICE
                )
                self.speech_buffer = SpeechBuffer(self.tts)
                print("✓ Text-to-Speech initialized")
            except Exception as e:
                print(f"⚠ Warning: Could not initialize TTS: {e}")
        else:
            print("⚠ Warning: OpenAI API key not found. TTS disabled.")
        
        # Initialize camera - try multiple indices if first fails
        self.cap = None
        camera_indices = [Config.CAMERA_INDEX, 0, 1, 2]  # Try configured, then 0, 1, 2
        
        for idx in camera_indices:
            print(f"Trying camera index {idx}...")
            cap_test = cv2.VideoCapture(idx)
            if cap_test.isOpened():
                # Test if we can actually read a frame
                ret, frame = cap_test.read()
                if ret and frame is not None:
                    self.cap = cap_test
                    Config.CAMERA_INDEX = idx
                    print(f"✓ Successfully opened camera {idx}")
                    break
                else:
                    cap_test.release()
            else:
                cap_test.release()
        
        if self.cap is None or not self.cap.isOpened():
            print("\n✗ Error: Could not open any camera")
            print("Troubleshooting:")
            print("  1. Make sure you're running in Windows PowerShell (not WSL)")
            print("  2. Check if camera is in use by another application")
            print("  3. Try running: python src/hand_detector.py")
            raise RuntimeError("Could not open camera")
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        
        # UI components
        self.fps_counter = FPSCounter()
        self.window_name = Config.WINDOW_NAME
        
        # Application state
        self.running = False
        self.paused = False
        self.current_gesture = None
        self.current_confidence = 0.0
        self.accumulated_text = ""
        self.last_added_gesture = None  # Track last gesture to avoid duplicates
        self.gesture_frames_count = 0  # Count frames gesture is held
        self.frames_required = 30  # Frames to hold before adding (1 second at 30fps)
        self.auto_speak = True  # Auto speak when gesture detected
        self.last_spoken_gesture = None  # Track last spoken gesture
        
        print("✓ Application initialized successfully!\n")
    
    def draw_ui(self, img):
        """
        Draw user interface elements on image
        
        Args:
            img: Image to draw on
        """
        h, w = img.shape[:2]
        
        # Draw FPS if enabled
        if Config.SHOW_FPS:
            self.fps_counter.draw_fps(img, (10, 30))
        
        # Draw info panel
        info = {
            'Gesture': self.current_gesture or 'None',
            'Confidence': f'{self.current_confidence:.2f}',
            'Status': 'PAUSED' if self.paused else 'ACTIVE'
        }
        UIComponents.draw_info_panel(img, info, (10, 60), 250)
        
        # Draw accumulated text at bottom
        if self.speech_buffer:
            text = self.speech_buffer.get_text()
            if text:
                # Draw text box at bottom
                text_display = f"Text: {text}"
                TextRenderer.draw_text(
                    img, text_display,
                    (10, h - 80),
                    font_scale=0.8,
                    color=(255, 255, 255),
                    thickness=2,
                    bg_color=(0, 0, 0)
                )
        
        # Draw instructions
        instructions = [
            "Controls:",
            "SPACE - Add space",
            "ENTER - Speak text",
            "BACKSPACE - Delete last",
            "C - Clear all",
            "P - Pause/Resume",
            "Q - Quit"
        ]
        
        y_start = h - 250
        for i, instruction in enumerate(instructions):
            TextRenderer.draw_text(
                img, instruction,
                (w - 250, y_start + i * 25),
                font_scale=0.5,
                color=(200, 200, 200),
                thickness=1
            )
    
    def _gesture_to_speech(self, gesture: str) -> str:
        """
        Convert gesture name to speech-friendly text
        
        Args:
            gesture: Gesture name (e.g., "A", "THUMBS_UP", "PEACE")
            
        Returns:
            Speech text
        """
        # Single letter gestures - spell phonetically
        if len(gesture) == 1 and gesture.isalpha():
            return f"Letter {gesture}"
        
        # Replace underscores with spaces and capitalize
        return gesture.replace("_", " ").title()
    
    def process_frame(self, img):
        """
        Process a single frame
        
        Args:
            img: Input frame
            
        Returns:
            Processed frame
        """
        # Detect hands
        img = self.detector.find_hands(img, draw=True)
        
        if not self.paused:
            # Get hand landmarks
            landmarks = self.detector.find_position(img, draw=False)
            
            if len(landmarks) > 0:
                # Recognize gesture (rule-based, no ML needed!)
                gesture, confidence = self.recognizer.recognize(landmarks)
                
                # Update state
                self.current_gesture = gesture
                self.current_confidence = confidence
                
                # Auto-speak gesture name if confidence is high enough
                if gesture and confidence >= Config.GESTURE_CONFIDENCE_THRESHOLD:
                    # Check if this is the same gesture
                    if gesture == self.last_added_gesture:
                        # Same gesture - count frames
                        self.gesture_frames_count += 1
                    else:
                        # New gesture detected!
                        if self.last_added_gesture and self.gesture_frames_count >= self.frames_required:
                            # Previous gesture held long enough - speak it!
                            if self.tts and self.auto_speak:
                                # Speak the gesture name
                                speech_text = self._gesture_to_speech(self.last_added_gesture)
                                print(f"🔊 Speaking: {speech_text}")
                                self.tts.text_to_speech(speech_text, save_file=False)
                            
                            # Also add letter to buffer if it's a letter
                            if len(self.last_added_gesture) == 1 and self.last_added_gesture.isalpha() and self.speech_buffer:
                                self.speech_buffer.add_character(self.last_added_gesture)
                        
                        # Reset for new gesture
                        self.last_added_gesture = gesture
                        self.gesture_frames_count = 1
            else:
                # No hand detected - speak last gesture if held long enough
                if self.last_added_gesture and self.gesture_frames_count >= self.frames_required:
                    if self.tts and self.auto_speak:
                        speech_text = self._gesture_to_speech(self.last_added_gesture)
                        print(f"🔊 Speaking: {speech_text}")
                        self.tts.text_to_speech(speech_text, save_file=False)
                    
                    # Also add letter to buffer
                    if len(self.last_added_gesture) == 1 and self.last_added_gesture.isalpha() and self.speech_buffer:
                        self.speech_buffer.add_character(self.last_added_gesture)
                
                # Reset state
                self.current_gesture = None
                self.current_confidence = 0.0
                self.last_added_gesture = None
                self.gesture_frames_count = 0
        
        return img
    
    def handle_keyboard(self, key):
        """
        Handle keyboard input
        
        Args:
            key: Key code
        """
        if self.speech_buffer is None:
            return
        
        # Space - add space/complete word
        if key == 32:
            self.speech_buffer.add_space()
        
        # Enter - speak accumulated text
        elif key == 13:
            self.speech_buffer.speak_current()
        
        # Backspace - delete last character
        elif key == 8:
            self.speech_buffer.delete_last_character()
        
        # C - clear all
        elif key == ord('c') or key == ord('C'):
            self.speech_buffer.clear()
        
        # P - pause/resume
        elif key == ord('p') or key == ord('P'):
            self.paused = not self.paused
            print(f"{'PAUSED' if self.paused else 'RESUMED'}")
        
        # Add detected letter to text
        if self.current_gesture and len(self.current_gesture) == 1:
            if self.current_confidence >= Config.GESTURE_CONFIDENCE_THRESHOLD:
                # Simple debouncing: only add if different from last added
                # This is a basic implementation
                pass
    
    def run(self):
        """Main application loop"""
        self.running = True
        
        print("\n" + "="*60)
        print("SIGN LANGUAGE RECOGNITION STARTED")
        print("="*60)
        print("\nInstructions:")
        print("  - Hold your hand in front of the camera")
        print("  - Make sign language gestures")
        print("  - Press SPACE to add space between words")
        print("  - Press ENTER to speak the accumulated text")
        print("  - Press C to clear text")
        print("  - Press P to pause/resume")
        print("  - Press Q to quit")
        print("="*60 + "\n")
        
        try:
            while self.running:
                # Read frame
                success, img = self.cap.read()
                if not success:
                    print("Failed to read from camera")
                    break
                
                # Process frame
                img = self.process_frame(img)
                
                # Draw UI
                self.draw_ui(img)
                
                # Update FPS
                self.fps_counter.update()
                
                # Display
                cv2.imshow(self.window_name, img)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\nQuitting...")
                    break
                elif key != 255:  # Key was pressed
                    self.handle_keyboard(key)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        
        self.running = False
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if self.detector is not None:
            self.detector.close()
        
        if self.tts is not None:
            self.tts.cleanup()
        
        print("✓ Cleanup complete")
        print("\nThank you for using Sign Language Recognition!")


def main():
    """Main entry point"""
    try:
        app = SignLanguageApp()
        app.run()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
