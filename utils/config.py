"""
Configuration Management
Loads and manages application settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any


# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    """Application configuration"""
    
    # OpenAI Settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    # Camera Settings
    CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
    CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '1280'))
    CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '720'))
    
    # MediaPipe Settings
    MIN_DETECTION_CONFIDENCE = float(os.getenv('MIN_DETECTION_CONFIDENCE', '0.7'))
    MIN_TRACKING_CONFIDENCE = float(os.getenv('MIN_TRACKING_CONFIDENCE', '0.5'))
    MAX_HANDS = 1  # Support single hand for now
    
    # Gesture Recognition Settings
    GESTURE_CONFIDENCE_THRESHOLD = float(os.getenv('GESTURE_CONFIDENCE_THRESHOLD', '0.8'))
    BUFFER_SIZE = int(os.getenv('BUFFER_SIZE', '30'))
    
    # Text-to-Speech Settings
    TTS_MODEL = os.getenv('TTS_MODEL', 'tts-1')
    TTS_VOICE = os.getenv('TTS_VOICE', 'alloy')
    TTS_LANGUAGE = os.getenv('TTS_LANGUAGE', 'en')
    
    # Application Settings
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    SHOW_FPS = os.getenv('SHOW_FPS', 'True').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    USE_TFLITE_PIPELINE = os.getenv('USE_TFLITE_PIPELINE', 'False').lower() == 'true'
    ENABLE_GESTURE_DATA_LOGGING = os.getenv('ENABLE_GESTURE_DATA_LOGGING', 'False').lower() == 'true'
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / 'models'
    DATA_DIR = BASE_DIR / 'data'
    OUTPUT_DIR = BASE_DIR / 'output'
    NOTEBOOKS_DIR = BASE_DIR / 'notebooks'
    DOCS_DIR = BASE_DIR / 'docs'
    
    # Gesture assets
    GESTURE_ASSETS_DIR = MODELS_DIR / 'gesture'
    GESTURE_KEYPOINT_DIR = GESTURE_ASSETS_DIR / 'keypoint_classifier'
    GESTURE_POINT_HISTORY_DIR = GESTURE_ASSETS_DIR / 'point_history_classifier'
    GESTURE_KEYPOINT_TFLITE = Path(os.getenv(
        'GESTURE_KEYPOINT_TFLITE',
        GESTURE_KEYPOINT_DIR / 'keypoint_classifier.tflite'
    ))
    GESTURE_POINT_HISTORY_TFLITE = Path(os.getenv(
        'GESTURE_POINT_HISTORY_TFLITE',
        GESTURE_POINT_HISTORY_DIR / 'point_history_classifier.tflite'
    ))
    GESTURE_KEYPOINT_LABELS = Path(os.getenv(
        'GESTURE_KEYPOINT_LABELS',
        GESTURE_KEYPOINT_DIR / 'keypoint_classifier_label.csv'
    ))
    GESTURE_POINT_HISTORY_LABELS = Path(os.getenv(
        'GESTURE_POINT_HISTORY_LABELS',
        GESTURE_POINT_HISTORY_DIR / 'point_history_classifier_label.csv'
    ))
    GESTURE_KEYPOINT_CSV = Path(os.getenv(
        'GESTURE_KEYPOINT_CSV',
        GESTURE_KEYPOINT_DIR / 'keypoint.csv'
    ))
    GESTURE_POINT_HISTORY_CSV = Path(os.getenv(
        'GESTURE_POINT_HISTORY_CSV',
        GESTURE_POINT_HISTORY_DIR / 'point_history.csv'
    ))
    GESTURE_POINT_HISTORY_TRIGGER_IDS = [
        int(item.strip()) for item in os.getenv(
            'GESTURE_POINT_HISTORY_TRIGGER_IDS',
            '2'
        ).split(',') if item.strip().isdigit()
    ] or [2]
    
    # Model Path
    GESTURE_MODEL_PATH = MODELS_DIR / 'gesture_model.pkl'
    
    # UI Settings
    WINDOW_NAME = "Sign Language Recognition"
    FONT_SCALE = 1.0
    FONT_THICKNESS = 2
    
    # Colors (BGR format for OpenCV)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_BLUE = (255, 0, 0)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_YELLOW = (0, 255, 255)
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (cls.DATA_DIR / 'training').mkdir(parents=True, exist_ok=True)
        (cls.DATA_DIR / 'testing').mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate configuration
        
        Returns:
            True if configuration is valid, False otherwise
        """
        errors = []
        
        # Check required settings
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is not set")
        
        # Check value ranges
        if not 0.0 <= cls.MIN_DETECTION_CONFIDENCE <= 1.0:
            errors.append("MIN_DETECTION_CONFIDENCE must be between 0 and 1")
        
        if not 0.0 <= cls.MIN_TRACKING_CONFIDENCE <= 1.0:
            errors.append("MIN_TRACKING_CONFIDENCE must be between 0 and 1")
        
        if not 0.0 <= cls.GESTURE_CONFIDENCE_THRESHOLD <= 1.0:
            errors.append("GESTURE_CONFIDENCE_THRESHOLD must be between 0 and 1")
        
        # Print errors
        if errors:
            print("Configuration Errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "=" * 60)
        print("APPLICATION CONFIGURATION")
        print("=" * 60)
        print(f"Camera: Index={cls.CAMERA_INDEX}, Resolution={cls.CAMERA_WIDTH}x{cls.CAMERA_HEIGHT}")
        print(f"MediaPipe: Detection={cls.MIN_DETECTION_CONFIDENCE}, Tracking={cls.MIN_TRACKING_CONFIDENCE}")
        print(f"Gesture: Threshold={cls.GESTURE_CONFIDENCE_THRESHOLD}, Buffer={cls.BUFFER_SIZE}")
        print(f"TTS: Model={cls.TTS_MODEL}, Voice={cls.TTS_VOICE}")
        print(f"Debug Mode: {cls.DEBUG_MODE}")
        print(f"Show FPS: {cls.SHOW_FPS}")
        print(f"Use TFLite Pipeline: {cls.USE_TFLITE_PIPELINE}")
        print(f"Gesture Assets Dir: {cls.GESTURE_ASSETS_DIR}")
        print("=" * 60 + "\n")
    
    @classmethod
    def get_dict(cls) -> Dict[str, Any]:
        """
        Get configuration as dictionary
        
        Returns:
            Configuration dictionary
        """
        return {
            'camera': {
                'index': cls.CAMERA_INDEX,
                'width': cls.CAMERA_WIDTH,
                'height': cls.CAMERA_HEIGHT
            },
            'mediapipe': {
                'detection_confidence': cls.MIN_DETECTION_CONFIDENCE,
                'tracking_confidence': cls.MIN_TRACKING_CONFIDENCE,
                'max_hands': cls.MAX_HANDS
            },
            'gesture': {
                'confidence_threshold': cls.GESTURE_CONFIDENCE_THRESHOLD,
                'buffer_size': cls.BUFFER_SIZE
            },
            'tts': {
                'model': cls.TTS_MODEL,
                'voice': cls.TTS_VOICE,
                'language': cls.TTS_LANGUAGE
            },
            'app': {
                'debug_mode': cls.DEBUG_MODE,
                'show_fps': cls.SHOW_FPS,
                'log_level': cls.LOG_LEVEL
            }
        }


# NOTE: Directories will be created only when needed
# Uncomment the line below if you want to auto-create directories:
# Config.create_directories()


if __name__ == "__main__":
    # Test configuration
    Config.print_config()
    
    is_valid = Config.validate()
    if is_valid:
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has errors")
