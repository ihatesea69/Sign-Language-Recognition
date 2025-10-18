"""
Helper Utilities
Common utility functions used across the application
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional


class FPSCounter:
    """Calculate and display FPS"""
    
    def __init__(self, avg_frames: int = 30):
        """
        Initialize FPS counter
        
        Args:
            avg_frames: Number of frames to average over
        """
        self.avg_frames = avg_frames
        self.frame_times = []
        self.fps = 0
    
    def update(self) -> float:
        """
        Update FPS calculation
        
        Returns:
            Current FPS
        """
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only recent frames
        if len(self.frame_times) > self.avg_frames:
            self.frame_times.pop(0)
        
        # Calculate FPS
        if len(self.frame_times) >= 2:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            self.fps = (len(self.frame_times) - 1) / time_diff if time_diff > 0 else 0
        
        return self.fps
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.fps
    
    def draw_fps(self, img: np.ndarray, position: Tuple[int, int] = (10, 30)):
        """
        Draw FPS on image
        
        Args:
            img: Image to draw on
            position: Text position (x, y)
        """
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(
            img, fps_text, position,
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )


class TextRenderer:
    """Utility for rendering text on images"""
    
    @staticmethod
    def draw_text(
        img: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float = 1.0,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        bg_color: Optional[Tuple[int, int, int]] = None
    ):
        """
        Draw text on image with optional background
        
        Args:
            img: Image to draw on
            text: Text to draw
            position: Text position (x, y)
            font_scale: Font scale
            color: Text color (BGR)
            thickness: Text thickness
            bg_color: Background color (BGR), None for transparent
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        x, y = position
        
        # Draw background rectangle if specified
        if bg_color is not None:
            padding = 5
            cv2.rectangle(
                img,
                (x - padding, y - text_height - padding),
                (x + text_width + padding, y + baseline + padding),
                bg_color,
                -1
            )
        
        # Draw text
        cv2.putText(
            img, text, (x, y),
            font, font_scale, color, thickness
        )
    
    @staticmethod
    def draw_multiline_text(
        img: np.ndarray,
        lines: list,
        start_position: Tuple[int, int],
        font_scale: float = 0.7,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        line_spacing: int = 30
    ):
        """
        Draw multiple lines of text
        
        Args:
            img: Image to draw on
            lines: List of text lines
            start_position: Starting position (x, y)
            font_scale: Font scale
            color: Text color (BGR)
            thickness: Text thickness
            line_spacing: Space between lines
        """
        x, y = start_position
        
        for i, line in enumerate(lines):
            current_y = y + (i * line_spacing)
            TextRenderer.draw_text(
                img, line, (x, current_y),
                font_scale, color, thickness
            )


class UIComponents:
    """Common UI components for the application"""
    
    @staticmethod
    def draw_info_panel(
        img: np.ndarray,
        info: dict,
        position: Tuple[int, int] = (10, 30),
        width: int = 300
    ):
        """
        Draw information panel on image
        
        Args:
            img: Image to draw on
            info: Dictionary of key-value pairs to display
            position: Panel position
            width: Panel width
        """
        x, y = position
        line_height = 30
        padding = 10
        
        # Calculate panel height
        num_lines = len(info)
        height = (num_lines * line_height) + (2 * padding)
        
        # Draw semi-transparent background
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + width, y + height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Draw border
        cv2.rectangle(
            img,
            (x, y),
            (x + width, y + height),
            (255, 255, 255),
            2
        )
        
        # Draw text
        current_y = y + padding + 20
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(
                img, text,
                (x + padding, current_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1
            )
            current_y += line_height
    
    @staticmethod
    def draw_button(
        img: np.ndarray,
        text: str,
        position: Tuple[int, int],
        size: Tuple[int, int] = (150, 50),
        color: Tuple[int, int, int] = (100, 100, 100),
        text_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Tuple[int, int, int, int]:
        """
        Draw a button on image
        
        Args:
            img: Image to draw on
            text: Button text
            position: Button position (x, y)
            size: Button size (width, height)
            color: Button color (BGR)
            text_color: Text color (BGR)
            
        Returns:
            Button bounding box (x, y, width, height)
        """
        x, y = position
        width, height = size
        
        # Draw button rectangle
        cv2.rectangle(
            img,
            (x, y),
            (x + width, y + height),
            color,
            -1
        )
        
        # Draw border
        cv2.rectangle(
            img,
            (x, y),
            (x + width, y + height),
            (255, 255, 255),
            2
        )
        
        # Draw text (centered)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        text_x = x + (width - text_width) // 2
        text_y = y + (height + text_height) // 2
        
        cv2.putText(
            img, text,
            (text_x, text_y),
            font, font_scale, text_color, thickness
        )
        
        return (x, y, width, height)


def resize_with_aspect_ratio(
    img: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    inter=cv2.INTER_AREA
) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        img: Input image
        width: Target width (optional)
        height: Target height (optional)
        inter: Interpolation method
        
    Returns:
        Resized image
    """
    dim = None
    (h, w) = img.shape[:2]
    
    if width is None and height is None:
        return img
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    return cv2.resize(img, dim, interpolation=inter)


def is_point_in_rect(
    point: Tuple[int, int],
    rect: Tuple[int, int, int, int]
) -> bool:
    """
    Check if point is inside rectangle
    
    Args:
        point: Point (x, y)
        rect: Rectangle (x, y, width, height)
        
    Returns:
        True if point is inside rectangle
    """
    px, py = point
    rx, ry, rw, rh = rect
    
    return rx <= px <= rx + rw and ry <= py <= ry + rh


def create_gradient_background(
    width: int,
    height: int,
    color1: Tuple[int, int, int] = (50, 50, 50),
    color2: Tuple[int, int, int] = (20, 20, 20)
) -> np.ndarray:
    """
    Create gradient background image
    
    Args:
        width: Image width
        height: Image height
        color1: Start color (BGR)
        color2: End color (BGR)
        
    Returns:
        Gradient image
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        ratio = i / height
        color = tuple(
            int(c1 * (1 - ratio) + c2 * ratio)
            for c1, c2 in zip(color1, color2)
        )
        img[i, :] = color
    
    return img


# Example usage
if __name__ == "__main__":
    print("Helper Utilities Module")
    print("This module provides utility functions for the application")
    
    # Test FPS counter
    fps_counter = FPSCounter()
    for _ in range(100):
        fps = fps_counter.update()
        time.sleep(0.01)
    
    print(f"Average FPS: {fps_counter.get_fps():.2f}")
