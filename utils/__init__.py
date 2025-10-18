"""
Utility Package
Provides configuration and helper functions
"""

from .config import Config
from .helpers import (
    FPSCounter,
    TextRenderer,
    UIComponents,
    resize_with_aspect_ratio,
    is_point_in_rect,
    create_gradient_background
)

__all__ = [
    'Config',
    'FPSCounter',
    'TextRenderer',
    'UIComponents',
    'resize_with_aspect_ratio',
    'is_point_in_rect',
    'create_gradient_background'
]
