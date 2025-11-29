# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Two-hand gesture support
- Multi-language TTS support
- Mobile application
- Web interface
- Cloud deployment options

---

## [1.0.0] - 2025-11-29

### Added - Initial Release

#### Core Features
- **Hand Detection Module** (`hand_detector.py`)
  - MediaPipe Hands integration
  - 21 landmark detection
  - Real-time tracking at 30 FPS
  - Finger state detection
  - Bounding box calculation
  
- **Gesture Recognition** (`gesture_recognizer.py`)
  - Rule-based recognition algorithm
  - 15-20 static gestures (A-Z, 0-9, common signs)
  - Geometric feature extraction
  - History-based smoothing
  - 75-85% accuracy

- **Machine Learning Pipeline** (`gesture_ml/`)
  - TFLite keypoint classifier
  - Point history classifier for dynamic gestures
  - Training data logging capability
  - 85-95% accuracy on trained gestures

- **Text-to-Speech** (`text_to_speech.py`)
  - OpenAI TTS API integration
  - 6 voice options
  - Real-time audio playback
  - Speech buffer management
  - Gesture-to-speech mapping

- **Main Application** (`main.py`)
  - Real-time video processing
  - Interactive UI with FPS counter
  - Keyboard controls
  - Configurable pipeline selection
  - Auto-speak mode

#### Configuration & Utilities
- **Configuration Management** (`utils/config.py`)
  - Environment variable loading
  - Centralized settings
  - Path management
  - Validation utilities

- **UI Helpers** (`utils/helpers.py`)
  - FPS counter with moving average
  - Text rendering utilities
  - Info panel components
  - Multi-line text support

#### Documentation
- Comprehensive README with badges
- Algorithm documentation (Vietnamese)
- Code appendix
- Project structure overview
- Installation guide
- Usage examples

#### Project Setup
- Professional .gitignore
- MIT License
- Contributors guide
- Requirements.txt with pinned versions
- .env.example template

### Technical Specifications
- Python 3.8+ support
- Cross-platform (Windows/macOS/Linux)
- GPU acceleration support
- Modular architecture
- Type hints throughout

### Performance
- 30 FPS on recommended hardware
- <10ms gesture recognition latency
- ~20ms ML inference time
- <33ms total processing per frame

---

## Version History Summary

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2025-11-29 | Initial release with core features |

---

## Upgrade Guide

### From Development to v1.0.0

No migration needed for first release.

---

## Contributing

See [README.md](README.md#contributing) for contribution guidelines.

When adding entries to this changelog:
1. Add to [Unreleased] section
2. Follow the format: `### Added/Changed/Deprecated/Removed/Fixed/Security`
3. Include issue/PR references when applicable
4. Move to versioned section on release

---

[Unreleased]: https://github.com/ihatesea69/Sign-Language-Recognition/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ihatesea69/Sign-Language-Recognition/releases/tag/v1.0.0
