# 🤟 Sign Language Recognition - Nhận Diện Ngôn Ngữ Ký Hiệu

## 📋 Tổng Quan

Ứng dụng Python nhận diện ngôn ngữ ký hiệu (ASL alphabet) và chuyển đổi thành giọng nói để hỗ trợ giao tiếp cho người khiếm thính.

### 🎯 Mục Tiêu
- Nhận diện các ký hiệu tay (A-Z) qua camera
- Chuyển đổi thành văn bản
- Phát ra giọng nói tự nhiên bằng AI

### 🛠️ Tech Stack
- **MediaPipe Hands**: Phát hiện và theo dõi bàn tay (21 landmarks)
- **OpenCV**: Xử lý video real-time
- **OpenAI TTS**: Chuyển văn bản thành giọng nói
- **Scikit-learn**: Training gesture classifier
- **Python 3.11**: Ngôn ngữ chính ⚠️ MediaPipe không support Python 3.13

---

## 📁 Cấu Trúc Dự Án

```
School Computer Vision/
├── src/
│   ├── hand_detector.py          # Phát hiện bàn tay (21 landmarks)
│   ├── gesture_classifier.py     # Phân loại ký hiệu A-Z
│   ├── text_to_speech.py         # OpenAI TTS integration
│   └── main.py                   # Ứng dụng chính
├── utils/
│   ├── config.py                 # Quản lý cấu hình
│   └── helpers.py                # UI utilities
├── models/                       # ML models (sau khi train)
├── data/                         # Training/testing data
│   ├── training/
│   └── testing/
├── .env                          # API keys & settings
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation này
└── MEMORY.md                     # Ghi chú cho AI
```

---

## 🔄 Workflow

```
Camera → MediaPipe Hands → Classifier → Speech Buffer → OpenAI TTS
  🎥         👋 (21 pts)       A-Z        "HELLO"         🔊
```

**Chi tiết:**
1. **Camera**: Capture video frame
2. **Hand Detector**: Detect hand → 21 landmarks (x, y coordinates)
3. **Gesture Classifier**: Landmarks → Letter prediction (A, B, C... + confidence)
4. **Speech Buffer**: Accumulate letters into words
5. **Text-to-Speech**: Convert text to voice

---

## 🚀 Quick Start

### 1. Fix Python Version ⚠️ IMPORTANT
**MediaPipe chỉ hỗ trợ Python 3.8 - 3.11**

```powershell
# Download Python 3.11 từ python.org
# Tạo virtual environment:
py -3.11 -m venv venv
.\venv\Scripts\activate
```

### 2. Cài Đặt Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Cấu Hình

```powershell
copy .env.example .env
# Edit .env: Thêm OPENAI_API_KEY=your-key-here
```

### 4. Test

```powershell
# Test hand detection
python src/hand_detector.py

# Test TTS (cần API key)
python src/text_to_speech.py

# Run full app
python src/main.py
```

---

## 🎮 Keyboard Controls

| Key | Action |
|-----|--------|
| `SPACE` | Thêm space giữa các từ |
| `ENTER` | Speak text hiện tại |
| `BACKSPACE` | Xóa ký tự cuối |
| `C` | Clear all text |
| `P` | Pause/Resume |
| `Q` | Quit |

---

## 📊 Kế Hoạch 8 Tuần

### Week 1-2: Setup ✅ DONE
- [x] Project structure
- [x] Hand detector module
- [x] Basic modules

### Week 3-4: Data & Training 🔄 CURRENT
- [ ] Data collection tool
- [ ] Collect 100+ samples/letter (A-Z)
- [ ] Train classifier (target: 85% accuracy)

### Week 5: TTS Integration
- [ ] Complete OpenAI TTS
- [ ] Speech buffer system
- [ ] Word completion logic

### Week 6: UI/UX
- [ ] User interface
- [ ] Visual feedback
- [ ] Settings panel

### Week 7-8: Testing & Polish
- [ ] Unit tests
- [ ] User testing
- [ ] Performance optimization
- [ ] Documentation

---

## 💻 Code Examples

### Hand Detection
```python
from hand_detector import HandDetector
import cv2

detector = HandDetector(max_hands=1, detection_confidence=0.7)
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = detector.find_hands(img, draw=True)
    landmarks = detector.find_position(img)
    
    if landmarks:
        print(f"Found {len(landmarks)} landmarks")
    
    cv2.imshow("Hand Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Gesture Classification
```python
from gesture_classifier import GestureClassifier

classifier = GestureClassifier(model_path="models/gesture_model.pkl")
gesture, confidence = classifier.classify_gesture(landmarks)
print(f"Detected: {gesture} (confidence: {confidence:.2f})")
```

### Text-to-Speech
```python
from text_to_speech import TextToSpeech

tts = TextToSpeech(api_key="your-key", voice="alloy")
tts.text_to_speech("Hello World")  # 🔊
```

---

## 🐛 Troubleshooting

### MediaPipe Install Error
**Problem**: `Could not find mediapipe`  
**Solution**: Python version phải là 3.8-3.11 (không phải 3.13)

```powershell
python --version  # Check version
py -3.11 -m venv venv  # Use Python 3.11
```

### Camera Not Working
**Problem**: Camera không mở  
**Solution**: Thử đổi `CAMERA_INDEX` trong .env (0, 1, 2...)

### Low FPS
**Problem**: FPS < 20  
**Solution**: Giảm resolution
```python
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
```

### OpenAI API Error
**Problem**: TTS không work  
**Solution**: 
- Check API key đúng
- Verify account có credits
- Check internet connection

---

## 📦 Dependencies

```txt
opencv-python>=4.8.0          # Video processing
mediapipe>=0.10.0             # Hand detection
numpy>=1.24.0                 # Numerical computing
openai>=1.0.0                 # Text-to-Speech
pygame>=2.5.0                 # Audio playback
scikit-learn>=1.3.0           # ML classification
python-dotenv>=1.0.0          # Environment variables
tqdm>=4.65.0                  # Progress bars
loguru>=0.7.0                 # Logging
```

---

## 🎯 Success Metrics

| Metric | Target |
|--------|--------|
| Hand Detection FPS | >= 30 FPS |
| Gesture Accuracy | >= 85% |
| TTS Latency | < 2 seconds |
| System Stability | No crash 30min |

---

## 🔧 Configuration (.env)

```bash
# OpenAI
OPENAI_API_KEY=your_api_key_here

# Camera
CAMERA_INDEX=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720

# MediaPipe
MIN_DETECTION_CONFIDENCE=0.7
MIN_TRACKING_CONFIDENCE=0.5

# Gesture Recognition
GESTURE_CONFIDENCE_THRESHOLD=0.8
BUFFER_SIZE=30

# TTS
TTS_MODEL=tts-1
TTS_VOICE=alloy  # Options: alloy, echo, fable, onyx, nova, shimmer

# App
DEBUG_MODE=False
SHOW_FPS=True
```

---

## 📚 Learning Resources

- [MediaPipe Hands Docs](https://google.github.io/mediapipe/solutions/hands.html)
- [ASL Alphabet](https://www.startasl.com/american-sign-language-alphabet/)
- [OpenAI TTS API](https://platform.openai.com/docs/guides/text-to-speech)
- [Scikit-learn](https://scikit-learn.org/stable/)

---

## 🎯 Next Steps

### This Week
1. Fix Python version (3.13 → 3.11)
2. Install MediaPipe successfully
3. Test hand detection

### Week 3-4
4. Create data collection tool
5. Collect training data (A-Z)
6. Train classifier model

### Week 5+
7. Integrate TTS
8. Polish UI
9. User testing

---

## 📝 Known Issues

1. **Python 3.13**: MediaPipe chưa support → Cần 3.11
2. **No Trained Model**: Cần collect data và train
3. **Static Only**: Chỉ nhận diện static gestures (A-Z)
4. **ASL Only**: Chưa support Vietnamese Sign Language

---

## 🤝 Contributing

Improvements welcome:
- Data collection
- Model optimization
- New features
- Bug fixes
- Documentation

---

## 📄 License
MIT License

---

## 📞 Support

- Check [MEMORY.md](MEMORY.md) for AI context
- Review code comments in src/
- Google error với "MediaPipe" hoặc "OpenCV"

---

**Version**: 0.1.0 (MVP Phase)  
**Last Updated**: October 13, 2025  
**Status**: Development - Hand Detection Working ✅
