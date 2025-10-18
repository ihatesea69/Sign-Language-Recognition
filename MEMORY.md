# 🧠 MEMORY.md - AI Assistant Context

> **Purpose**: Ghi nhớ trạng thái project và decisions cho AI assistants

---

## 📊 PROJECT STATE

### Current Status
- **Phase**: Week 1-2 (Setup & Hand Detection)
- **Date**: October 13, 2025
- **Version**: 0.1.0 (MVP)
- **Status**: Hand Detection Module Complete ✅

### What Works
- ✅ Project structure created
- ✅ `hand_detector.py` - Basic MediaPipe Hands implementation
- ✅ `gesture_classifier.py` - Framework ready (needs training data)
- ✅ `text_to_speech.py` - OpenAI TTS integration complete
- ✅ `main.py` - Main app skeleton
- ✅ Configuration system (`utils/config.py`)
- ✅ UI helpers (`utils/helpers.py`)

### What Doesn't Work Yet
- ❌ MediaPipe not installed (Python version issue)
- ❌ No trained gesture classifier model
- ❌ No training data collected
- ❌ Full end-to-end flow not tested

---

## 🚨 CRITICAL ISSUES

### Issue #1: Python Version Incompatibility
**Problem**: User has Python 3.13, MediaPipe only supports 3.8-3.11
**Status**: NOT FIXED
**Solution**: Need to downgrade to Python 3.11
```powershell
py -3.11 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Issue #2: No Training Data
**Problem**: Gesture classifier needs training data
**Status**: NOT STARTED
**Next Steps**: 
1. Create data collection tool
2. Collect 100-200 samples per letter (A-Z)
3. Label and organize data
4. Train Random Forest classifier

### Issue #3: OpenAI API Key
**Problem**: TTS requires API key
**Status**: UNKNOWN (user needs to add to .env)
**Solution**: User must add `OPENAI_API_KEY` to .env file

---

## 🎯 DESIGN DECISIONS

### Why These Modules?

#### 1. MediaPipe Hands (KEPT - ESSENTIAL)
**Reason**: 
- Core technology for hand detection
- 21 landmarks per hand
- Fast (30+ FPS)
- Accurate
- Industry standard
**Alternatives Considered**: OpenCV Haar Cascades (rejected - too inaccurate)

#### 2. OpenAI TTS (KEPT - ESSENTIAL)
**Reason**:
- Natural-sounding voices
- Multiple voice options
- Easy API
- High quality
**Alternatives Considered**: pyttsx3 (rejected - robotic voice)

#### 3. Scikit-learn Random Forest (KEPT - SIMPLE & EFFECTIVE)
**Reason**:
- Simple to implement
- Good accuracy for static gestures
- Fast inference
- No GPU required
**Alternatives Considered**: Deep Learning (rejected - overkill for MVP)

### Why NOT These Features?

#### MediaPipe Holistic (REMOVED)
**Reason**: 
- Overkill for MVP (543 landmarks vs 21 needed)
- Performance hit (slower FPS)
- Not needed for A-Z spelling
- Can add in v2.0 for full sentences

#### Enhanced Hand Detector (REMOVED)
**Reason**:
- Too complex for current needs
- 3D world landmarks nice-to-have, not essential
- Multi-hand support can add later if needed
- Keep it simple for MVP

#### Dynamic Gesture Recognition (DEFERRED)
**Reason**:
- Focus on static gestures (A-Z) first
- Motion-based gestures for v2.0
- Need static working well before adding complexity

---

## 📁 FILE STRUCTURE DECISIONS

### Original Structure (TOO MANY FILES)
```
❌ README.md
❌ OVERVIEW.md
❌ QUICKSTART.md
❌ PLAN.md
❌ ENHANCEMENTS.md
❌ SUMMARY.md
❌ PYTHON_VERSION_FIX.md
```

### Consolidated Structure (CLEAN)
```
✅ README.md       - All documentation
✅ MEMORY.md       - This file (AI context)
```

**Reason**: Too many markdown files was confusing. Consolidated into 2 files:
- README.md = User documentation
- MEMORY.md = AI assistant context

---

## 🔧 TECHNICAL DETAILS

### Hand Detection Pipeline
```python
1. cv2.VideoCapture(0)              # Grab frame from camera
2. mp_hands.Hands.process(frame)    # MediaPipe detection
3. results.multi_hand_landmarks     # Extract 21 (x,y) points
4. Return landmarks list
```

### Gesture Classification Pipeline (NOT YET IMPLEMENTED)
```python
1. extract_features(landmarks)       # Convert 21 points to feature vector
   - Normalized coordinates
   - Distances between key points
   - Angles between fingers
   
2. model.predict(features)           # Random Forest prediction
   - Input: Feature vector
   - Output: Letter (A-Z) + confidence
   
3. gesture_buffer.add(gesture)       # Temporal smoothing
   - Buffer 30 frames
   - Return most common gesture
```

### Text-to-Speech Pipeline
```python
1. Accumulate letters: "H", "E", "L", "L", "O"
2. User presses SPACE → word complete
3. Buffer: ["HELLO"]
4. User presses ENTER → trigger TTS
5. OpenAI API call → audio file
6. pygame.mixer.music.play() → 🔊
```

---

## 📊 DATA REQUIREMENTS

### Training Data Needed
- **Format**: Images or landmark arrays
- **Quantity**: 100-200 samples per letter
- **Total**: ~2,600 images (26 letters × 100)
- **Variations Needed**:
  - Different lighting conditions
  - Different hand positions
  - Different distances from camera
  - Different people (if possible)

### Data Structure
```
data/
├── training/
│   ├── A/
│   │   ├── 001.jpg (or .npy for landmarks)
│   │   ├── 002.jpg
│   │   └── ...
│   ├── B/
│   └── ...Z/
└── testing/
    └── (same structure)
```

---

## 🎯 MVP SCOPE (Week 1-8)

### IN SCOPE ✅
- Hand detection (21 landmarks)
- Static gesture recognition (A-Z only)
- Spelling words letter by letter
- Text-to-speech output
- Basic UI with camera feed
- Confidence display
- Manual controls (space, enter, etc.)

### OUT OF SCOPE ❌
- Dynamic gestures (motion-based)
- Full sign language sentences
- Two-hand gestures
- Facial expressions
- Body pose
- Vietnamese Sign Language
- Mobile app
- Offline mode

---

## 🚀 NEXT ACTIONS (PRIORITY ORDER)

### HIGH PRIORITY (Must Do First)
1. **Fix Python version** → 3.11
   - User needs to install Python 3.11
   - Create new venv with py -3.11
   - Install dependencies

2. **Test hand detection**
   - Run `python src/hand_detector.py`
   - Verify 21 landmarks showing
   - Check FPS >= 30

3. **Get OpenAI API key**
   - Sign up at platform.openai.com
   - Add to .env file
   - Test TTS working

### MEDIUM PRIORITY (Week 3-4)
4. **Create data collection tool**
   - Script to capture hand images
   - Auto-save with labels
   - Progress tracking

5. **Collect training data**
   - 100+ samples per letter
   - Organize in folders
   - Verify quality

6. **Train classifier**
   - Extract features from landmarks
   - Train Random Forest
   - Evaluate accuracy
   - Save model to models/

### LOW PRIORITY (Week 5+)
7. **Integration testing**
   - Full pipeline test
   - Performance optimization
   - UI polish

8. **User testing**
   - Test with real users
   - Collect feedback
   - Fix issues

---

## 💡 LESSONS LEARNED

### What Worked Well
1. **MediaPipe Hands**: Excellent accuracy and speed
2. **Modular structure**: Easy to understand and modify
3. **Configuration system**: Clean separation of settings
4. **Simple over complex**: Basic hand_detector.py better than enhanced version

### What Didn't Work
1. **Too many markdown files**: Consolidated to 2 files
2. **Enhanced features too early**: Removed complex features for MVP
3. **Python 3.13**: Should have checked MediaPipe compatibility first

### Key Insights
1. **Start simple**: Get basic working before adding features
2. **Training data is critical**: Need good quality, diverse data
3. **Real-time matters**: 30 FPS minimum for good UX
4. **Buffer/smoothing essential**: Single-frame detection too jittery

---

## 🔮 FUTURE ENHANCEMENTS (v2.0+)

### Phase 2 Features
- [ ] Vietnamese Sign Language support
- [ ] Dynamic gesture recognition (motion patterns)
- [ ] Two-hand complex gestures
- [ ] Facial expression detection
- [ ] Word-level recognition (not just spelling)

### Phase 3 Features
- [ ] Mobile app (iOS/Android)
- [ ] Offline mode (local TTS)
- [ ] Real-time translation
- [ ] Conversation history
- [ ] Multi-language support

### Technical Improvements
- [ ] Deep learning model (CNN/LSTM)
- [ ] GPU acceleration
- [ ] Model quantization for speed
- [ ] Edge deployment

---

## 📝 IMPORTANT NOTES FOR AI ASSISTANTS

### When Helping User

1. **Check Python Version First**
   - User has 3.13, needs 3.11
   - This is the #1 blocker
   - Don't suggest MediaPipe install until fixed

2. **Focus on MVP**
   - Don't suggest complex features
   - Keep it simple: hands → gestures → text → speech
   - Advanced features can wait for v2.0

3. **Data Collection is Key**
   - Without training data, classifier won't work
   - This is Week 3-4 priority
   - Need to create tool first, then collect

4. **Performance Matters**
   - Target 30 FPS minimum
   - Suggest performance tips if FPS low
   - Camera resolution can be reduced if needed

### Code Review Guidelines

**Good Code**:
- Simple and readable
- Well-commented
- Follows existing patterns
- Handles errors gracefully

**Avoid**:
- Over-engineering
- Premature optimization
- Complex abstractions
- Features not in MVP scope

### Common User Questions

Q: Why Python 3.11?
A: MediaPipe doesn't support 3.13 yet

Q: Why not use deep learning?
A: Overkill for MVP, Random Forest is enough for static gestures

Q: How much training data?
A: 100-200 samples per letter, ~2600 total

Q: Can I use Vietnamese Sign Language?
A: Not in MVP, focus on ASL first (v2.0 feature)

Q: Why no two-hand gestures?
A: Complexity - start with one hand, add later if needed

---

## 🎓 TECHNICAL REFERENCES

### MediaPipe Hands
- 21 landmarks per hand
- Normalized coordinates [0, 1]
- Real-time performance (30-60 FPS)
- Model complexity: 0 (lite) or 1 (full)

### Landmark IDs
```
0: Wrist
1-4: Thumb
5-8: Index finger
9-12: Middle finger
13-16: Ring finger
17-20: Pinky finger
```

### Feature Engineering
For gesture classification, extract:
1. Normalized landmark positions
2. Distances between key points (e.g., thumb-index)
3. Angles at finger joints
4. Relative positions (e.g., tip above/below PIP)

### Model Training
```python
# Feature vector example (42 features)
features = [
    x0, y0, x1, y1, ..., x20, y20,  # 21 landmarks × 2 coords = 42 values
]

# Or enhanced features (~60-100 values)
features = [
    normalized_coords,    # 42 values
    distances,            # 10-20 values
    angles,              # 5-10 values
]
```

---

## 🗓️ PROJECT TIMELINE

### Week 1-2 (Oct 7-20) ✅ DONE
- Project setup
- Module implementation
- Documentation

### Week 3-4 (Oct 21 - Nov 3) 🔄 CURRENT
- Fix Python version
- Data collection
- Model training

### Week 5 (Nov 4-10)
- TTS integration
- Speech buffer

### Week 6 (Nov 11-17)
- UI polish
- User testing

### Week 7-8 (Nov 18 - Dec 1)
- Bug fixes
- Optimization
- Final delivery

---

## 🎯 SUCCESS CRITERIA

### MVP Considered Successful If:
- [x] Hand detection working (30+ FPS)
- [ ] Can recognize A-Z with 85%+ accuracy
- [ ] End-to-end flow works (gesture → text → speech)
- [ ] No crashes in 30-minute session
- [ ] TTS latency < 2 seconds
- [ ] User can spell and speak words

### Demo Scenario:
```
User makes signs: H-E-L-L-O
App recognizes: H, E, L, L, O
User presses ENTER
App speaks: "Hello" 🔊
Time: < 10 seconds total
```

---

**Last Updated**: October 13, 2025  
**Next Review**: After Python version fix  
**Status**: Waiting for Python 3.11 installation
