#🤟SignLanguageRecognitionwithText-to-Speech

[](https://www.python.org/downloads/)
[](https://opencv.org/)
[](https://mediapipe.dev/)
[](https://www.tensorflow.org/)
[](LICENSE)


>**Areal-timesignlanguagerecognitionsystemwithText-to-Speechoutput,poweredbyMediaPipeandMachineLearning**



---

##TableofContents

-[Overview](#-overview)
-[Features](#-features)
-[SystemArchitecture](#-system-architecture)
-[Installation](#-installation)
-[Usage](#-usage)
-[ProjectStructure](#-project-structure)
-[Algorithms](#-algorithms)
-[Configuration](#-configuration)
-[Performance](#-performance)
-[Development](#-development)
-[Contributing](#-contributing)
-[License](#-license)
-[Acknowledgments](#-acknowledgments)

---

##Overview

Thisprojectimplementsa**real-timesignlanguagerecognitionsystem**that:
-Detectshandgesturesusing**MediaPipeHands**
-Recognizesgesturesthrough**Rule-Based**and**MachineLearning**algorithms
-Convertsrecognizedsignstospeechusing**OpenAIText-to-SpeechAPI**
-Providesaninteractiveinterfaceforseamlesscommunication

###Demo



---

##Features

###HandDetection&Tracking
-**21handlandmarks**detectionusingMediaPipe
-Real-timetrackingwith**30FPS**performance
-Supportforboth**leftandrighthands**
-Robustdetectionundervariouslightingconditions

###GestureRecognition
####Rule-BasedRecognition
-**15-20staticgestures**(A-Zletters,numbers0-9)
-Commonsigns:OK,Peace,ThumbsUp,Fist,etc.
-**Geometricfeatureextraction**(angles,distances)
-**Notrainingrequired**

####MachineLearningRecognition(TFLite)
-🤖**NeuralNetwork**forstatichandsigns
-🤖**LSTM/GRU**fordynamicmotiongestures
-🤖**Keypointclassification**(42features)
-🤖**Pointhistorytracking**(16-pointbuffer)
-🤖**85-95%accuracy**ontrainedgestures

###Text-to-Speech
-**OpenAITTSAPI**integration
-**6voiceoptions**(alloy,echo,fable,onyx,nova,shimmer)
-Natural-soundingspeechoutput
-Real-timeaudioplaybackwithpygame

###UserInterface
-**Real-timeFPScounter**
-**Gestureconfidencedisplay**
-**Textbuffervisualization**
-**Keyboardshortcuts**forquickactions
-**Clean,informativeoverlay**

---

##SystemArchitecture

```

CAMERAINPUT
1280×720@30fps

↓

HANDDETECTION
MediaPipeHands(GPU)
•Detect21handlandmarks
•Normalizecoordinates
•Trackhandmovement

↓



Rule-BasedTFLiteML
RecognitionPipeline

•Geometric•Keypoint
FeaturesClassifier
•Heuristics•Point
•75-85%History
Accuracy•85-95%
Accuracy



↓

SpeechBuffer
•Accumulate
•Format

↓

Text-to-Speech
(OpenAIAPI)

↓

AudioOutput
(Pygame)

```

---

##Installation

###Prerequisites

-**Python3.8+**
-**Webcam**(built-inorexternal)
-**OpenAIAPIKey**(forText-to-Speechfeature)
-**GPU**(optional,forbetterperformance)

###Step1:ClonetheRepository

```bash
gitclonehttps://github.com/ihatesea69/Sign-Language-Recognition.git
cdSign-Language-Recognition
```

###Step2:CreateVirtualEnvironment

```bash
#Windows
python-mvenvvenv
venv\Scripts\activate

#macOS/Linux
python3-mvenvvenv
sourcevenv/bin/activate
```

###Step3:InstallDependencies

```bash
pipinstall-rrequirements.txt
```

###Step4:ConfigureEnvironment

1.Copytheexampleenvironmentfile:
```bash
copy.env.example.env#Windows
#or
cp.env.example.env#macOS/Linux
```

2.Edit`.env`andaddyourOpenAIAPIkey:
```env
OPENAI_API_KEY=your_api_key_here
```

###Step5:(Optional)DownloadTFLiteModels

IfusingMachineLearningrecognition:
```bash
#Modelsshouldbeplacedin:
#models/gesture/keypoint_classifier/keypoint_classifier.tflite
#models/gesture/point_history_classifier/point_history_classifier.tflite
```

---

##Usage

###BasicUsage

Runthemainapplication:
```bash
pythonsrc/main.py
```

###KeyboardControls

|Key|Action|
|-----|--------|
|**SPACE**|Addspacetotextbuffer|
|**ENTER**|Speakaccumulatedtext|
|**BACKSPACE**|Deletelastcharacter|
|**C**|Cleartextbuffer|
|**P**|Pause/Resumedetection|
|**Q**|Quitapplication|

###TFLiteTrainingMode(Optional)

|Key|Action|
|-----|--------|
|**0-9**|Selectlabelforlogging|
|**K**|Logkeypointdata|
|**H**|Logpointhistorydata|
|**N**|Stoplogging|

###ConfigurationOptions

Edit`.env`filetocustomize:

```env
#CameraSettings
CAMERA_INDEX=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720

#DetectionSettings
MIN_DETECTION_CONFIDENCE=0.7
MIN_TRACKING_CONFIDENCE=0.5
GESTURE_CONFIDENCE_THRESHOLD=0.8

#TTSSettings
TTS_MODEL=tts-1
TTS_VOICE=alloy
TTS_LANGUAGE=en

#RecognitionMode
USE_TFLITE_PIPELINE=False
ENABLE_GESTURE_DATA_LOGGING=False

#Display
SHOW_FPS=True
DEBUG_MODE=False
```

---

##ProjectStructure

```
Sign-Language-Recognition/
README.md#Thisfile
requirements.txt#Pythondependencies
.env.example#Environmentvariablestemplate
.gitignore#Gitignorerules

src/#Sourcecode
__init__.py
main.py#Mainapplicationentrypoint
hand_detector.py#MediaPipehanddetection
gesture_recognizer.py#Rule-basedrecognition
text_to_speech.py#TTSintegration

gesture_ml/#MachineLearningpipeline
__init__.py
tflite_pipeline.py#TFLitegesturepipeline
keypoint_classifier.py
point_history_classifier.py

utils/#Utilitymodules
__init__.py
config.py#Configurationmanagement
helpers.py#UIcomponents,FPScounter

models/#Trainedmodels
gesture/
keypoint_classifier/
keypoint_classifier.tflite
keypoint_classifier_label.csv
point_history_classifier/
point_history_classifier.tflite
point_history_classifier_label.csv

data/#Trainingdata(optional)
raw/
processed/

docs/#Documentation
PHU_LUC_CODE.md#Codeappendix(Vietnamese)
PHU_LUC_THUAT_TOAN_CHINH.md#Algorithmappendix
assets/#Images,diagrams

notebooks/#Jupyternotebooks(ifany)
```

---

##🧠Algorithms

###1.HandDetectionAlgorithm

**Method:**MediaPipeHands(BlazePalm+BlazeLandmark)

```python
#Pseudo-code
defdetect_hand(image):
1.ConvertBGRtoRGB
2.ApplyMediaPipeHandsdetection
3.Extract21landmarks(ifdetected)
4.Normalizecoordinatesto[0,1]
5.Converttopixelcoordinates
returnlandmarks
```

**Complexity:**O(1)-constanttime(optimizedCNN)

###2.Rule-BasedGestureRecognition

**Features:**
-Fingerstates(up/down)
-Anglesatjoints(PIP,MCP)
-Distancesbetweenlandmarks
-Palmsizenormalization

```python
#Simplifiedalgorithm
defrecognize_gesture(landmarks):
1.Extractgeometricfeatures
-fingers_up=[thumb,index,middle,ring,pinky]
-angles=compute_joint_angles(landmarks)
-distances=compute_pairwise_distances(landmarks)

2.Applyrulematching(priorityorder)
-OKsign:thumb+indextouching
-Peace:index+middleseparated
-Fist:allfingersdown
-...

3.Smoothwithhistorybuffer

return(gesture_name,confidence)
```

**Complexity:**O(1)-fixednumberoflandmarksandrules

###3.MachineLearningRecognition

**Architecture:**
```
Input:21landmarks×2coords=42features
↓
Preprocessing:Normalize&Flatten
↓

KeypointClassifierPointHistoryTracker
(StaticGestures)(DynamicGestures)

DenseNeuralNetLSTM/GRUNetwork
Output:ClassIDOutput:MotionID

↓
Prediction:(hand_sign,finger_gesture)
```

**Preprocessing:**
```python
defpreprocess_landmarks(landmarks):
1.Translatetoorigin(wrist=0,0)
2.Flattento1Darray[x0,y0,x1,y1,...,x20,y20]
3.Normalizebymaxabsolutevalue
returnnormalized_vector
```

**Complexity:**
-Preprocessing:O(n)wheren=21
-Inference:O(m)wherem=modelparameters

---

##Configuration

###CameraSettings

```python
CAMERA_INDEX=0#Cameradeviceindex
CAMERA_WIDTH=1280#Resolutionwidth
CAMERA_HEIGHT=720#Resolutionheight
```

###DetectionThresholds

```python
MIN_DETECTION_CONFIDENCE=0.7#Handdetectionthreshold
MIN_TRACKING_CONFIDENCE=0.5#Handtrackingthreshold
GESTURE_CONFIDENCE_THRESHOLD=0.8#Gestureacceptancethreshold
```

###RecognitionMode

```python
USE_TFLITE_PIPELINE=False#True:ML,False:Rule-based
ENABLE_GESTURE_DATA_LOGGING=False#Enabletrainingdatacollection
```

###Text-to-Speech

```python
TTS_MODEL="tts-1"#Options:"tts-1","tts-1-hd"
TTS_VOICE="alloy"#Options:alloy,echo,fable,onyx,nova,shimmer
TTS_LANGUAGE="en"#Languagecode
```

---

##Performance

###SystemRequirements

|Component|Minimum|Recommended|
|-----------|---------|-------------|
|**CPU**|Inteli5/AMDRyzen5|Inteli7/AMDRyzen7|
|**RAM**|4GB|8GB|
|**GPU**|Integrated|NVIDIAGTX1050+|
|**Camera**|720p@30fps|1080p@60fps|
|**Python**|3.8+|3.10+|

###BenchmarkResults

|Metric|Rule-Based|TFLiteML|
|--------|-----------|-----------|
|**Accuracy**|75-85%|85-95%|
|**FPS**|~30|~25|
|**Latency**|<10ms|~20ms|
|**Gestures**|15-20|10+(expandable)|
|**Training**|None|Required|

###PerformanceBreakdown(perframe)

```
ComponentTime%Total

CameraCapture5ms15%
HandDetection15ms45%
GestureRecognition8ms24%
UIRendering3ms9%
Other2ms6%

Total33ms100%
ExpectedFPS~30
```

---

##Development

###SettingUpDevelopmentEnvironment

1.**Installdevelopmentdependencies:**
```bash
pipinstall-rrequirements-dev.txt#Ifavailable
```

2.**Enabledebugmode:**
```env
DEBUG_MODE=True
LOG_LEVEL=DEBUG
```

3.**Runtests:**
```bash
#Unittests
python-mpytesttests/

#Integrationtests
python-mpytesttests/integration/
```

###TrainingCustomGestures(TFLite)

1.**Enableloggingmode:**
```env
ENABLE_GESTURE_DATA_LOGGING=True
```

2.**Collecttrainingdata:**
```bash
pythonsrc/main.py
#Press0-9toselectlabel
#PressKtologkeypoints
#PressHtologpointhistory
#Repeatforeachgesture
```

3.**Trainmodels:**
```bash
#Trainkeypointclassifier
pythonscripts/train_keypoint_classifier.py

#Trainpointhistoryclassifier
pythonscripts/train_point_history_classifier.py
```

4.**Deploymodels:**
```bash
#Copytrained.tflitefilestomodels/gesture/
```

###CodeStyle

-**PEP8**compliance
-**Typehints**forfunctionsignatures
-**Docstrings**forclassesandmethods
-**Comments**forcomplexlogic

###GitWorkflow

```bash
#Createfeaturebranch
gitcheckout-bfeature/your-feature-name

#Makechangesandcommit
gitadd.
gitcommit-m"feat:addnewgesturerecognition"

#Pushtoremote
gitpushoriginfeature/your-feature-name

#CreatepullrequestonGitHub
```

---

##🤝Contributing

Contributionsarewelcome!Pleasefollowthesesteps:

1.**Fork**therepository
2.**Create**afeaturebranch(`gitcheckout-bfeature/amazing-feature`)
3.**Commit**yourchanges(`gitcommit-m'Addamazingfeature'`)
4.**Push**tothebranch(`gitpushoriginfeature/amazing-feature`)
5.**Open**aPullRequest

###ContributionGuidelines

-Writeclean,documentedcode
-Addtestsfornewfeatures
-Updatedocumentationasneeded
-Followexistingcodestyle
-Berespectfulandconstructive

---

##License

Thisprojectislicensedunderthe**MITLicense**-seethe[LICENSE](LICENSE)filefordetails.

```
MITLicense

Copyright(c)2025SignLanguageRecognitionTeam

Permissionisherebygranted,freeofcharge,toanypersonobtainingacopy
ofthissoftwareandassociateddocumentationfiles(the"Software"),todeal
intheSoftwarewithoutrestriction,includingwithoutlimitationtherights
touse,copy,modify,merge,publish,distribute,sublicense,and/orsell
copiesoftheSoftware,andtopermitpersonstowhomtheSoftwareis
furnishedtodoso,subjecttothefollowingconditions:

Theabovecopyrightnoticeandthispermissionnoticeshallbeincludedinall
copiesorsubstantialportionsoftheSoftware.
```

---

##Acknowledgments

###Technologies&Libraries

-**[MediaPipe](https://mediapipe.dev/)**-Handdetectionandtracking
-**[OpenCV](https://opencv.org/)**-Computervisionoperations
-**[TensorFlowLite](https://www.tensorflow.org/lite)**-MLinference
-**[OpenAI](https://openai.com/)**-Text-to-SpeechAPI
-**[Pygame](https://www.pygame.org/)**-Audioplayback
-**[Python](https://www.python.org/)**-Programminglanguage

###Inspiration&References

-MediaPipeHands:[GoogleAIBlog](https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html)
-SignLanguageDatasets:[WLASL](https://dxli94.github.io/WLASL/),[MS-ASL](https://www.microsoft.com/en-us/research/project/ms-asl/)
-TFLiteGestureRecognition:[Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)

###Team

-**ComputerVisionCourse**-AcademicProject
-**Contributors**-See[CONTRIBUTORS.md](CONTRIBUTORS.md)

---

##Contact&Support

###Issues&BugReports

Ifyouencounteranyissues,please[openanissue](https://github.com/ihatesea69/Sign-Language-Recognition/issues)onGitHub.

###Questions&Discussions

Forquestionsanddiscussions,use[GitHubDiscussions](https://github.com/ihatesea69/Sign-Language-Recognition/discussions).

###Documentation

-**FullDocumentation:**[docs/](docs/)
-**APIReference:**[docs/api/](docs/api/)
-**Tutorials:**[docs/tutorials/](docs/tutorials/)

---

##Roadmap

###CurrentVersion(v1.0)
-Real-timehanddetection
-Rule-basedgesturerecognition
-TFLiteMLpipeline
-Text-to-Speechintegration
-BasicUI

###FutureEnhancements(v2.0)
-Two-handgesturesupport
-Sentenceformation
-Multi-languagesupport
-Mobileapp(iOS/Android)
-Web-basedinterface
-Clouddeployment
-Videorecording&playback
-Gesturecustomization

###Long-termVision
-Communitygesturedatabase
-Real-timetranslation
-AR/VRintegration
-Accessibilityfeatures

---

##Statistics








---

<divalign="center">

**Madewithforthedeafandhard-of-hearingcommunity**

**Starthisrepoifyoufindithelpful!**

[ReportBug](https://github.com/ihatesea69/Sign-Language-Recognition/issues)·
[RequestFeature](https://github.com/ihatesea69/Sign-Language-Recognition/issues)·
[Documentation](docs/)

</div>
