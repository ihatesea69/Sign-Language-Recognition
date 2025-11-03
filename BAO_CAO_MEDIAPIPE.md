# BÁO CÁO KẾT THÚC HỌC PHẦN: THỊ GIÁC MÁY TÍNH

<div align="center">

**TRƯỜNG ĐẠI HỌC NGOẠI NGỮ - TIN HỌC TP. HỒ CHÍ MINH**

**KHOA CÔNG NGHỆ THÔNG TIN**

---

### BÁO CÁO KẾT THÚC HỌC PHẦN

## NHẬN DIỆN CỬ CHỈ TAY THỜI GIAN THỰC VỚI MEDIAPIPE

---

**Tên đề tài:** Hệ thống nhận diện cử chỉ tay thời gian thực bằng MediaPipe và chuyển văn bản thành giọng nói

**Giảng viên hướng dẫn:** ................................

**Sinh viên thực hiện:** ............................

**MSSV:** ............................

**Thành phố Hồ Chí Minh, tháng 11 năm 2025**

</div>

---

<div style="page-break-after: always;"></div>

# THÔNG TIN LỚP HỌC PHẦN

**Mã lớp học phần:** ..................

**Năm học:** 2025 – 2026

**Học kỳ:** 1

---

<div style="page-break-after: always;"></div>

# MỤC LỤC

- [CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI](#chuong-1)
  - [1.1. Giới thiệu bài toán](#11-gioi-thieu-bai-toan)
  - [1.2. Mục tiêu và phạm vi](#12-muc-tieu-va-pham-vi)
  - [1.3. Đóng góp của đề tài](#13-dong-gop)
- [CHƯƠNG 2. KIẾN THỨC NỀN TẢNG MEDIAPIPE](#chuong-2)
  - [2.1. Tổng quan Framework concepts](#21-overview)
  - [2.2. Graphs (đồ thị xử lý)](#22-graphs)
  - [2.3. Calculators (nodes)](#23-calculators)
  - [2.4. Packets](#24-packets)
  - [2.5. Visualizer (công cụ trực quan hóa)](#25-visualizer)
- [CHƯƠNG 3. KIẾN TRÚC HỆ THỐNG VÀ LUỒNG XỬ LÝ](#chuong-3)
  - [3.1. Mô tả tổng quan pipeline](#31-pipeline)
  - [3.2. Ánh xạ thành phần dự án vào khái niệm MediaPipe](#32-anh-xa)
  - [3.3. Chính sách đồng bộ và thời gian thực](#33-realtime)
- [CHƯƠNG 4. DỮ LIỆU, TIỀN XỬ LÝ VÀ NHẬN DẠNG CỬ CHỈ](#chuong-4)
  - [4.1. Dữ liệu và tiền xử lý landmarks](#41-du-lieu)
  - [4.2. Phân loại cử chỉ](#42-phan-loai)
  - [4.3. TTS (Text-To-Speech) và giao diện](#43-tts-ui)
- [CHƯƠNG 5. THỰC NGHIỆM VÀ ĐÁNH GIÁ](#chuong-5)
  - [5.1. Thiết lập môi trường](#51-thiet-lap)
  - [5.2. Chỉ số đánh giá](#52-chi-so)
  - [5.3. Kết quả và phân tích](#53-ket-qua)
- [CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN](#chuong-6)
- [TÀI LIỆU THAM KHẢO](#tai-lieu-tham-khao)
- [PHỤ LỤC](#phu-luc)

---

<div style="page-break-after: always;"></div>

# CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI {#chuong-1}

## 1.1. Giới thiệu bài toán {#11-gioi-thieu-bai-toan}

Dự án xây dựng ứng dụng Python nhận diện ngôn ngữ ký hiệu (bảng chữ cái ASL) theo thời gian thực từ webcam và chuyển kết quả thành giọng nói tự nhiên để hỗ trợ giao tiếp cho người khiếm thính. Hệ thống sử dụng MediaPipe Hands để phát hiện và theo dõi 21 điểm mốc (landmarks) trên bàn tay, trích xuất đặc trưng, phân loại ký hiệu, tích lũy văn bản và phát âm qua dịch vụ TTS.

Tóm tắt pipeline (workflow):

Camera → MediaPipe Hands → Classifier → Speech Buffer → OpenAI TTS

## 1.2. Mục tiêu, phạm vi và ràng buộc {#12-muc-tieu-va-pham-vi}

- Mục tiêu chức năng:
  - Nhận diện các ký hiệu tay A–Z từ webcam theo thời gian thực (>= 30 FPS mục tiêu).
  - Chuyển ký hiệu thành văn bản và phát ra giọng nói tự nhiên (độ trễ TTS mục tiêu < 2 giây).
  - Giao diện hiển thị thông tin thời gian thực (FPS, cử chỉ hiện tại, trạng thái, hướng dẫn phím tắt).

- Phạm vi kỹ thuật và giả định:
  - Nguồn vào: webcam, môi trường ánh sáng trong nhà phổ thông, một người dùng tại một thời điểm.
  - Bộ cử chỉ: ký hiệu tĩnh bảng chữ cái ASL (A–Z); chưa hỗ trợ ngôn ngữ ký hiệu Việt Nam động.
  - Nền tảng: Windows 10/11, Python 3.11 (MediaPipe chưa hỗ trợ 3.13), chạy trong PowerShell.

- Ràng buộc/cảnh báo quan trọng (theo README):
  - MediaPipe hỗ trợ Python 3.8–3.11; cần thiết lập `venv` Python 3.11 để cài đặt thành công.
  - Hiệu năng phụ thuộc độ phân giải camera (khuyến nghị 640×480 để ổn định FPS).
  - TTS yêu cầu `OPENAI_API_KEY`; nếu không có, ứng dụng vẫn chạy nhưng không phát âm.

## 1.3. Công nghệ và kiến trúc tổng quan {#13-dong-gop}

- Tech stack chính:
  - MediaPipe Hands: phát hiện, theo dõi bàn tay (21 landmarks).
  - OpenCV: đọc webcam, hiển thị UI thời gian thực.
  - OpenAI TTS: chuyển văn bản thành giọng nói, bộ đệm phát âm (speech buffer).
  - Scikit-learn (định hướng): huấn luyện/bổ sung bộ phân loại khi có dữ liệu; hiện tại nhận dạng rule-based.
  - Python 3.11, NumPy, dotenv, tqdm, loguru (theo `requirements.txt`).

- Kiến trúc module (thư mục `src/`):
  - `hand_detector.py`: Bao MediaPipe Hands; tìm/vẽ landmarks, trả toạ độ pixel, nhãn tay trái/phải, bounding box.
  - `gesture_recognizer.py`/`gesture_classifier.py`: Xây dựng đặc trưng từ landmarks và phân loại ký hiệu (hiện rule-based, định hướng ML).
  - `text_to_speech.py`: Lớp TTS và `SpeechBuffer` để tích lũy văn bản và phát âm theo sự kiện.
  - `main.py`: Vòng lặp ứng dụng, kết nối các thành phần, debounce theo khung hình, quản lý UI và phím tắt.

## 1.4. Mục tiêu chất lượng và tiêu chí thành công {#14-tieu-chi}

- Chỉ số đích (tham chiếu README):
  - Hand Detection FPS: ≥ 30 FPS.
  - Gesture Accuracy: ≥ 85% (khi có mô hình ML huấn luyện; bản rule-based đạt ổn định cơ bản theo bộ luật).
  - TTS Latency: < 2 giây (khi có kết nối mạng ổn định và API hợp lệ).
  - System Stability: không crash trong 30 phút hoạt động liên tục.

## 1.5. Đóng góp và điểm khác biệt {#15-dong-gop}

- Xây dựng pipeline real-time hoàn chỉnh từ camera đến giọng nói, có cơ chế debounce để giảm nhiễu nhận dạng theo thời gian.
- Tích hợp MediaPipe Hands với UI tối giản, hiển thị FPS, trạng thái, hướng dẫn thao tác bàn phím (SPACE, ENTER, BACKSPACE, C, P, Q).
- Thiết kế mở để chuyển đổi từ rule-based sang mô hình ML (Scikit-learn) khi có dữ liệu huấn luyện A–Z.
- Tài liệu hoá đầy đủ kiến thức nền MediaPipe (chap 2) và ánh xạ trực tiếp vào codebase hiện tại (`src/`).

---

<div style="page-break-after: always;"></div>

# CHƯƠNG 2. KIẾN THỨC NỀN TẢNG MEDIAPIPE {#chuong-2}

Các khái niệm dưới đây được tóm lược từ tài liệu chính thức của MediaPipe và dùng xuyên suốt báo cáo:

## 2.1. Tổng quan Framework concepts {#21-overview}

### 2.1.1. Packet

- Đơn vị luồng dữ liệu cơ bản gồm timestamp số và payload bất biến (immutable). Packet là lớp giá trị (value class), sao chép rẻ với cơ chế đồng sở hữu payload (reference counting). Mỗi bản sao có thể gắn timestamp khác nhau mà không nhân đôi payload lớn.
- Hệ quả: an toàn luồng (do payload bất biến), hiệu quả bộ nhớ (do chia sẻ), thuận tiện fan-out cùng payload tới nhiều nhánh xử lý độc lập.

### 2.1.2. Graph

- Toàn bộ xử lý diễn ra trong đồ thị (graph) định tuyến các packets giữa nodes. Đồ thị hỗ trợ nhiều đầu vào/đầu ra, nhánh/gộp. Dòng dữ liệu thường xuôi nhưng có thể có vòng lặp (xem 2.2.5 Cycles).
- `CalculatorGraphConfig` mô tả đồ thị (pbtxt). Ứng dụng có thể nạp đồ thị, đưa packets vào, nhận kết quả qua callbacks.

### 2.1.3. Nodes (Calculators)

- Node tiêu thụ/sinh packets, là nơi thực hiện tính toán chính. Mỗi node có các cổng (ports) vào/ra (xác định bằng tag/index), và có tuỳ chọn cấu hình (`node_options`).
- Có thể là nguồn (source), trung gian (transform/filter), hay đích (sink).

### 2.1.4. Streams

- Stream là kết nối giữa hai nodes, mang chuỗi packets có timestamp tăng đơn điệu. Mỗi consumer có hàng đợi riêng, tiêu thụ theo nhịp riêng mà không chặn nhau.

### 2.1.5. Side packets

- Kết nối một-lần giữa nodes mang một packet (timestamp không xác định). Dùng để truyền cấu hình/siêu dữ liệu bất biến trong suốt vòng đời graph (ví dụ: tham số mô hình, kích thước tensor).

### 2.1.6. Packet Ports

- Mỗi port có kiểu; packets qua port phải đúng kiểu. Output có thể fan-out tới nhiều input cùng kiểu; side packet output cũng có thể cấp cho nhiều side packet input.
- Port có thể required hoặc optional. Lưu ý: một stream required vẫn có thể “vắng packet” tại một số timestamps.

### 2.1.7. Input và Output của ứng dụng

- Nguồn dữ liệu: từ source nodes (đọc file/camera) hoặc graph input streams (ứng dụng đẩy vào).
- Kết quả: tới sink nodes (ghi file/bộ nhớ) hoặc ứng dụng nhận qua callbacks.

### 2.1.8. Runtime behavior

- Vòng đời graph: khởi tạo → start → xử lý đến khi streams đóng hoặc graph bị cancel → có thể hủy và start lại.
- Vòng đời node: `Open` (một lần, side packets đã sẵn sàng) → `Process` (nhiều lần, mỗi lần là một “tập input” có cùng timestamp, theo input policy) → `Close` (một lần).

### 2.1.9. Input policies và Real-time streams

- Mặc định: deterministic collation theo timestamp. Node nhận đủ các inputs cùng timestamp trước khi `Process`. Điều này có thể trì hoãn nếu một số stream chưa có packet tương ứng.
- Có thể thay đổi bằng `InputStreamHandler` cho các kịch bản đặc thù (bất đồng bộ, early-close...).
- Đồ thị thời gian thực: calculators cần công bố “output timestamp bounds” dựa trên ngõ vào để scheduler có thể kích hoạt downstream sớm, giảm độ trễ end-to-end.

Tham khảo: [Framework concepts — Overview](https://ai.google.dev/edge/mediapipe/framework/framework_concepts/overview)

## 2.2. Graphs (đồ thị xử lý) {#22-graphs}

### 2.2.1. `CalculatorGraphConfig` (pbtxt)

- Mô tả topo đồ thị: danh sách `node` (calculator/subgraph), `input_stream`, `output_stream`, `input_side_packet`, `output_side_packet` và thiết lập toàn cục: executors, số luồng, queue size.

Ví dụ graph pbtxt đơn giản (chuỗi passthrough):

```pbtxt
input_stream: "in"
output_stream: "out"
node {
  calculator: "PassThroughCalculator"
  input_stream: "in"
  output_stream: "out1"
}
node {
  calculator: "PassThroughCalculator"
  input_stream: "out1"
  output_stream: "out2"
}
node {
  calculator: "PassThroughCalculator"
  input_stream: "out2"
  output_stream: "out"
}
```

### 2.2.2. Đại diện C++ cho graph

```cpp
CalculatorGraphConfig BuildGraphConfig() {
  Graph graph;
  // Inputs
  Stream<AnyType> in = graph.In(0).SetName("in");

  auto pass = [](Stream<AnyType> s, Graph& g) -> Stream<AnyType> {
    auto& n = g.AddNode("PassThroughCalculator");
    s.ConnectTo(n.In(0));
    return n.Out(0);
  };

  Stream<AnyType> out4 = pass(pass(pass(pass(in, graph), graph), graph), graph);
  out4.SetName("out").ConnectTo(graph.Out(0));
  return graph.GetConfig();
}
```

### 2.2.3. Subgraph (mô-đun hoá)

- Định nghĩa subgraph pbtxt, đăng ký `register_as`, và dùng như một calculator trong graph chính. Subgraph sẽ được “mở rộng” thành các calculators khi nạp đồ thị, giữ nguyên hiệu năng/ngữ nghĩa.

Ví dụ subgraph 2 passthrough:

```pbtxt
type: "TwoPassThroughSubgraph"
input_stream: "out1"
output_stream: "out3"
node { calculator: "PassThroughCalculator" input_stream: "out1" output_stream: "out2" }
node { calculator: "PassThroughCalculator" input_stream: "out2" output_stream: "out3" }
```

### 2.2.4. `node_options` và `option_value` (ProtoPath)

- `node_options`: đặt giá trị trực tiếp theo cú pháp protobuf text.
- `option_value: "LHS:RHS"`: ánh xạ từ “graph options” tới “calculator options” bằng đường dẫn trường (ProtoPath). Dùng để truyền tham số (ví dụ `tensor_width/height`, `delegate`, `model_path`) từ options cấp graph xuống calculators bên trong subgraph.

### 2.2.5. Cycles (vòng lặp trong đồ thị)

- Mặc định đồ thị acyclic. Nếu cần vòng lặp: (1) đánh dấu một cạnh là `back_edge` để topological sort hợp lệ; (2) cung cấp initial packet tại ngõ vào vòng lặp để node có thể chạy từ timestamp đầu; (3) chèn delay một đơn vị thời gian để căn chỉnh `sum(prev)` với `next input`.
- Có thể dùng `EarlyCloseInputStreamHandler` để calculator kết thúc sớm khi một input stream đã hoàn tất.

Ví dụ đoạn cấu hình vòng lặp (ý tưởng):

```pbtxt
node { calculator: 'IntAdderCalculator' input_stream: 'integers' input_stream: 'old_sum'
  input_stream_info: { tag_index: ':1' back_edge: true }
  output_stream: 'sum'
  input_stream_handler { input_stream_handler: 'EarlyCloseInputStreamHandler' }
}
node { calculator: 'UnitDelayCalculator' input_stream: 'sum' output_stream: 'old_sum' }
```

### 2.2.6. Executors/Threads và hiệu năng

- Gán calculators nặng (suy luận) lên executor riêng có thể cải thiện độ trễ thời gian thực nhờ locality/threading phù hợp; tinh chỉnh queue size, số luồng theo nền tảng.

Thực hành tốt: đặt tên streams/ports rõ ràng; gom nhóm bằng subgraph; cân nhắc fan-out/fan-in; theo dõi hàng đợi để phát hiện nghẽn.

Tham khảo: [Framework concepts — Graphs](https://ai.google.dev/edge/mediapipe/framework/framework_concepts/graphs)

## 2.3. Calculators (nodes) {#23-calculators}

### 2.3.1. Vai trò và phân loại

- Nguồn (source): tạo packets mà không cần input streams (đọc camera/file).
- Trung gian (transform/filter/aggregate): tiêu thụ và sinh packets.
- Đích (sink): tiêu thụ packets và ghi ra file/bộ nhớ hoặc gây side-effect (âm thanh, UI).

### 2.3.2. Giao diện (ports) và kiểu dữ liệu

- Mỗi input/output stream port có kiểu; phải khớp với kiểu packet. Ports định danh bằng tag hoặc index. Side packets là cổng riêng cho cấu hình.

### 2.3.3. Vòng đời

- `Open()`: cấp phát tài nguyên, nạp mô hình; tất cả input side packets required đã sẵn có.
- `Process()`: nhận tất cả inputs cùng timestamp (mặc định) → xử lý → phát outputs. Cập nhật output timestamp bounds để downstream không bị trễ.
- `Close()`: giải phóng tài nguyên; phát packets kết thúc nếu cần.

### 2.3.4. Input policies và InputStreamHandler

- Mặc định: deterministic collation (đồng bộ theo timestamp). Tuỳ chọn: dùng `InputStreamHandler` khác để xử lý khi một số luồng vắng packet hoặc để kết thúc sớm.

### 2.3.5. Thực hành tốt

- Tránh copy không cần thiết; tái sử dụng buffer; mở tài nguyên nặng trong `Open()`; đặt bounds đầy đủ; ghi log tối thiểu để không ảnh hưởng real-time.

Tham khảo: [Framework concepts — Calculators](https://ai.google.dev/edge/mediapipe/framework/framework_concepts/calculators)

## 2.4. Packets {#24-packets}

### 2.4.1. Cấu trúc và bất biến

- Packet = (timestamp, payload bất biến). Payload có thể là bất kỳ kiểu C++ (ảnh, tensor, landmarks...). Bất biến cho phép chia sẻ payload giữa nhiều node mà không cần khoá đối tượng.

### 2.4.2. Sao chép rẻ và re-timestamp

- Sao chép packet chỉ tăng bộ đếm tham chiếu payload; mỗi bản sao có thể gán timestamp khác (`.At(...)`) để phục vụ fan-out hoặc sắp xếp lại.

### 2.4.3. Tạo/đóng dấu và truy cập (C++)

```cpp
// Tạo packet chứa payload mới
Packet p = MakePacket<MyDataClass>("constructor_argument");
// Tạo packet mới với cùng payload nhưng timestamp khác
Packet p2 = p.At(Timestamp::PostStream());

// Hoặc dùng Adopt để chuyển quyền sở hữu con trỏ
auto data = absl::make_unique<MyDataClass>("constructor_argument");
Packet p3 = Adopt(data.release()).At(Timestamp(0));

// Truy cập dữ liệu
const MyDataClass& ref = p3.Get<MyDataClass>();
```

### 2.4.4. Semantics timestamp và lập lịch

- Timestamps tăng đơn điệu trên mỗi stream. Các mốc đặc biệt: `Unset()`, `PreStream()`, `PostStream()`. Lập lịch calculators dựa trên ràng buộc timestamp và output bounds được công bố.

### 2.4.5. Bối cảnh Python

- API Python trừu tượng hoá Packet, nhưng semantics vẫn áp dụng: mỗi frame tương ứng một timestamp; các bước downstream tuân thủ ràng buộc thời gian khi phối hợp nhiều luồng.

Tham khảo: [Framework concepts — Packets](https://ai.google.dev/edge/mediapipe/framework/framework_concepts/packets)

## 2.5. Visualizer (công cụ trực quan hóa) {#25-visualizer}

### 2.5.1. Mục đích

- Hiển thị đồ thị pbtxt/subgraph; theo dõi connections, dòng packets, hàng đợi, và thời gian.

### 2.5.2. Quy trình gợi ý

1) Viết/khai thác `CalculatorGraphConfig` (có thể từ C++ builder).
2) Mở bằng Visualizer để kiểm tra: đặt tên streams/ports, nhánh/gộp, fan-out/fan-in.
3) Phát hiện tắc nghẽn: hàng đợi dài, node không phát bounds, cycles thiếu delay/initial packet.

### 2.5.3. Trường hợp sử dụng

- Debug đồ thị phức tạp; tối ưu real-time; kiểm tra mapping `option_value` giữa graph options và calculators.

Tham khảo: [Framework tools — Visualizer](https://ai.google.dev/edge/mediapipe/framework/tools/visualizer)

---

<div style="page-break-after: always;"></div>

# CHƯƠNG 3. KIẾN TRÚC HỆ THỐNG VÀ LUỒNG XỬ LÝ {#chuong-3}

## 3.1. Mô tả tổng quan pipeline {#31-pipeline}

Camera → MediaPipe Hands (phát hiện + landmarks) → Trích chọn/chuẩn hóa đặc trưng → Nhận dạng cử chỉ (rule-based) → Ổn định theo thời gian (debounce khung hình) → TTS → Hiển thị UI.

## 3.2. Ánh xạ thành phần dự án vào khái niệm MediaPipe {#32-anh-xa}

- `src/hand_detector.py`: tương đương một calculator xử lý khung hình đầu vào, tạo streams landmarks tay (`multi_hand_landmarks`) và handedness; sử dụng `mp.solutions.hands` và vẽ connections. Cung cấp hàm lấy toạ độ pixel cho 21 landmarks/hand.
- `src/gesture_recognizer.py`: xây dựng đặc trưng từ landmarks và phân loại cử chỉ (rule-based), sinh nhãn cử chỉ và độ tin cậy.
- `src/text_to_speech.py`: sink component phát âm thanh từ chuỗi văn bản; đóng vai trò đích (sink node) trong pipeline.
- `src/main.py`: thành phần điều phối (ứng dụng) đọc khung hình, chuyển qua các “node” trên, hợp nhất kết quả, cập nhật UI, quản lý trạng thái và thời gian thực (fps, debounce, trigger TTS).

Streams tương ứng:
- Video frames (source) → Hand landmarks stream → Gesture features stream → Gesture label stream → Speech text (side-effect).

Side packets (tương đương cấu hình bất biến):
- Ngưỡng tin cậy, số khung yêu cầu để chấp nhận cử chỉ, tham số TTS (model/voice), chỉ số camera, kích thước khung hình.

## 3.3. Chính sách đồng bộ và thời gian thực {#33-realtime}

- Mỗi khung hình mang một timestamp; các bước downstream xử lý khi đủ inputs cùng timestamp (semantics của input policy mặc định).
- Ổn định thời gian: đếm số khung giữ cùng một cử chỉ trước khi phát âm (debounce) để giảm jitter; đảm bảo “output timestamp bounds” không chặn scheduler.
- Tối ưu: bật/tắt vẽ landmarks để giảm chi phí; cố định độ phân giải camera để duy trì fps ổn định.

---

<div style="page-break-after: always;"></div>

# CHƯƠNG 4. DỮ LIỆU, TIỀN XỬ LÝ VÀ NHẬN DẠNG CỬ CHỈ {#chuong-4}

## 4.1. Dữ liệu và tiền xử lý landmarks {#41-du-lieu}

- Nguồn dữ liệu: webcam, độ phân giải thiết lập trong `Config`.
- Tiền xử lý: chuyển BGR→RGB cho MediaPipe, chuẩn hóa tọa độ landmarks theo kích thước khung hình, trích bounding box hỗ trợ hiển thị.

## 4.2. Phân loại cử chỉ {#42-phan-loai}

- Cách tiếp cận: rule-based từ trạng thái ngón tay (mở/đóng) và quan hệ tương đối các khớp (ví dụ tip so với PIP). Có thể mở rộng sang mô hình ML nếu có tập huấn luyện.
- Ổn định theo thời gian: yêu cầu số khung liên tục đạt ngưỡng tin cậy trước khi xác nhận cử chỉ.

## 4.3. TTS (Text-To-Speech) và giao diện {#43-tts-ui}

- Khi cử chỉ được xác nhận, chuyển nhãn cử chỉ sang câu đọc thân thiện (ví dụ ký tự A→“Letter A”) và phát âm qua TTS.
- UI hiển thị fps, trạng thái, cử chỉ hiện tại, văn bản đã tích lũy, hướng dẫn phím tắt.

---

<div style="page-break-after: always;"></div>

# CHƯƠNG 5. THỰC NGHIỆM VÀ ĐÁNH GIÁ {#chuong-5}

## 5.1. Thiết lập môi trường {#51-thiet-lap}

- Hệ điều hành: Windows 10/11 (PowerShell).
- Ngôn ngữ: Python 3.11 (theo `venv`).
- Thư viện: OpenCV, MediaPipe, NumPy, TTS client (theo `requirements.txt`).
- Cách chạy: `python src/main.py` (thiết lập `.env` nếu dùng TTS API).

## 5.2. Chỉ số đánh giá {#52-chi-so}

- Độ trễ end-to-end và FPS.
- Tỉ lệ nhận đúng cử chỉ trong điều kiện ánh sáng/góc nhìn khác nhau.
- Tỉ lệ kích hoạt sai (false trigger) khi dùng debounce.

## 5.3. Kết quả và phân tích {#53-ket-qua}

- Hệ thống đạt thời gian thực ở độ phân giải 640×480; debounce giúp giảm kích hoạt sai nhưng tăng trễ xác nhận (~1s với 30 khung yêu cầu).
- Ánh sáng mạnh từ phía sau và che khuất làm giảm độ ổn định landmarks; có thể cải thiện bằng adaptive threshold hoặc tăng cường dữ liệu huấn luyện nếu chuyển sang ML.

---

<div style="page-break-after: always;"></div>

# CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN {#chuong-6}

## Kết luận

Đề tài triển khai thành công pipeline nhận diện cử chỉ tay thời gian thực dựa trên MediaPipe, tích hợp TTS và UI trực quan. Báo cáo đính kèm tóm lược khái niệm cốt lõi của MediaPipe và cách ánh xạ vào mã nguồn thực tế.

## Hướng phát triển

- Mở rộng bộ cử chỉ, hỗ trợ đa người/đa tay.
- Chuyển sang mô hình học máy cho phân loại cử chỉ; huấn luyện trên tập dữ liệu ký hiệu.
- Xuất cấu hình pbtxt/subgraph và sử dụng Visualizer để tối ưu hiệu năng.
- Bổ sung cơ chế ước lượng độ sâu hoặc đa camera để tăng độ bền vững.

---

<div style="page-break-after: always;"></div>

# TÀI LIỆU THAM KHẢO {#tai-lieu-tham-khao}

[1] Framework concepts — Overview. Link: https://ai.google.dev/edge/mediapipe/framework/framework_concepts/overview

[2] Framework concepts — Graphs. Link: https://ai.google.dev/edge/mediapipe/framework/framework_concepts/graphs

[3] Framework concepts — Calculators. Link: https://ai.google.dev/edge/mediapipe/framework/framework_concepts/calculators

[4] Framework concepts — Packets. Link: https://ai.google.dev/edge/mediapipe/framework/framework_concepts/packets

[5] Framework tools — Visualizer. Link: https://ai.google.dev/edge/mediapipe/framework/tools/visualizer

---

<div style="page-break-after: always;"></div>

# PHỤ LỤC {#phu-luc}

## Phụ lục A: Cấu trúc mã nguồn chính (`src/`)

- `hand_detector.py`: Khởi tạo MediaPipe Hands, tìm và vẽ landmarks, truy xuất toạ độ, nhãn tay L/R, hộp bao quanh.
- `gesture_recognizer.py`: Xây dựng luật nhận dạng cử chỉ từ landmarks, trả về (nhãn, độ tin cậy).
- `text_to_speech.py`: Lớp TTS và bộ đệm văn bản; phát âm trực tiếp hoặc theo lô.
- `main.py`: Ứng dụng chính, vòng lặp camera, xử lý frame, debounce cử chỉ, trigger TTS, UI/FPS/keyboard.

## Phụ lục B: Hướng dẫn chạy nhanh

1. Cài đặt môi trường ảo và phụ thuộc theo `requirements.txt`.
2. (Tùy chọn) Tạo file `.env` cấu hình `OPENAI_API_KEY`, `TTS_MODEL`, `TTS_VOICE`.
3. Chạy demo: `python src/main.py`.
4. Phím tắt: SPACE (dấu cách), ENTER (đọc văn bản), C (xoá), P (tạm dừng), Q (thoát).

## Phụ lục C: Gợi ý tối ưu thời gian thực

- Cố định độ phân giải khung hình; tắt bớt vẽ nếu không cần.
- Thiết lập “output timestamp bounds” đúng khi dùng calculators tuỳ biến.
- Dùng Visualizer để phát hiện nút nghẽn nếu chuyển sang graph pbtxt.


