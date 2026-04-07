# 🗼 SentryBeacon: Traffic Vision System
> **"Smart Vision for Safer Roads – Drive with care, someone is waiting for you."** 🛡️

![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Computer Vision](https://img.shields.io/badge/Focus-Computer%20Vision-orange.svg)

## 📝 Giới thiệu chung (Overview)
**SentryBeacon** là một hệ thống giám sát giao thông thông minh dựa trên kỹ thuật Thị giác máy tính (Computer Vision). Dự án tập trung vào việc tự động hóa quá trình nhận diện, phân loại phương tiện và hỗ trợ giám sát an toàn đường bộ, giúp giảm thiểu sai sót do yếu tố con người.

## 🚀 Các tính năng cơ bản (Key Features)
- **Vehicle Detection:** Nhận diện đa dạng các loại phương tiện (Ô tô, xe máy, xe tải, xe buýt) trong thời gian thực.
- **License Plate Recognition (LPR):** Tự động tách vùng biển số và nhận diện ký tự (OCR) với độ chính xác cao.
- **Traffic Flow Analysis:** Đếm số lượng phương tiện và phân tích mật độ giao thông theo làn đường.
- **Safety Alerts:** Cảnh báo các hành vi vi phạm cơ bản hoặc khu vực có nguy cơ mất an toàn.

## 🧠 Khái niệm chính (Core Concepts)

### 1. Object Detection (Phát hiện vật thể)
Sử dụng các mô hình học sâu (Deep Learning) như **YOLO** để xác định vị trí (Bounding Box) của phương tiện trong từng khung hình từ camera.

### 2. Character Segmentation (Phân đoạn ký tự)
Kỹ thuật tách từng chữ cái và con số từ vùng biển số xe. Chúng tôi sử dụng các thuật toán xử lý ảnh để loại bỏ nhiễu và căn chỉnh ký tự trước khi đưa vào bộ nhận diện.

### 3. IoU & Overlap Removal (Xử lý chồng lấp)
Sử dụng chỉ số **Intersection over Union (IoU)** để loại bỏ các vùng nhận diện dư thừa, đảm bảo mỗi phương tiện/biển số chỉ được nhận diện một lần duy nhất với độ tin cậy cao nhất.

### 4. Optical Character Recognition (OCR)
Chuyển đổi hình ảnh ký tự sau khi phân đoạn thành dữ liệu văn bản (String) để lưu trữ vào cơ sở dữ liệu.

## 🛠️ Công nghệ sử dụng (Tech Stack)
* **Ngôn ngữ:** Python, C++
* **Thư viện chính:** OpenCV, PyTorch/TensorFlow, Flask (cho Dashboard quản lý).
* **Công cụ:** Jupyter Notebook (phân tích dữ liệu), Git LFS (lưu trữ model nặng).

## ❤️ Thông điệp từ SentryBeacon
> *Chúng tôi tin rằng công nghệ có thể làm cho con đường về nhà của mỗi người trở nên an toàn hơn. Hãy luôn giữ vững tay lái và tuân thủ luật lệ giao thông.*
## 🔗 Liên kết nhanh (Quick Links)
Dưới đây là các tài nguyên chính của dự án:

* **[🌐 Demo Website](https://your-link-here.com)**: Trải nghiệm hệ thống trực tuyến (Dashboard).

---
© 2026 **SentryBeacon Team** - Developed by [Nguyen Duc Manh](https://github.com/ducmanh-jr)
