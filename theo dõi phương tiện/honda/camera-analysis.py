"""
================================================================================
PHÂN TÍCH CAMERA GÓC CAO & TỐI ƯU HỆ THỐNG
================================================================================
Dựa trên ảnh mẫu của bạn:
- Camera gắn cao (6-10m)
- Góc nhìn: ~30-45 độ (nhìn chéo xuống)
- Quay được toàn cảnh giao lộ
- Phương tiện nhỏ ở xa, lớn hơn ở gần camera
================================================================================
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# ============================================================================
# MODULE 1: PHÂN TÍCH CAMERA & PERSPECTIVE
# ============================================================================

class CameraAnalyzer:
    """
    Phân tích góc camera và tính toán perspective transformation
    
    Từ ảnh mẫu, camera có đặc điểm:
    1. Góc cao: ~6-10m
    2. Góc nghiêng: ~30-45°
    3. Hiệu ứng perspective: Xa nhỏ - Gần lớn
    4. Biển số rất nhỏ (không đọc được)
    """
    
    def __init__(self):
        self.camera_height = 8  # meters (ước tính)
        self.tilt_angle = 40  # degrees (ước tính)
        self.perspective_matrix = None
        self.inverse_perspective_matrix = None
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Phân tích frame để xác định vùng quan trọng
        
        Với camera góc cao:
        - Vùng GẦN camera (dưới frame): Phương tiện LỚN, rõ nét
        - Vùng XA camera (trên frame): Phương tiện NHỎ, mờ
        - Vùng TRUNG TÂM: Giao lộ quan trọng nhất
        """
        h, w = frame.shape[:2]
        
        analysis = {
            "frame_size": (w, h),
            "zones": {
                # Vùng gần (bottom 30% frame) - Phương tiện lớn, dễ track
                "near_zone": {
                    "bbox": (0, int(h * 0.7), w, h),
                    "priority": "high",
                    "quality": "excellent",
                    "note": "Phương tiện lớn, tracking tốt"
                },
                # Vùng trung (middle 40% frame) - Giao lộ chính
                "intersection_zone": {
                    "bbox": (0, int(h * 0.3), w, int(h * 0.7)),
                    "priority": "critical",
                    "quality": "good",
                    "note": "Khu vực vi phạm chính"
                },
                # Vùng xa (top 30% frame) - Phương tiện nhỏ
                "far_zone": {
                    "bbox": (0, 0, w, int(h * 0.3)),
                    "priority": "low",
                    "quality": "poor",
                    "note": "Phương tiện nhỏ, khó tracking"
                }
            },
            "recommendations": {
                "min_detection_area": 500,  # pixels² - Bỏ qua xe quá nhỏ
                "min_track_length": 15,  # frames - Tăng từ 10
                "conf_threshold_near": 0.3,  # Vùng gần: threshold thấp
                "conf_threshold_far": 0.5,  # Vùng xa: threshold cao
            }
        }
        
        return analysis
    
    def calculate_perspective_transform(self, frame: np.ndarray, 
                                       roi_points: List[Tuple[int, int]]) -> np.ndarray:
        """
        Tính perspective transform để "làm phẳng" view
        
        Input: 4 điểm góc của khu vực quan tâm (ROI) trong perspective view
        Output: Ma trận transform để chuyển sang bird's eye view
        
        LƯU Ý: Bird's eye view giúp:
        - Đo khoảng cách chính xác hơn
        - Phân tích hướng di chuyển dễ hơn
        - NHƯNG: Mất thông tin phương tiện ở xa
        """
        h, w = frame.shape[:2]
        
        # Điểm nguồn (trong perspective view)
        src_points = np.float32(roi_points)
        
        # Điểm đích (rectangle trong bird's eye view)
        dst_points = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])
        
        # Tính ma trận transform
        self.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inverse_perspective_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
        
        return self.perspective_matrix
    
    def apply_perspective_transform(self, frame: np.ndarray) -> np.ndarray:
        """Áp dụng perspective transform lên frame"""
        if self.perspective_matrix is None:
            return frame
        
        h, w = frame.shape[:2]
        return cv2.warpPerspective(frame, self.perspective_matrix, (w, h))
    
    def inverse_transform_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """Chuyển điểm từ bird's eye về perspective view"""
        if self.inverse_perspective_matrix is None:
            return point
        
        pts = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.inverse_perspective_matrix)
        return tuple(transformed[0][0].astype(int))

# ============================================================================
# MODULE 2: ADAPTIVE DETECTION DỰA TRÊN VỊ TRÍ
# ============================================================================

class AdaptiveDetector:
    """
    Detector thông minh điều chỉnh theo vị trí trong frame
    
    Nguyên tắc:
    - Vùng GẦN: Confidence thấp OK, phương tiện lớn dễ detect
    - Vùng XA: Cần confidence cao, lọc noise
    - Kích thước tối thiểu tăng dần theo khoảng cách
    """
    
    def __init__(self, frame_height: int):
        self.frame_height = frame_height
        
        # Chia frame thành 3 vùng
        self.zone_boundaries = {
            "near": (int(frame_height * 0.7), frame_height),
            "middle": (int(frame_height * 0.3), int(frame_height * 0.7)),
            "far": (0, int(frame_height * 0.3))
        }
        
        # Tham số cho từng vùng
        self.zone_params = {
            "near": {
                "conf_threshold": 0.25,
                "min_area": 800,
                "iou_threshold": 0.4,
                "weight": 1.0
            },
            "middle": {
                "conf_threshold": 0.35,
                "min_area": 500,
                "iou_threshold": 0.45,
                "weight": 1.5  # Quan trọng hơn
            },
            "far": {
                "conf_threshold": 0.50,
                "min_area": 300,
                "iou_threshold": 0.5,
                "weight": 0.7  # Ít tin cậy hơn
            }
        }
    
    def get_zone(self, y_center: int) -> str:
        """Xác định vùng của phương tiện dựa trên y_center"""
        for zone_name, (y_min, y_max) in self.zone_boundaries.items():
            if y_min <= y_center < y_max:
                return zone_name
        return "middle"
    
    def should_keep_detection(self, detection: Dict) -> bool:
        """
        Quyết định có giữ detection này không
        
        Dựa trên:
        - Vị trí (zone)
        - Kích thước
        - Confidence
        """
        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox
        y_center = (y1 + y2) // 2
        area = (x2 - x1) * (y2 - y1)
        conf = detection["confidence"]
        
        # Xác định zone
        zone = self.get_zone(y_center)
        params = self.zone_params[zone]
        
        # Kiểm tra điều kiện
        if conf < params["conf_threshold"]:
            return False
        
        if area < params["min_area"]:
            return False
        
        # Lọc aspect ratio bất thường (xe bị cắt/occlusion)
        w = x2 - x1
        h = y2 - y1
        aspect_ratio = w / h if h > 0 else 0
        
        if aspect_ratio < 0.3 or aspect_ratio > 5.0:
            return False
        
        return True
    
    def get_detection_weight(self, y_center: int) -> float:
        """Trọng số của detection dựa trên vị trí"""
        zone = self.get_zone(y_center)
        return self.zone_params[zone]["weight"]

# ============================================================================
# MODULE 3: LANE CONFIGURATION CHO CAMERA GÓC CAO
# ============================================================================

class SmartLaneConfig:
    """
    Cấu hình lane thông minh cho camera góc cao
    
    Đặc điểm:
    - Lane xa (trên frame): Polygon nhỏ, hẹp
    - Lane gần (dưới frame): Polygon lớn, rộng
    - Có hiệu chỉnh perspective
    """
    
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.lanes = {}
        self.stop_lines = []
        
        # Tự động tạo cấu hình mẫu dựa trên kích thước frame
        self._generate_default_config()
    
    def _generate_default_config(self):
        """
        Tạo cấu hình lane mặc định cho camera góc cao
        
        Giả định:
        - Camera ở giữa, nhìn xuống
        - 2 làn xe máy bên trái
        - 3 làn ô tô bên phải
        - Phương tiện đi từ xa (trên) xuống gần (dưới)
        """
        w, h = self.frame_width, self.frame_height
        
        # Điểm tham chiếu
        # Vùng xa (top): hẹp, tập trung giữa
        far_left = int(w * 0.2)
        far_center = int(w * 0.5)
        far_right = int(w * 0.8)
        
        # Vùng gần (bottom): rộng hơn
        near_left = int(w * 0.05)
        near_center = int(w * 0.5)
        near_right = int(w * 0.95)
        
        # Y coordinates
        y_far = int(h * 0.2)      # Vùng xa
        y_middle = int(h * 0.5)    # Vùng giữa
        y_near = int(h * 0.85)     # Vùng gần
        
        # ===== MOTORCYCLE LANES (bên trái) =====
        self.lanes["motorcycle_lane_left"] = {
            "polygon": [
                (far_left, y_far),
                (far_left + 80, y_far),
                (near_left + 150, y_near),
                (near_left, y_near)
            ],
            "direction": "down",
            "allowed_vehicles": ["motorcycle"],
            "color": (0, 255, 255),  # Vàng
            "name": "Làn xe máy trái"
        }
        
        self.lanes["motorcycle_lane_right"] = {
            "polygon": [
                (far_left + 85, y_far),
                (far_center - 20, y_far),
                (near_left + 300, y_near),
                (near_left + 155, y_near)
            ],
            "direction": "down",
            "allowed_vehicles": ["motorcycle"],
            "color": (0, 255, 255),
            "name": "Làn xe máy phải"
        }
        
        # ===== CAR LANES (bên phải) =====
        self.lanes["car_lane_1"] = {
            "polygon": [
                (far_center - 15, y_far),
                (far_center + 60, y_far),
                (near_center + 100, y_near),
                (near_left + 305, y_near)
            ],
            "direction": "down",
            "allowed_vehicles": ["car", "bus", "truck"],
            "color": (255, 0, 0),  # Xanh dương
            "name": "Làn ô tô 1"
        }
        
        self.lanes["car_lane_2"] = {
            "polygon": [
                (far_center + 65, y_far),
                (far_center + 140, y_far),
                (near_center + 250, y_near),
                (near_center + 105, y_near)
            ],
            "direction": "down",
            "allowed_vehicles": ["car", "bus", "truck"],
            "color": (255, 0, 0),
            "name": "Làn ô tô 2"
        }
        
        self.lanes["car_lane_3"] = {
            "polygon": [
                (far_center + 145, y_far),
                (far_right, y_far),
                (near_right, y_near),
                (near_center + 255, y_near)
            ],
            "direction": "down",
            "allowed_vehicles": ["car", "bus", "truck"],
            "color": (255, 0, 0),
            "name": "Làn ô tô 3"
        }
        
        # ===== STOP LINES =====
        # Vạch dừng ở vùng giữa frame
        y_stop = int(h * 0.6)
        self.stop_lines.append({
            "name": "stop_line_main",
            "line": [(near_left, y_stop), (near_right, y_stop)],
            "color": (0, 0, 255)  # Đỏ
        })
        
        # Vạch dừng phụ (nếu có ngã tư)
        y_stop_2 = int(h * 0.45)
        self.stop_lines.append({
            "name": "stop_line_secondary",
            "line": [(near_left, y_stop_2), (near_right, y_stop_2)],
            "color": (0, 100, 255)  # Cam
        })
    
    def save_to_json(self, path: str = "lane_config.json"):
        """Lưu cấu hình ra JSON"""
        config = {
            "frame_size": [self.frame_width, self.frame_height],
            "lanes": self.lanes,
            "stop_lines": self.stop_lines,
            "notes": {
                "camera_type": "high_angle",
                "tilt_angle": "30-45 degrees",
                "notes": "Lanes được điều chỉnh cho camera góc cao với perspective distortion"
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Đã lưu cấu hình lane: {path}")
    
    def load_from_json(self, path: str = "lane_config.json"):
        """Tải cấu hình từ JSON"""
        if not Path(path).exists():
            print(f"⚠ File không tồn tại: {path}")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.lanes = config.get("lanes", {})
            self.stop_lines = config.get("stop_lines", [])
        
        print(f"✓ Đã tải cấu hình lane từ: {path}")
    
    def visualize_lanes(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ lanes lên frame để kiểm tra"""
        vis_frame = frame.copy()
        
        # Vẽ lanes
        for lane_name, lane_info in self.lanes.items():
            pts = np.array(lane_info["polygon"], np.int32)
            
            # Vẽ polygon
            cv2.polylines(vis_frame, [pts], True, lane_info["color"], 2)
            
            # Vẽ tên lane
            center = np.mean(pts, axis=0).astype(int)
            cv2.putText(vis_frame, lane_info["name"], tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, lane_info["color"], 2)
        
        # Vẽ stop lines
        for stop_line in self.stop_lines:
            pt1, pt2 = stop_line["line"]
            cv2.line(vis_frame, pt1, pt2, stop_line["color"], 3)
            
            # Vẽ tên
            mid_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2 - 10)
            cv2.putText(vis_frame, stop_line["name"], mid_point,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, stop_line["color"], 2)
        
        return vis_frame

# ============================================================================
# MODULE 4: ENHANCED TRACKING CHO CAMERA GÓC CAO
# ============================================================================

class EnhancedTracker:
    """
    Tracking cải tiến cho camera góc cao
    
    Đặc điểm:
    - Track riêng biệt cho từng zone (gần/xa)
    - Tham số IoU khác nhau cho từng zone
    - Xử lý occlusion tốt hơn
    """
    
    def __init__(self, adaptive_detector: AdaptiveDetector):
        self.adaptive_detector = adaptive_detector
        self.tracks = {}
        self.next_id = 1
        self.max_age = 30
    
    def update(self, detections: List[Dict]) -> Dict:
        """Cập nhật tracks với adaptive matching"""
        
        # 1. Tăng age cho tất cả tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]["age"] += 1
            
            # Xóa track quá cũ
            if self.tracks[track_id]["age"] > self.max_age:
                del self.tracks[track_id]
        
        # 2. Match detections với tracks
        matched_tracks = set()
        
        for det in detections:
            # Kiểm tra detection có đủ chất lượng không
            if not self.adaptive_detector.should_keep_detection(det):
                continue
            
            det_bbox = det["bbox"]
            det_center = det["center"]
            det_zone = self.adaptive_detector.get_zone(det_center[1])
            
            best_iou = 0
            best_track_id = None
            
            # Tìm track khớp nhất
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                
                # Chỉ match với track trong cùng zone (hoặc zone kế bên)
                track_zone = self.adaptive_detector.get_zone(track["center"][1])
                if not self._zones_compatible(det_zone, track_zone):
                    continue
                
                # Tính IoU
                iou = self._calculate_iou(det_bbox, track["bbox"])
                
                # Threshold IoU tùy zone
                iou_threshold = self.adaptive_detector.zone_params[det_zone]["iou_threshold"]
                
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            # Cập nhật hoặc tạo mới
            if best_track_id is not None:
                # Cập nhật track hiện tại
                self.tracks[best_track_id].update({
                    "bbox": det_bbox,
                    "center": det_center,
                    "confidence": det["confidence"],
                    "age": 0,
                    "zone": det_zone,
                    "weight": self.adaptive_detector.get_detection_weight(det_center[1])
                })
                
                self.tracks[best_track_id]["trajectory"].append(det_center)
                
                # Giới hạn trajectory
                if len(self.tracks[best_track_id]["trajectory"]) > 100:
                    self.tracks[best_track_id]["trajectory"].pop(0)
                
                matched_tracks.add(best_track_id)
            else:
                # Tạo track mới
                self.tracks[self.next_id] = {
                    "bbox": det_bbox,
                    "center": det_center,
                    "class": det["class"],
                    "confidence": det["confidence"],
                    "trajectory": [det_center],
                    "age": 0,
                    "zone": det_zone,
                    "weight": self.adaptive_detector.get_detection_weight(det_center[1]),
                    "violations": [],
                    "created_at": det_center  # Vị trí tạo track
                }
                self.next_id += 1
        
        return self.tracks
    
    def _zones_compatible(self, zone1: str, zone2: str) -> bool:
        """Kiểm tra 2 zones có thể match được không"""
        # Near có thể match với middle
        # Middle có thể match với near và far
        # Far có thể match với middle
        if zone1 == zone2:
            return True
        
        if (zone1 == "near" and zone2 == "middle") or (zone1 == "middle" and zone2 == "near"):
            return True
        
        if (zone1 == "middle" and zone2 == "far") or (zone1 == "far" and zone2 == "middle"):
            return True
        
        return False
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Tính IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

# ============================================================================
# MODULE 5: CALIBRATION TOOL CẢI TIẾN
# ============================================================================

class AdvancedCalibrationTool:
    """
    Tool calibration nâng cao với hỗ trợ camera góc cao
    
    Features:
    - Click để tạo lanes
    - Hiển thị zones (near/middle/far)
    - Test detection threshold cho từng zone
    - Export config JSON
    """
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.frame = None
        self.points = []
        self.lanes = []
        self.current_lane_type = "car"  # car hoặc motorcycle
        self.stop_lines = []
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback xử lý click chuột"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"Điểm {len(self.points)}: ({x}, {y})")
            
            # Vẽ điểm
            cv2.circle(self.frame_display, (x, y), 5, (0, 255, 0), -1)
            
            # Vẽ line nếu có >= 2 điểm
            if len(self.points) >= 2:
                cv2.line(self.frame_display, self.points[-2], self.points[-1], 
                        (0, 255, 0), 2)
            
            cv2.imshow("Calibration", self.frame_display)
    
    def calibrate(self):
        """Chạy calibration tool"""
        cap = cv2.VideoCapture(self.video_path)
        ret, self.frame = cap.read()
        
        if not ret:
            print("✗ Không thể đọc video")
            return
        
        h, w = self.frame.shape[:2]
        self.frame_display = self.frame.copy()
        
        # Vẽ zones mặc định
        self._draw_zones()
        
        print("\n" + "="*80)
        print("CALIBRATION TOOL - CAMERA GÓC CAO")
        print("="*80)
        print("Hướng dẫn:")
        print("  Click chuột trái: Đánh dấu góc lane (4 điểm, theo chiều kim đồng hồ)")
        print("  [m]: Chuyển sang chế độ Motorcycle lane")
        print("  [c]: Chuyển sang chế độ Car lane")
        print("  [n]: Hoàn thành lane hiện tại, bắt đầu lane mới")
        print("  [s]: Lưu tất cả lanes vào JSON")
        print("  [r]: Reset lane hiện tại")
        print("  [z]: Toggle hiển thị zones")
        print("  [q]: Thoát")
        print("="*80)
        print(f"Chế độ hiện tại: {self.current_lane_type.upper()}")
        print("="*80 + "\n")
        
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse_callback)
        
        show_zones = True
        
        while True:
            cv2.imshow("Calibration", self.frame_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('m'):  # Motorcycle mode
                self.current_lane_type = "motorcycle"
                print("→ Chế độ: MOTORCYCLE LANE")
            
            elif key == ord('c'):  # Car mode
                self.current_lane_type = "car"
                print("→ Chế độ: CAR LANE")
            
            elif key == ord('n'):  # Next lane
                if len(self.points) >= 4:
                    lane_data = {
                        "polygon": self.points[:4],
                        "type": self.current_lane_type,
                        "direction": "down",  # Mặc định
                        "allowed_vehicles": ["motorcycle"] if self.current_lane_type == "motorcycle" 
                                          else ["car", "bus", "truck"],
                        "color": (0, 255, 255) if self.current_lane_type == "motorcycle" 
                                else (255, 0, 0)
                    }
                    self.lanes.append(lane_data)
                    print(f"✓ Lane {len(self.lanes)} ({self.current_lane_type}): {self.points[:4]}")
                    
                    # Reset
                    self.points = []
                    self.frame_display = self.frame.copy()
                    if show_zones:
                        self._draw_zones()
                    self._draw_saved_lanes()
                else:
                    print("⚠ Cần ít nhất 4 điểm để tạo lane")
            
            elif key == ord('r'):  # Reset current lane
                self.points = []
                self.frame_display = self.frame.copy()
                if show_zones:
                    self._draw_zones()
                self._draw_saved_lanes()
                print("↻ Reset lane hiện tại")
            
            elif key == ord('z'):  # Toggle zones
                show_zones = not show_zones
                self.frame_display = self.frame.copy()
                if show_zones:
                    self._draw_zones()
                self._draw_saved_lanes()
            
            elif key == ord('s'):  # Save
                if self.points and len(self.points) >= 4:
                    lane_data = {
                        "polygon": self.points[:4],
                        "type": self.current_lane_type,
                        "direction": "down",
                        "allowed_vehicles": ["motorcycle"] if self.current_lane_type == "motorcycle" 
                                          else ["car", "bus", "truck"],
                        "color": (0, 255, 255) if self.current_lane_type == "motorcycle" 
                                else (255, 0, 0)
                    }
                    self.lanes.append(lane_data)
                
                # Lưu vào JSON
                self._save_config()
                print(f"✓ Đã lưu {len(self.lanes)} lanes vào lane_config.json")
                break
            
            elif key == ord('q'):  # Quit
                print("Thoát calibration")
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _draw_zones(self):
        """Vẽ các zones (near/middle/far) lên frame"""
        h, w = self.frame.shape[:2]
        
        # Far zone (top 30%)
        y_far = int(h * 0.3)
        cv2.line(self.frame_display, (0, y_far), (w, y_far), (100, 100, 100), 2)
        cv2.putText(self.frame_display, "FAR ZONE (low quality)", (10, y_far - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        
        # Middle zone (30-70%)
        y_middle = int(h * 0.7)
        cv2.line(self.frame_display, (0, y_middle), (w, y_middle), (150, 150, 150), 2)
        cv2.putText(self.frame_display, "MIDDLE ZONE (good quality)", (10, int(h * 0.5)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        
        # Near zone (bottom 30%)
        cv2.putText(self.frame_display, "NEAR ZONE (excellent quality)", (10, y_middle + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    def _draw_saved_lanes(self):
        """Vẽ lại các lanes đã lưu"""
        for i, lane in enumerate(self.lanes):
            pts = np.array(lane["polygon"], np.int32)
            color = lane["color"]
            cv2.polylines(self.frame_display, [pts], True, color, 2)
            
            # Vẽ số thứ tự
            center = np.mean(pts, axis=0).astype(int)
            cv2.putText(self.frame_display, f"L{i+1}", tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    def _save_config(self):
        """Lưu cấu hình vào JSON"""
        h, w = self.frame.shape[:2]
        
        # Chuyển đổi lanes sang format chuẩn
        lanes_dict = {}
        for i, lane in enumerate(self.lanes):
            lane_name = f"{lane['type']}_lane_{i+1}"
            lanes_dict[lane_name] = {
                "polygon": lane["polygon"],
                "direction": lane["direction"],
                "allowed_vehicles": lane["allowed_vehicles"],
                "color": lane["color"],
                "name": f"Làn {lane['type']} {i+1}"
            }
        
        # Stop lines mặc định
        stop_lines = [
            {
                "name": "stop_line_main",
                "line": [(0, int(h * 0.6)), (w, int(h * 0.6))],
                "color": (0, 0, 255)
            }
        ]
        
        config = {
            "frame_size": [w, h],
            "lanes": lanes_dict,
            "stop_lines": stop_lines,
            "camera_info": {
                "type": "high_angle",
                "estimated_height": "6-10m",
                "tilt_angle": "30-45 degrees",
                "notes": "Camera góc cao với perspective distortion"
            }
        }
        
        with open("lane_config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

# ============================================================================
# MODULE 6: MAIN SYSTEM TÍCH HỢP TẤT CẢ
# ============================================================================

class OptimizedTrafficSystem:
    """
    Hệ thống hoàn chỉnh được tối ưu cho camera góc cao
    
    Tích hợp:
    - Camera analysis
    - Adaptive detection
    - Enhanced tracking
    - Smart lane config
    """
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        
        # Load video để lấy thông tin
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Không thể đọc video")
        
        h, w = frame.shape[:2]
        cap.release()
        
        # Khởi tạo các module
        self.camera_analyzer = CameraAnalyzer()
        self.adaptive_detector = AdaptiveDetector(h)
        self.lane_config = SmartLaneConfig(w, h)
        self.tracker = EnhancedTracker(self.adaptive_detector)
        
        # Phân tích camera
        self.camera_info = self.camera_analyzer.analyze_frame(frame)
        
        print("\n" + "="*80)
        print("HỆ THỐNG PHÁT HIỆN VI PHẠM - TỐI ƯU CHO CAMERA GÓC CAO")
        print("="*80)
        print(f"Frame size: {w}x{h}")
        print(f"Camera zones:")
        for zone_name, zone_info in self.camera_info["zones"].items():
            print(f"  - {zone_name}: {zone_info['note']}")
        print("="*80 + "\n")
        
        # Load YOLO model
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            self.model.to('cpu')
            print("✓ YOLOv8n loaded")
        except:
            print("✗ Không thể load YOLO model")
            self.model = None
        
        # Statistics
        self.stats = {
            "total_frames": 0,
            "total_detections": 0,
            "total_tracks": 0,
            "violations": defaultdict(int)
        }
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """Phát hiện phương tiện với adaptive threshold"""
        if self.model is None:
            return []
        
        h, w = frame.shape[:2]
        detections = []
        
        # Chạy YOLO với confidence thấp (sẽ lọc sau)
        results = self.model(frame, conf=0.2, iou=0.5, verbose=False)
        
        target_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                
                if cls not in target_classes:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                detection = {
                    "bbox": (x1, y1, x2, y2),
                    "class": target_classes[cls],
                    "confidence": conf,
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2)
                }
                
                # Áp dụng adaptive filtering
                if self.adaptive_detector.should_keep_detection(detection):
                    detections.append(detection)
        
        return detections
    
    def check_violations(self, tracks: Dict) -> Dict:
        """Kiểm tra vi phạm"""
        violations = defaultdict(list)
        
        for track_id, track in tracks.items():
            trajectory = track["trajectory"]
            
            # Cần đủ điểm để phân tích
            if len(trajectory) < 15:
                continue
            
            vehicle_type = track["class"]
            current_pos = track["center"]
            
            # 1. Kiểm tra đi sai làn
            wrong_lane = self._check_wrong_lane(track_id, vehicle_type, current_pos)
            if wrong_lane:
                violations[track_id].append(wrong_lane)
                self.stats["violations"]["wrong_lane"] += 1
            
            # 2. Kiểm tra cắt vạch
            stop_line = self._check_stop_line(trajectory)
            if stop_line:
                violations[track_id].append(stop_line)
                self.stats["violations"]["stop_line"] += 1
        
        return violations
    
    def _check_wrong_lane(self, track_id: int, vehicle_type: str, position: Tuple) -> Optional[Dict]:
        """Kiểm tra đi sai làn"""
        for lane_name, lane_info in self.lane_config.lanes.items():
            if self._point_in_polygon(position, lane_info["polygon"]):
                if vehicle_type not in lane_info["allowed_vehicles"]:
                    return {
                        "type": "wrong_lane",
                        "vehicle_type": vehicle_type,
                        "lane": lane_name,
                        "position": position
                    }
        return None
    
    def _check_stop_line(self, trajectory: List[Tuple]) -> Optional[Dict]:
        """Kiểm tra cắt vạch dừng"""
        if len(trajectory) < 2:
            return None
        
        p1 = trajectory[-2]
        p2 = trajectory[-1]
        
        for stop_line in self.lane_config.stop_lines:
            line = stop_line["line"]
            if self._line_intersect(p1, p2, line[0], line[1]):
                return {
                    "type": "cross_stop_line",
                    "stop_line": stop_line["name"],
                    "position": p2
                }
        
        return None
    
    def _point_in_polygon(self, point: Tuple, polygon: List[Tuple]) -> bool:
        """Kiểm tra điểm trong polygon"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _line_intersect(self, p1: Tuple, p2: Tuple, p3: Tuple, p4: Tuple) -> bool:
        """Kiểm tra 2 line có giao nhau"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0:
            return False
        
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        
        return 0 <= ua <= 1 and 0 <= ub <= 1
    
    def visualize(self, frame: np.ndarray, tracks: Dict, violations: Dict) -> np.ndarray:
        """Vẽ visualization"""
        vis = frame.copy()
        
        # Vẽ lanes
        vis = self.lane_config.visualize_lanes(vis)
        
        # Vẽ tracks
        for track_id, track in tracks.items():
            x1, y1, x2, y2 = track["bbox"]
            center = track["center"]
            
            # Màu: Đỏ nếu vi phạm, Xanh nếu OK
            has_violation = track_id in violations and len(violations[track_id]) > 0
            color = (0, 0, 255) if has_violation else (0, 255, 0)
            
            # Vẽ bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ ID và class
            label = f"ID:{track_id} {track['class']}"
            cv2.putText(vis, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Vẽ trajectory
            trajectory = track["trajectory"]
            for i in range(1, len(trajectory)):
                cv2.line(vis, trajectory[i-1], trajectory[i], color, 2)
            
            # Vẽ vi phạm
            if has_violation:
                for v in violations[track_id]:
                    v_text = v["type"].replace("_", " ").upper()
                    cv2.putText(vis, f"VI PHAM: {v_text}", (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Vẽ info
        info_y = 30
        cv2.putText(vis, f"Tracks: {len(tracks)}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Violations: {sum(len(v) for v in violations.values())}", 
                   (10, info_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis
    
    def process_video(self, output_path: str = "output_optimized.mp4"):
        """Xử lý video chính"""
        cap = cv2.VideoCapture(self.video_path)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        import time
        start_time = time.time()
        
        print("Bắt đầu xử lý video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.stats["total_frames"] += 1
            
            # Detection
            detections = self.detect_vehicles(frame)
            self.stats["total_detections"] += len(detections)
            
            # Tracking
            tracks = self.tracker.update(detections)
            self.stats["total_tracks"] = len(tracks)
            
            # Violation check
            violations = self.check_violations(tracks)
            
            # Visualization
            vis_frame = self.visualize(frame, tracks, violations)
            
            # FPS info
            fps_current = self.stats["total_frames"] / (time.time() - start_time)
            cv2.putText(vis_frame, f"FPS: {fps_current:.1f}", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            out.write(vis_frame)
            cv2.imshow("Traffic System", vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Progress
            if self.stats["total_frames"] % 30 == 0:
                progress = self.stats["total_frames"] / total_frames * 100
                print(f"Progress: {progress:.1f}% | FPS: {fps_current:.1f} | "
                      f"Tracks: {self.stats['total_tracks']}")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Summary
        elapsed = time.time() - start_time
        print("\n" + "="*80)
        print("HOÀN THÀNH")
        print("="*80)
        print(f"Frames: {self.stats['total_frames']}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Avg FPS: {self.stats['total_frames']/elapsed:.2f}")
        print(f"Total detections: {self.stats['total_detections']}")
        print(f"Violations:")
        for v_type, count in self.stats['violations'].items():
            print(f"  - {v_type}: {count}")
        print(f"Output: {output_path}")
        print("="*80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Chương trình chính"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Hệ thống phát hiện vi phạm - Tối ưu camera góc cao'
    )
    parser.add_argument('--mode', type=str, default='detect',
                       choices=['calibrate', 'detect', 'test_config'],
                       help='Chế độ chạy')
    parser.add_argument('--video', type=str, default='video.mp4',
                       help='Đường dẫn video')
    parser.add_argument('--output', type=str, default='output_optimized.mp4',
                       help='Video output')
    
    args = parser.parse_args()
    
    if args.mode == 'calibrate':
        # Chạy calibration tool
        print("Bắt đầu Calibration Tool...")
        tool = AdvancedCalibrationTool(args.video)
        tool.calibrate()
    
    elif args.mode == 'test_config':
        # Test visualization của config
        print("Test lane configuration...")
        cap = cv2.VideoCapture(args.video)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            h, w = frame.shape[:2]
            config = SmartLaneConfig(w, h)
            
            # Thử load config nếu có
            if Path("lane_config.json").exists():
                config.load_from_json()
            
            vis = config.visualize_lanes(frame)
            cv2.imshow("Lane Configuration", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    else:
        # Chạy detection
        print("Bắt đầu phát hiện vi phạm...")
        system = OptimizedTrafficSystem(args.video)
        system.process_video(args.output)

if __name__ == "__main__":
    main()
