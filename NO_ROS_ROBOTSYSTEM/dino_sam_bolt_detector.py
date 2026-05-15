#!/usr/bin/env python3
"""
dino_sam_bolt_detector.py - DINOv2 + MobileSAM 기반 볼트 검출 + 3D 좌표 변환

[파이프라인]
1. 스캔 포즈로 이동 (go_to_scan_pose.py 실행 후)
2. RealSense D405 손목 카메라로 이미지 캡처
3. DINOv2 (ViT-S)로 볼트 영역 feature 추출
4. MobileSAM으로 볼트 세그멘테이션
5. 픽셀 좌표 → 로봇 베이스 3D 좌표 변환

[실행]
  conda activate trossen
  cd /home/trossen/jk_ws/src/NO_ROS_ROBOTSYSTEM
  python dino_sam_bolt_detector.py

[의존성 설치]
  pip install timm transformers opencv-python pyrealsense2
"""

import sys
import os

# 필수 패키지 확인
try:
    import cv2
    import numpy as np
    import torch
    import pyrealsense2 as rs
    from PIL import Image
except ImportError as e:
    print(f"❌ 필수 패키지 누락: {e}")
    print("다음 명령으로 설치하세요:")
    print("  pip install opencv-python numpy torch pyrealsense2 pillow")
    sys.exit(1)

# MobileSAM 경로 추가
sys.path.insert(0, '/home/trossen/MobileSAM')

try:
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
except ImportError as e:
    print(f"❌ MobileSAM 로드 실패: {e}")
    print("\n필요한 패키지를 설치하세요:")
    print("  pip install timm")
    print("\nMobileSAM 경로 확인:")
    print("  ls /home/trossen/MobileSAM/")
    sys.exit(1)

# Transformers (DINOv2)
try:
    from transformers import AutoImageProcessor, AutoModel
except ImportError:
    print("❌ transformers 패키지 누락")
    print("  pip install transformers")
    sys.exit(1)

# ── 설정 ───────────────────────────────────────────────────────────
SERIAL = '230322274740'  # RealSense D405 시리얼 번호
RESOLUTION = (640, 480)

# DINOv2 모델 (ViT-S)
DINO_MODEL = "facebook/dinov2-small"

# MobileSAM 체크포인트
MOBILESAM_CKPT = "/home/trossen/MobileSAM/weights/mobile_sam.pt"

# 손목 카메라 캘리브레이션 (MuJoCo 스캔 포즈에서 측정)
CAM_POS = np.array([-0.287903504167727, -0.01521966546096749, 0.2957348288931705])
CAM_ROT = np.array([
    [-0.021590975726096084, -0.9923771018982799, 0.12133226032332137], 
    [0.9997668877129284, -0.021431385837587025, 0.002620292709859584], 
    [1.3877787807814457e-17, 0.12136055095891567, 0.9926084911338147]
])
BOLT_Z = 0.03  # 볼트 테이블 높이 (m)
CALIB_OFFSET = np.array([0.0, 0.0])

# 검출 파라미터
MIN_MASK_AREA = 500   # 최소 마스크 면적 (픽셀)
MAX_MASK_AREA = 50000 # 최대 마스크 면적
CONF_THRESH = 0.5     # 신뢰도 임계값


def pixel_to_3d(px, py, depth_value, intrinsics):
    """
    픽셀 좌표 + Depth → 로봇 베이스 3D 좌표 변환
    
    Args:
        px, py: 픽셀 좌표
        depth_value: depth 값 (mm)
        intrinsics: RealSense 카메라 내부 파라미터
    
    Returns:
        np.array([x, y, z]): 로봇 베이스 좌표 (m)
    """
    if depth_value == 0:
        return None
    
    # RealSense deproject: 픽셀 + depth → 카메라 좌표계 3D
    point_cam = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth_value)
    point_cam = np.array(point_cam) / 1000.0  # mm → m
    
    # 카메라 좌표계 → 로봇 베이스 좌표계
    # point_cam: [x_cam, y_cam, z_cam]
    # 카메라 회전 적용
    point_world = CAM_ROT @ point_cam + CAM_POS
    point_world[0] += CALIB_OFFSET[0]
    point_world[1] += CALIB_OFFSET[1]
    
    return point_world


def load_models(device):
    """MobileSAM 모델 로드"""
    print("[SAM] MobileSAM 모델 로딩...")
    sam = sam_model_registry["vit_t"](checkpoint=MOBILESAM_CKPT)
    sam.to(device)
    sam.eval()
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=MIN_MASK_AREA,
    )
    print("[SAM] MobileSAM 로드 완료")
    
    return mask_generator





def filter_bolt_masks(masks):
    """
    SAM 마스크를 볼트 형태 기준으로 필터링
    - 원형도 (circularity) 체크
    - 적절한 크기 범위
    """
    bolt_masks = []
    for mask_data in masks:
        area = mask_data['area']
        bbox = mask_data['bbox']  # [x, y, w, h]
        
        # 크기 필터링
        if area < MIN_MASK_AREA or area > MAX_MASK_AREA:
            continue
        
        # 원형도 계산 (4π*면적/둘레²)
        # 완전한 원 = 1.0, 길쭉한 형태 = 0에 가까움
        contours, _ = cv2.findContours(
            mask_data['segmentation'].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            continue
        
        perimeter = cv2.arcLength(contours[0], True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter ** 2)
        mask_data['circularity'] = circularity
        
        # 볼트는 대체로 원형 (0.6 이상)
        if circularity > 0.6:
            bolt_masks.append(mask_data)
    
    return bolt_masks


def visualize_results(image, depth_frame, bolt_masks, intrinsics):
    """검출 결과 시각화"""
    vis = image.copy()
    depth_image = np.asanyarray(depth_frame.get_data())
    
    for i, bolt in enumerate(bolt_masks):
        mask = bolt['segmentation']
        bbox = bolt['bbox']  # [x, y, w, h]
        x, y, w, h = bbox
        
        # 마스크 오버레이 (반투명)
        color = np.random.randint(0, 255, 3).tolist()
        overlay = vis.copy()
        overlay[mask] = color
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        # 바운딩 박스
        cv2.rectangle(vis, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
        
        # 중심점
        cx, cy = int(x + w/2), int(y + h/2)
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        
        # Depth 값 가져오기
        depth_value = depth_image[cy, cx]
        
        # 3D 좌표 계산
        pos_3d = pixel_to_3d(cx, cy, depth_value, intrinsics)
        if pos_3d is not None:
            text = f"#{i+1} 3D: ({pos_3d[0]:.3f}, {pos_3d[1]:.3f}, {pos_3d[2]:.3f})"
            cv2.putText(vis, text, (int(x), int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            print(f"  볼트 #{i+1}: 픽셀=({cx}, {cy}), depth={depth_value}mm, 3D={pos_3d.round(3)}, "
                  f"면적={bolt['area']}, 원형도={bolt['circularity']:.2f}")
        else:
            print(f"  볼트 #{i+1}: depth 값 없음 (depth={depth_value}mm)")
    
    return vis


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    
    # 모델 로드
    mask_generator = load_models(device)
    
    # RealSense 카메라 초기화
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(SERIAL)
    config.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, RESOLUTION[0], RESOLUTION[1], rs.format.z16, 30)
    profile = pipeline.start(config)
    
    # Align depth to color
    align = rs.align(rs.stream.color)

    # Depth 노이즈 감소 필터
    spatial  = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()
    
    intrinsics = rs.video_stream_profile(
        profile.get_stream(rs.stream.color)).get_intrinsics()
    
    print(f"[INFO] 카메라 연결 완료 (S/N: {SERIAL})")
    print(f"[INFO] 내부 파라미터: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
    print("[INFO] 's' 키: 캡처 & 검출, 'q' 키: 종료")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            
            # Align depth to color
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            # 필터 적용 (노이즈 감소)
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)
            
            if not color_frame or not depth_frame:
                continue
            
            image = np.asanyarray(color_frame.get_data())

            # Depth 시각화 (0~1000mm 범위 클리핑 후 컬러맵)
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_clipped = np.clip(depth_raw, 0, 1000).astype(np.float32)
            depth_norm = (depth_clipped / 1000 * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

            combined = np.hstack([image, depth_colormap])
            cv2.imshow('RGB | Depth (s: detect, q: quit)', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("\n[DETECT] 검출 시작...")
                
                # 1. MobileSAM 세그멘테이션
                print("[SAM] 세그멘테이션 중...")
                masks = mask_generator.generate(image)
                print(f"[SAM] 총 {len(masks)}개 마스크 생성")
                
                # 2. 볼트 마스크 필터링
                print("[FILTER] 볼트 필터링 중...")
                bolt_masks = filter_bolt_masks(masks)
                print(f"[FILTER] {len(bolt_masks)}개 볼트 검출")
                
                # 3. 결과 시각화
                if len(bolt_masks) > 0:
                    vis = visualize_results(image, depth_frame, bolt_masks, intrinsics)
                    cv2.imshow('Detection Result', vis)
                    cv2.imwrite('bolt_detection_result.png', vis)
                    print("[INFO] 결과 저장: bolt_detection_result.png")
                else:
                    print("[WARN] 볼트를 찾지 못했습니다.")
                    # 디버그: 모든 마스크 보기
                    print(f"[DEBUG] 전체 마스크 정보:")
                    for i, m in enumerate(masks[:10]):  # 처음 10개만
                        print(f"  마스크 {i}: 면적={m['area']}, bbox={m['bbox']}")
                
                print("[DETECT] 완료\n")
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] 종료")


if __name__ == '__main__':
    main()
