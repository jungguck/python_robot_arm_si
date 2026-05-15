#!/usr/bin/env python3
"""
sam2_bolt_detector.py - SAM 2 기반 볼트 검출 + 3D 좌표 변환

[파이프라인]
1. 스캔 포즈로 이동
2. RealSense D405 손목 카메라로 이미지 캡처
3. SAM 2로 볼트 세그멘테이션
4. 픽셀 좌표 → 로봇 베이스 3D 좌표 변환

[실행]
  conda activate trossen
  cd /home/trossen/jk_ws/src/NO_ROS_ROBOTSYSTEM
  python sam2_bolt_detector.py

[의존성 설치]
  pip install opencv-python numpy torch pyrealsense2
  pip install git+https://github.com/facebookresearch/segment-anything-2.git
"""

import sys
import os
import cv2
import numpy as np
import torch
import pyrealsense2 as rs

# SAM 2 import
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as e:
    print(f"❌ SAM 2 로드 실패: {e}")
    print("\nSAM 2 설치:")
    print("  pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    sys.exit(1)

# ── 설정 ───────────────────────────────────────────────────────────
SERIAL = '230322274740'  # RealSense D405 시리얼 번호
RESOLUTION = (640, 480)

# SAM 2 체크포인트 (tiny 모델 - 빠름)
SAM2_CHECKPOINT = "/home/trossen/segment-anything-2/checkpoints/sam2_hiera_tiny.pt"
SAM2_CONFIG = "sam2_hiera_t.yaml"

# 손목 카메라 캘리브레이션
CAM_POS = np.array([-0.287903504167727, -0.01521966546096749, 0.2957348288931705])
CAM_ROT = np.array([
    [-0.021590975726096084, -0.9923771018982799, 0.12133226032332137], 
    [0.9997668877129284, -0.021431385837587025, 0.002620292709859584], 
    [1.3877787807814457e-17, 0.12136055095891567, 0.9926084911338147]
])
BOLT_Z = 0.03  # 볼트 테이블 높이 (m)
CALIB_OFFSET = np.array([0.0, 0.0])

# 검출 파라미터
MIN_MASK_AREA = 300      # 최소 마스크 면적 (픽셀)
MAX_MASK_AREA = 50000    # 최대 마스크 면적
MIN_CIRCULARITY = 0.5    # 최소 원형도 (0~1)


def pixel_to_3d(px, py, intrinsics):
    """픽셀 좌표 → 로봇 베이스 3D 좌표 변환"""
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy
    
    d_cam = np.array([
        (px - cx) / fx,
        -(py - cy) / fy,
        -1.0
    ])
    
    d_world = CAM_ROT @ d_cam
    
    if abs(d_world[2]) < 1e-6:
        return None
    
    t = (BOLT_Z - CAM_POS[2]) / d_world[2]
    if t < 0:
        return None
    
    pos = CAM_POS + t * d_world
    pos[0] += CALIB_OFFSET[0]
    pos[1] += CALIB_OFFSET[1]
    
    return pos


def load_sam2(device):
    """SAM 2 모델 로드"""
    print("[SAM2] 모델 로딩...")
    
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        min_mask_region_area=MIN_MASK_AREA,
    )
    
    print("[SAM2] 로드 완료")
    return mask_generator


def filter_bolt_masks(masks):
    """
    SAM 2 마스크를 볼트 형태 기준으로 필터링
    - 원형도 체크
    - 적절한 크기 범위
    """
    bolt_masks = []
    
    for mask_data in masks:
        area = mask_data['area']
        
        # 크기 필터링
        if area < MIN_MASK_AREA or area > MAX_MASK_AREA:
            continue
        
        # 원형도 계산
        segmentation = mask_data['segmentation'].astype(np.uint8)
        contours, _ = cv2.findContours(
            segmentation,
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
        
        # 볼트는 원형
        if circularity > MIN_CIRCULARITY:
            bolt_masks.append(mask_data)
    
    return bolt_masks


def visualize_results(image, bolt_masks, intrinsics):
    """검출 결과 시각화"""
    vis = image.copy()
    
    for i, bolt in enumerate(bolt_masks):
        mask = bolt['segmentation']
        bbox = bolt['bbox']  # [x, y, w, h]
        x, y, w, h = bbox
        
        # 마스크 오버레이
        color = np.random.randint(0, 255, 3).tolist()
        overlay = vis.copy()
        overlay[mask] = color
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        # 바운딩 박스
        cv2.rectangle(vis, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
        
        # 중심점
        cx, cy = int(x + w/2), int(y + h/2)
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        
        # 3D 좌표 계산
        pos_3d = pixel_to_3d(cx, cy, intrinsics)
        if pos_3d is not None:
            text = f"#{i+1} ({pos_3d[0]:.3f}, {pos_3d[1]:.3f}, {pos_3d[2]:.3f})"
            cv2.putText(vis, text, (int(x), int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            print(f"  볼트 #{i+1}: 픽셀=({cx}, {cy}), 3D={pos_3d.round(3)}, "
                  f"면적={bolt['area']}, 원형도={bolt['circularity']:.2f}")
    
    return vis


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    
    # SAM 2 로드
    mask_generator = load_sam2(device)
    
    # RealSense 카메라 초기화
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(SERIAL)
    config.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, 30)
    profile = pipeline.start(config)
    
    intrinsics = rs.video_stream_profile(
        profile.get_stream(rs.stream.color)).get_intrinsics()
    
    print(f"[INFO] 카메라 연결 완료 (S/N: {SERIAL})")
    print(f"[INFO] 내부 파라미터: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
    print("[INFO] 's' 키: 캡처 & 검출, 'q' 키: 종료")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            image = np.asanyarray(color_frame.get_data())
            cv2.imshow('Wrist Camera (s: detect, q: quit)', image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("\n[DETECT] 검출 시작...")
                
                # SAM 2 세그멘테이션
                print("[SAM2] 세그멘테이션 중...")
                masks = mask_generator.generate(image)
                print(f"[SAM2] 총 {len(masks)}개 마스크 생성")
                
                # 볼트 필터링
                print("[FILTER] 볼트 필터링 중...")
                bolt_masks = filter_bolt_masks(masks)
                print(f"[FILTER] {len(bolt_masks)}개 볼트 검출")
                
                # 결과 시각화
                if len(bolt_masks) > 0:
                    vis = visualize_results(image, bolt_masks, intrinsics)
                    cv2.imshow('Detection Result', vis)
                    cv2.imwrite('bolt_detection_sam2.png', vis)
                    print("[INFO] 결과 저장: bolt_detection_sam2.png")
                else:
                    print("[WARN] 볼트를 찾지 못했습니다.")
                    print(f"[DEBUG] 전체 마스크 정보 (처음 10개):")
                    for i, m in enumerate(masks[:10]):
                        print(f"  마스크 {i}: 면적={m['area']}")
                
                print("[DETECT] 완료\n")
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] 종료")


if __name__ == '__main__':
    main()
