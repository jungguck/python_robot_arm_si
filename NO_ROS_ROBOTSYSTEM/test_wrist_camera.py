#!/usr/bin/env python3
"""
test_wrist_camera.py - 손목 카메라 bolt_head / bolt_shaft 검출 테스트

[실행]
  conda activate trossen
  cd /home/trossen/jk_ws/src/NO_ROS_ROBOTSYSTEM
  python test_wrist_camera.py

[확인 항목]
  - bolt_head(초록), bolt_shaft(청록) 박스가 잘 그려지는지
  - shaft→head 방향 각도가 올바른지 (head가 12시 = 0°)
  - shaft 락이 유지되는지
  - q 키로 종료
"""
import sys
import math
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import pyrealsense2 as rs

sys.path.insert(0, '/home/trossen/RT-DETR/rtdetrv2_pytorch')
sys.path.insert(0, '/home/trossen/RT-DETR/rtdetrv2_pytorch/src')
from src.core import YAMLConfig
import src.zoo  # noqa

# ── 설정 ───────────────────────────────────────────────────────────
WRIST_CONFIG = '/home/trossen/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_bolt_head_shaft.yml'
WRIST_CKPT   = '/home/trossen/RT-DETR/rtdetrv2_pytorch/output/rtdetrv2_r18vd_bolt_head_shaft/best.pth'
SERIAL       = '230322274740'
CONF_THRESH  = 0.9
INPUT_SIZE   = (640, 640)
LOCK_RADIUS  = 80
LOCK_MISS_MAX = 15

# ── 손목 카메라 캘리브레이션 데이터 (MuJoCo 스캔 포즈에서 측정) ──
CAM_POS = np.array([-0.287903504167727, -0.01521966546096749, 0.2957348288931705])
CAM_ROT = np.array([
    [-0.021590975726096084, -0.9923771018982799, 0.12133226032332137], 
    [0.9997668877129284, -0.021431385837587025, 0.002620292709859584], 
    [1.3877787807814457e-17, 0.12136055095891567, 0.9926084911338147]
])
BOLT_Z = 0.03  # 볼트가 놓인 테이블 높이 (m)
CALIB_OFFSET = np.array([0.0, 0.0])  # 필요시 보정값 (x, y)


def load_model(config_path, ckpt_path, device):
    cfg  = YAMLConfig(config_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('ema', {}).get('module') or ckpt.get('model') or ckpt
    model = cfg.model
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    post = cfg.postprocessor
    post.eval().to(device)
    return model, post


def infer(model, post, frame, device):
    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img  = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    img  = TF.resize(img, list(INPUT_SIZE))
    img_t = img.unsqueeze(0).to(device)
    orig  = torch.tensor([[w, h]], dtype=torch.float32, device=device)
    with torch.no_grad():
        results = post(model(img_t), orig)
    r = results[0]
    return r['labels'].cpu().numpy(), r['boxes'].cpu().numpy(), r['scores'].cpu().numpy()


def pixel_to_3d(px, py, intrinsics):
    """
    픽셀 좌표 → 로봇 베이스 3D 좌표 변환
    
    [원리]
    1. 픽셀 → 카메라 좌표계 광선 방향 계산
    2. 광선과 테이블 평면(z=BOLT_Z) 교점 계산
    3. 카메라 좌표 → 로봇 베이스 좌표 변환
    
    Args:
        px, py: 픽셀 좌표 (이미지 중심)
        intrinsics: RealSense 카메라 내부 파라미터
    
    Returns:
        np.array([x, y, z]): 로봇 베이스 좌표계 위치 (m)
        None: 교점 없음 (광선이 테이블과 평행)
    """
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy
    
    # 1. 픽셀 → 카메라 좌표계 정규화 방향 벡터
    d_cam = np.array([
        (px - cx) / fx,   # x 방향
        -(py - cy) / fy,  # y 방향 (이미지 좌표계는 아래가 +)
        -1.0              # z 방향 (카메라 앞쪽)
    ])
    
    # 2. 카메라 좌표 → 월드 좌표 회전
    d_world = CAM_ROT @ d_cam
    
    # 3. 광선-평면 교점 계산: CAM_POS + t * d_world = (x, y, BOLT_Z)
    #    → t = (BOLT_Z - CAM_POS[2]) / d_world[2]
    if abs(d_world[2]) < 1e-6:
        return None  # 광선이 테이블과 평행
    
    t = (BOLT_Z - CAM_POS[2]) / d_world[2]
    if t < 0:
        return None  # 교점이 카메라 뒤쪽
    
    # 4. 3D 위치 계산
    pos = CAM_POS + t * d_world
    
    # 5. 보정값 적용 (필요시)
    pos[0] += CALIB_OFFSET[0]
    pos[1] += CALIB_OFFSET[1]
    
    return pos


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[WRIST] Device: {device}')
    print('[WRIST] 모델 로딩...')
    model, post = load_model(WRIST_CONFIG, WRIST_CKPT, device)
    print('[WRIST] 모델 로드 완료')

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(SERIAL)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(cfg)

    align        = rs.align(rs.stream.color)
    spatial      = rs.spatial_filter()
    temporal     = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    # 카메라 내부 파라미터 가져오기
    intrinsics = rs.video_stream_profile(
        profile.get_stream(rs.stream.color)).get_intrinsics()
    
    print(f'[WRIST] 카메라 연결 완료 (S/N: {SERIAL})')
    print(f'[WRIST] 내부 파라미터: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, cx={intrinsics.ppx:.1f}, cy={intrinsics.ppy:.1f}')
    print('[WRIST] q 키로 종료')

    locked_shaft = None
    lock_miss    = 0

    try:
        while True:
            frames        = pipeline.wait_for_frames()
            aligned       = align.process(frames)
            color_frame   = aligned.get_color_frame()
            depth_frame   = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            frame = np.asanyarray(color_frame.get_data())
            vis   = frame.copy()

            # Depth 컬러맵
            depth_raw      = np.asanyarray(depth_frame.get_data())
            depth_clipped  = np.clip(depth_raw, 0, 1000).astype(np.float32)
            depth_colormap = cv2.applyColorMap(
                (depth_clipped / 1000 * 255).astype(np.uint8),
                cv2.COLORMAP_INFERNO
            )

            labels, boxes, scores = infer(model, post, frame, device)

            shafts, heads = [], []
            for label, box, score in zip(labels, boxes, scores):
                if score < CONF_THRESH:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if int(label) == 1:   # shaft
                    shafts.append((cx, cy, float(score), (x1, y1, x2, y2)))
                elif int(label) == 0: # head
                    heads.append((cx, cy, (x1, y1, x2, y2)))

            # shaft 락 로직
            best_shaft = None
            if shafts:
                if locked_shaft is None:
                    best = max(shafts, key=lambda p: p[2])
                    locked_shaft = (best[0], best[1])
                    best_shaft   = locked_shaft
                    lock_miss    = 0
                else:
                    lx, ly = locked_shaft
                    nearby = [s for s in shafts if math.hypot(s[0]-lx, s[1]-ly) < LOCK_RADIUS]
                    if nearby:
                        s = min(nearby, key=lambda p: math.hypot(p[0]-lx, p[1]-ly))
                        best_shaft   = (s[0], s[1])
                        locked_shaft = (lx*0.7 + s[0]*0.3, ly*0.7 + s[1]*0.3)
                        lock_miss    = 0
                    else:
                        lock_miss += 1
                        if lock_miss >= LOCK_MISS_MAX:
                            locked_shaft = None
                            lock_miss    = 0
            else:
                if locked_shaft is not None:
                    lock_miss += 1
                    if lock_miss >= LOCK_MISS_MAX:
                        locked_shaft = None
                        lock_miss    = 0

            # 모든 shaft 그리기
            for scx, scy, sc, (x1,y1,x2,y2) in shafts:
                is_best = best_shaft is not None and (scx, scy) == best_shaft
                color = (0, 220, 220) if is_best else (60, 60, 60)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.circle(vis, (scx, scy), 5, color, -1)
                cv2.putText(vis, f'shaft {sc:.2f}', (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # 모든 head 그리기
            for hcx, hcy, (x1,y1,x2,y2) in heads:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(vis, (hcx, hcy), 5, (0, 255, 0), -1)
                cv2.putText(vis, 'head', (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # 락 범위 원
            if locked_shaft is not None:
                lx, ly = int(locked_shaft[0]), int(locked_shaft[1])
                cv2.circle(vis, (lx, ly), LOCK_RADIUS, (200, 100, 50), 1)

            # 각도 계산 및 표시
            if best_shaft is not None and heads:
                scx, scy = best_shaft
                hcx, hcy, _ = min(heads, key=lambda p: math.hypot(p[0]-scx, p[1]-scy))
                cv2.line(vis, (int(scx), int(scy)), (hcx, hcy), (0, 200, 255), 2)
                dx = hcx - scx
                dy = hcy - scy
                angle = (math.degrees(math.atan2(dx, -dy)) + 360) % 360
                
                # 3D 위치 계산 (shaft 중심 기준)
                pos_3d = pixel_to_3d(scx, scy, intrinsics)
                
                if pos_3d is not None:
                    # 화면 표시
                    cv2.putText(vis, f'angle: {angle:.1f} deg', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                    cv2.putText(vis, f'3D: X={pos_3d[0]:.3f} Y={pos_3d[1]:.3f} Z={pos_3d[2]:.3f}', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    lock_status = 'LOCKED' if locked_shaft else 'FREE'
                    cv2.putText(vis, f'[{lock_status}] miss:{lock_miss}', (10, 88),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                    
                    # 터미널 출력
                    print(f'\r[WRIST] angle={angle:.1f}°  3D=({pos_3d[0]:.3f}, {pos_3d[1]:.3f}, {pos_3d[2]:.3f})  lock={lock_status}   ',
                          end='', flush=True)
                else:
                    cv2.putText(vis, f'angle: {angle:.1f} deg (3D: FAIL)', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            else:
                cv2.putText(vis, 'NO HEAD+SHAFT', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            combined = np.hstack([vis, depth_colormap])
            cv2.imshow('RT-DETR Detection | Depth (q: quit)', combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print()
        pipeline.stop()
        cv2.destroyAllWindows()
        print('[WRIST] 종료')


if __name__ == '__main__':
    main()
