#!/usr/bin/env python3
"""
test_top_camera.py - 탑뷰 카메라 볼트 검출 테스트

[실행]
  conda activate trossen
  cd /home/trossen/jk_ws/src/NO_ROS_ROBOTSYSTEM
  python test_top_camera.py

[확인 항목]
  - 볼트 바운딩 박스가 잘 그려지는지
  - 3D 위치 변환값이 실제 위치와 근접한지
  - q 키로 종료
"""
import sys
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
TOP_CONFIG  = '/home/trossen/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_bolt.yml'
TOP_CKPT    = '/home/trossen/RT-DETR/rtdetrv2_pytorch/output/rtdetrv2_r18vd_bolt/best.pth'
SERIAL      = '412622271120'
CONF_THRESH = 0.9
INPUT_SIZE  = (640, 640)

CAM_POS      = np.array([-0.50, 0.0, 0.50])
CAM_ROT      = np.array([[ 0,  1,  0], [-1,  0,  0], [ 0,  0,  1]])
BOLT_Z       = 0.03
CALIB_OFFSET = np.array([-0.040, 0.036])


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
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy
    d_cam   = np.array([(px - cx) / fx, -(py - cy) / fy, -1.0])
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


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[TOP] Device: {device}')
    print('[TOP] 모델 로딩...')
    model, post = load_model(TOP_CONFIG, TOP_CKPT, device)
    print('[TOP] 모델 로드 완료')

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(SERIAL)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    intrinsics = rs.video_stream_profile(
        profile.get_stream(rs.stream.color)).get_intrinsics()
    print(f'[TOP] 카메라 연결 완료 (S/N: {SERIAL})')
    print('[TOP] q 키로 종료')

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            vis   = frame.copy()

            labels, boxes, scores = infer(model, post, frame, device)

            best_score, best_box = 0.0, None
            for label, box, score in zip(labels, boxes, scores):
                if score < CONF_THRESH:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f'{score:.2f}', (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if score > best_score:
                    best_score, best_box = score, box

            if best_box is not None:
                cx = (best_box[0] + best_box[2]) / 2
                cy = (best_box[1] + best_box[3]) / 2
                cv2.circle(vis, (int(cx), int(cy)), 6, (0, 0, 255), -1)

                pos = pixel_to_3d(cx, cy, intrinsics)
                if pos is not None:
                    txt = f'X:{pos[0]:.3f}  Y:{pos[1]:.3f}  Z:{pos[2]:.3f}'
                    cv2.putText(vis, txt, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                    print(f'\r[TOP] bolt 3D = {[round(v,3) for v in pos]}    ', end='', flush=True)
            else:
                cv2.putText(vis, 'NO DETECTION', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            cv2.imshow('Top Camera - Bolt Detection (q: quit)', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print()
        pipeline.stop()
        cv2.destroyAllWindows()
        print('[TOP] 종료')


if __name__ == '__main__':
    main()
