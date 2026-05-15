#!/usr/bin/env python3
"""
yolo_pipeline.py - RT-DETRv2 볼트 검출 모듈

사용법:
    yolo = YoloPipeline()
    yolo.step(state)  # 매 루프마다 호출 → state 업데이트

업데이트 항목:
    state.bolt_detected_pose  (탑뷰 → 볼트 3D 위치)
    state.bolt_angle          (손목 → 볼트 방향)
"""
import sys
import math
import collections

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, '/home/trossen/RT-DETR/rtdetrv2_pytorch')
sys.path.insert(0, '/home/trossen/RT-DETR/rtdetrv2_pytorch/src')

from src.core import YAMLConfig
import src.zoo  # noqa: RT-DETR 모듈 등록

from robot_state import RobotState


TOP_CONFIG   = '/home/trossen/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_bolt.yml'
TOP_CKPT     = '/home/trossen/RT-DETR/rtdetrv2_pytorch/output/rtdetrv2_r18vd_bolt/best.pth'
WRIST_CONFIG = '/home/trossen/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_bolt_head_shaft.yml'
WRIST_CKPT   = '/home/trossen/RT-DETR/rtdetrv2_pytorch/output/rtdetrv2_r18vd_bolt_head_shaft/best.pth'

CONF_THRESH  = 0.9
INPUT_SIZE   = (640, 640)

# 연속 N프레임 안정적으로 검출돼야 bolt_detected_pose 확정
STABLE_FRAMES   = 5
STABLE_DIST_M   = 0.02   # 연속 프레임 간 위치 편차 허용치 (2cm)

# ── 손목 카메라 shaft 락 파라미터 (ver3 동일) ─────────────────────────
LOCK_RADIUS   = 80    # 락된 shaft 위치에서 이 거리(px) 이내만 같은 볼트로 인정
LOCK_MISS_MAX = 15    # 락 범위 안에 shaft 없는 연속 프레임 초과 시 락 해제

# ── 탑뷰 카메라 → 로봇 월드 좌표 변환 (XML 기준) ─────────────────────
# cam_assy_dummy pos="-0.50 0.0 0.50" euler="0 0 -1.57"
CAM_POS = np.array([-0.50, 0.0, 0.50])
CAM_ROT = np.array([          # Rz(-90°): 카메라→월드 회전
    [ 0,  1,  0],
    [-1,  0,  0],
    [ 0,  0,  1],
])
BOLT_Z        = 0.03           # 볼트 평면 Z 높이 (m), bolt_0 pos z≈0.024 + 여유
CALIB_OFFSET  = np.array([-0.040, 0.036])  # 실측 보정 오프셋 (260511 실측)


class YoloPipeline:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'[YOLO] Device: {self.device}')

        print('[YOLO] 탑뷰 모델 로딩 (bolt 1-class)...')
        self.top_model, self.top_post = self._load(TOP_CONFIG, TOP_CKPT)

        print('[YOLO] 손목 모델 로딩 (bolt_head/shaft 2-class)...')
        self.wrist_model, self.wrist_post = self._load(WRIST_CONFIG, WRIST_CKPT)

        print('[YOLO] 모델 로드 완료')

        # 안정성 체크용 버퍼
        self._pose_history = collections.deque(maxlen=STABLE_FRAMES)

        # shaft 락 상태 (손목 카메라)
        self._locked_shaft_center = None   # None=락없음 / (cx,cy)=추적중
        self._lock_miss_count     = 0

    def _load(self, config_path: str, ckpt_path: str):
        cfg  = YAMLConfig(config_path)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if 'ema' in ckpt:
            state_dict = ckpt['ema']['module']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        model = cfg.model
        model.load_state_dict(state_dict, strict=False)
        model.eval().to(self.device)
        post = cfg.postprocessor
        post.eval().to(self.device)
        return model, post

    def _infer(self, model, post, frame):
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        img   = TF.resize(img, list(INPUT_SIZE))
        img_t = img.unsqueeze(0).to(self.device)
        orig  = torch.tensor([[w, h]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            results = post(model(img_t), orig)
        r = results[0]
        return r['labels'].cpu().numpy(), r['boxes'].cpu().numpy(), r['scores'].cpu().numpy()

    def _pixel_to_3d(self, px, py, intrinsics) -> np.ndarray | None:
        """픽셀 좌표 → 로봇 월드 좌표 (XML CAM_POS/CAM_ROT 기반, ver2 동일 방식)"""
        if intrinsics is not None:
            fx, fy = intrinsics.fx, intrinsics.fy
            cx, cy = intrinsics.ppx, intrinsics.ppy
        else:
            fx, fy, cx, cy = 383.8, 383.8, 320.0, 240.0

        # 픽셀 → 카메라 좌표계 방향 벡터
        d_cam = np.array([(px - cx) / fx, -(py - cy) / fy, -1.0])
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

    def _stable_pose_update(self, state, new_pose: np.ndarray):
        """N프레임 연속 안정적일 때만 bolt_detected_pose 확정"""
        self._pose_history.append(new_pose)
        if len(self._pose_history) < STABLE_FRAMES:
            return
        poses  = np.array(self._pose_history)
        mean   = poses.mean(axis=0)
        spread = np.max(np.linalg.norm(poses - mean, axis=1))
        if spread < STABLE_DIST_M:
            state.bolt_detected_pose = mean.copy()

    def _compute_bolt_angle(self, labels, boxes, scores) -> float | None:
        """shaft 락 로직 포함 볼트 각도 계산 (ver3 동일)"""
        # 클래스별 분리
        shafts, heads = [], []
        for label, box, score in zip(labels, boxes, scores):
            if score < CONF_THRESH:
                continue
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            if int(label) == 1:
                shafts.append((cx, cy, float(score)))  # confidence 포함
            elif int(label) == 0:
                heads.append((cx, cy))

        # shaft 락 로직
        best_shaft = None
        if shafts:
            if self._locked_shaft_center is None:
                # 최초 락: confidence 가장 높은 shaft 선택
                best = max(shafts, key=lambda p: p[2])
                best_shaft = (best[0], best[1])
                self._locked_shaft_center = best_shaft
                self._lock_miss_count = 0
            else:
                lx, ly = self._locked_shaft_center
                nearby = [(cx, cy) for cx, cy, _ in shafts
                          if math.hypot(cx - lx, cy - ly) < LOCK_RADIUS]
                if nearby:
                    # 락 위치에서 가장 가까운 shaft 선택
                    best_shaft = min(nearby, key=lambda p: math.hypot(p[0]-lx, p[1]-ly))
                    # 락 위치 부드럽게 갱신 (급격한 점프 방지)
                    self._locked_shaft_center = (
                        lx * 0.7 + best_shaft[0] * 0.3,
                        ly * 0.7 + best_shaft[1] * 0.3,
                    )
                    self._lock_miss_count = 0
                else:
                    self._lock_miss_count += 1
                    if self._lock_miss_count >= LOCK_MISS_MAX:
                        self._locked_shaft_center = None
                        self._lock_miss_count = 0
        else:
            if self._locked_shaft_center is not None:
                self._lock_miss_count += 1
                if self._lock_miss_count >= LOCK_MISS_MAX:
                    self._locked_shaft_center = None
                    self._lock_miss_count = 0

        if best_shaft is None or not heads:
            return None

        # 락된 shaft에서 가장 가까운 head 페어링
        scx, scy = best_shaft
        hcx, hcy = min(heads, key=lambda p: math.hypot(p[0]-scx, p[1]-scy))

        dx = hcx - scx
        dy = hcy - scy
        return (math.degrees(math.atan2(dx, -dy)) + 360) % 360

    def step(self, state: RobotState):
        # 탑뷰: bolt 위치 검출
        color = state.color_image
        if color is not None:
            color = color.copy()  # 스레드 안전
            labels, boxes, scores = self._infer(self.top_model, self.top_post, color)
            best_score, best_box = 0.0, None
            for label, box, score in zip(labels, boxes, scores):
                if score >= CONF_THRESH and score > best_score:
                    best_score, best_box = score, box

            if best_box is not None:
                cx = (best_box[0] + best_box[2]) / 2
                cy = (best_box[1] + best_box[3]) / 2

                if state.bolt_gt is not None:
                    # 시뮬: ground truth 사용
                    self._stable_pose_update(state, state.bolt_gt.copy())
                else:
                    # 실제 로봇: XML 카메라 파라미터 기반 pixel_to_3d
                    pose = self._pixel_to_3d(cx, cy, state.rs_intrinsics)
                    if pose is not None:
                        self._stable_pose_update(state, pose)
            else:
                # 미검출 시 버퍼 초기화
                self._pose_history.clear()

        # 손목: bolt 방향 검출
        wrist = state.wrist_image
        if wrist is not None:
            wrist  = wrist.copy()  # 스레드 안전
            labels, boxes, scores = self._infer(self.wrist_model, self.wrist_post, wrist)
            angle = self._compute_bolt_angle(labels, boxes, scores)
            if angle is not None:
                state.bolt_angle = angle
