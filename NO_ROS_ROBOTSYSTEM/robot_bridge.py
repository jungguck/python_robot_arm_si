#!/usr/bin/env python3
"""
robot_bridge.py - 실제 로봇 연동 (MuJoCo 없음)

sim_bridge.py를 대체합니다.

[역할]
  1. trossen_arm 드라이버 연결 + 내장 IK 사용
  2. RobotState 명령 → 실제 로봇 전송
  3. D405 카메라 (탑뷰 + 손목) → RobotState
  4. 관절 피드백 → RobotState
"""
import numpy as np
import trossen_arm

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

from robot_state import RobotState


ROBOT_IP          = '192.168.1.4'
D405_TOP_SERIAL   = '412622271120'
D405_WRIST_SERIAL = '230322274740'

GOAL_TIME   = 1.0    # 명령 목표 도달 시간 (초) — 루프(10ms)보다 길어야 trajectory 중단 없음
MOVE_THRESH = 0.001  # 관절 변화 임계값 (rad) — 이 미만이면 명령 재전송 안 함
SMOOTH_ALPHA = 0.3   # 위치 지수평활 계수 (0.1=느리고 부드럼, 0.5=빠르지만 약간 퍽퍽)


class RobotBridge:
    def __init__(self, state: RobotState):
        self.rs = state

        # ── 로봇 드라이버 ──────────────────────────────────────────────
        self.driver = trossen_arm.TrossenArmDriver()
        self.driver.configure(
            trossen_arm.Model.wxai_v0,
            trossen_arm.StandardEndEffector.wxai_v0_leader,
            ROBOT_IP,
            True
        )
        self.driver.set_joint_modes([trossen_arm.Mode.position] * 7)
        print('[RobotBridge] 로봇 연결 완료')

        # 초기 카르테시안 자세 읽기 (방향값 기준으로 사용)
        init_cart = list(self.driver.get_cartesian_positions())
        self._default_orient = init_cart[3:]   # [rx, ry, rz] 기본 자세 저장
        print(f'[RobotBridge] 초기 EE 자세: {[round(v,3) for v in init_cart]}')

        # ── 카메라 ──────────────────────────────────────────────────────
        self._init_cameras(state)

        # 이전 명령 추적 (변화 없을 때 trajectory reset 방지)
        self._last_joint_cmd  = [None] * 7
        self._smoothed_pos    = None   # exponential smoothing용 목표 위치

        # 초기 피드백
        self._update_feedback()
        print('[RobotBridge] 초기화 완료')

    # ── 카메라 초기화 ─────────────────────────────────────────────────
    def _init_cameras(self, state: RobotState):
        if not _RS_AVAILABLE:
            self.rs_top   = None
            self.rs_wrist = None
            print('[RobotBridge] pyrealsense2 없음 - 카메라 비활성화')
            return

        # 탑뷰 D405
        try:
            self.rs_top = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(D405_TOP_SERIAL)
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = self.rs_top.start(cfg)
            depth_profile      = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
            state.rs_intrinsics = depth_profile.get_intrinsics()
            print(f'[RobotBridge] 탑뷰 D405 연결 (S/N: {D405_TOP_SERIAL})')
        except Exception as e:
            self.rs_top = None
            print(f'[RobotBridge] 탑뷰 D405 실패: {e}')

        # 손목 D405
        try:
            self.rs_wrist = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(D405_WRIST_SERIAL)
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.rs_wrist.start(cfg)
            print(f'[RobotBridge] 손목 D405 연결 (S/N: {D405_WRIST_SERIAL})')
        except Exception as e:
            self.rs_wrist = None
            print(f'[RobotBridge] 손목 D405 실패: {e}')

    # ── 피드백: 실제 로봇 상태 → RobotState ──────────────────────────
    def _update_feedback(self):
        s = self.rs
        try:
            cart = list(self.driver.get_cartesian_positions())
            s.ee_pos = np.array(cart[:3])
            # 현재 EE orientation 갱신 (JOINT_SEQ 이후 자세 변화 반영)
            self._default_orient = cart[3:]

            joints  = list(self.driver.get_arm_positions())
            gripper = self.driver.get_gripper_position()
            s.joint_positions = joints + [gripper]

            if len(joints) >= 6:
                s.ee_joint3 = joints[3]
                s.ee_joint4 = joints[4]
                s.ee_joint5 = joints[5]
        except Exception as e:
            print(f'[RobotBridge] 피드백 오류: {e}')

    # ── 카메라: 실제 이미지 → RobotState ─────────────────────────────
    def _update_cameras(self):
        s = self.rs
        if self.rs_top is not None:
            frames = self.rs_top.poll_for_frames()
            if frames:
                cf = frames.get_color_frame()
                df = frames.get_depth_frame()
                if cf:
                    s.color_image = np.asanyarray(cf.get_data())
                if df:
                    s.depth_image = np.asanyarray(df.get_data())

        if self.rs_wrist is not None:
            frames = self.rs_wrist.poll_for_frames()
            if frames:
                cf = frames.get_color_frame()
                if cf:
                    s.wrist_image = np.asanyarray(cf.get_data())

    # ── 메인 스텝 (매 루프 호출) ──────────────────────────────────────
    def step(self):
        s = self.rs

        # ── 관절 직접 제어 (JOINT_SEQ) ──────────────────────────────
        # J0~J2 가 모두 설정돼 있으면 직접 관절 제어
        j012 = [s.target_joint0, s.target_joint1, s.target_joint2]
        if all(v is not None for v in j012):
            cmds = [s.target_joint0, s.target_joint1, s.target_joint2,
                    s.target_joint3, s.target_joint4, s.target_joint5]
            for i, val in enumerate(cmds):
                if val is None:
                    continue
                # 변화가 있을 때만 명령 전송 → trajectory 불필요한 리셋 방지
                prev = self._last_joint_cmd[i]
                if prev is None or abs(float(val) - prev) > MOVE_THRESH:
                    self.driver.set_joint_position(
                        i, float(val), goal_time=GOAL_TIME, blocking=False)
                    self._last_joint_cmd[i] = float(val)

        # ── 카르테시안 IK (ALIGN / DESCEND 등) ──────────────────────
        elif s.target_position is not None:
            # 지수 평활로 목표 위치 smoothing (MuJoCo 관성 대체)
            raw = np.array(s.target_position, dtype=float)
            if self._smoothed_pos is None:
                self._smoothed_pos = raw.copy()
            else:
                self._smoothed_pos = SMOOTH_ALPHA * raw + (1.0 - SMOOTH_ALPHA) * self._smoothed_pos

            # 방향: target_yaw 있으면 적용, 없으면 기본 자세 유지
            orient = list(self._default_orient)
            if s.target_yaw is not None:
                orient[2] = float(s.target_yaw)

            # J3/J4/J5 override 없는 경우: 순수 cartesian IK
            # override 있는 경우: cartesian 후 joint override
            # ※ set_cartesian_positions와 set_joint_position을 같은 틱에
            #   쓰면 충돌하므로, override 있을 땐 최소한으로만 사용
            goal = list(self._smoothed_pos) + orient
            has_override = any(v is not None for v in
                               [s.target_joint3, s.target_joint4, s.target_joint5])
            try:
                self.driver.set_cartesian_positions(
                    goal,
                    trossen_arm.InterpolationSpace.joint,
                    goal_time=GOAL_TIME,
                    blocking=False
                )
            except Exception:
                pass

            if has_override:
                # override 관절만 개별 전송 (변화 있을 때만)
                for i, val in [(3, s.target_joint3),
                               (4, s.target_joint4),
                               (5, s.target_joint5)]:
                    if val is None:
                        continue
                    prev = self._last_joint_cmd[i]
                    if prev is None or abs(float(val) - prev) > MOVE_THRESH:
                        self.driver.set_joint_position(
                            i, float(val), goal_time=GOAL_TIME, blocking=False)
                        self._last_joint_cmd[i] = float(val)
        else:
            # 목표 없으면 smoothed 위치 초기화
            self._smoothed_pos = None

        # ── 그리퍼 ────────────────────────────────────────────────────
        if s.gripper_target is not None:
            prev = self._last_joint_cmd[6]
            if prev is None or abs(float(s.gripper_target) - prev) > MOVE_THRESH:
                try:
                    self.driver.set_gripper_position(
                        float(s.gripper_target), goal_time=GOAL_TIME, blocking=False)
                    self._last_joint_cmd[6] = float(s.gripper_target)
                except Exception:
                    pass

        # ── 피드백 + 카메라 갱신 ─────────────────────────────────────
        self._update_feedback()
        self._update_cameras()
