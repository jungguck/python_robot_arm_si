#!/usr/bin/env python3
"""
robot_state.py - ROS 토픽을 대체하는 공유 상태 객체

ROS publisher/subscriber 대신 모든 컴포넌트가 이 객체를 공유합니다.
  - Commands   : GraspExecutor → SimBridge  (target_position, gripper_target 등)
  - Sensor data: SimBridge     → GraspExecutor (ee_pos, joint_positions 등)
  - Detection  : Detector      → GraspExecutor (bolt_detected_pose, bolt_angle)
"""
import numpy as np


class RobotState:
    def __init__(self):

        # ── Commands (GraspExecutor → SimBridge) ──────────────────────────
        self.target_position = None   # np.ndarray [x, y, z]
        self.gripper_target  = None   # float (미터)

        self.target_yaw      = None   # float(rad) | None = 해제
        self.target_joint0   = None   # float(rad) | None = 해제
        self.target_joint1   = None
        self.target_joint2   = None
        self.target_joint3   = None
        self.target_joint4   = None
        self.target_joint5   = None

        # ── Sensor data (SimBridge → GraspExecutor) ───────────────────────
        self.ee_pos          = None   # np.ndarray [x, y, z]
        self.bolt_gt         = None   # np.ndarray [x, y, z]  (시뮬 ground truth)

        self.bolt_yaw        = None   # float (rad)  볼트 샤프트 방향
        self.ee_yaw          = None   # float (rad)  EE yaw
        self.ee_joint3       = None   # float (rad)
        self.ee_joint4       = None
        self.ee_joint5       = None

        self.link4_pos       = None   # np.ndarray [x, y, z]
        self.joint_positions = None   # list[float] 길이 7  (qpos[:7])
        self.ee_xmat         = None   # np.ndarray shape (9,)

        # ── Images (SimBridge → Display) ──────────────────────────────────
        self.color_image     = None   # np.ndarray (H, W, 3) BGR
        self.wrist_image     = None
        self.depth_image     = None   # np.ndarray (H, W) uint16, mm 단위 (D405)
        self.rs_intrinsics   = None   # RealSense 내부 파라미터 (back-projection용)

        # ── Detection (Detector → GraspExecutor) ──────────────────────────
        self.bolt_detected_pose = None  # np.ndarray [x, y, z]
        self.bolt_angle         = None  # float (0~360°, 12시=0°, 시계방향)
