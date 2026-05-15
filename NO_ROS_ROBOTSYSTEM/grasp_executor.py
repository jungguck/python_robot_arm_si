#!/usr/bin/env python3
"""
grasp_executor.py - 볼트 파지 상태머신 (ROS 없이)

[원본] grasp_executor_node2.py (ROS2 GraspExecutor 노드)
[변경] ROS publisher/subscriber → RobotState 직접 읽기/쓰기
       create_timer 콜백 → step() 메서드 (main loop에서 직접 호출)
       get_logger() → print()

[상태 흐름]
IDLE → JOINT_SEQ → BOLT_CHECK → ALIGN → DESCEND → GRASP
     → LIFT → RETRACT → PLACE → APPROACH_GOAL → RELEASE → RETREAT → DONE
"""
import time
import numpy as np
from enum import Enum, auto

from robot_state import RobotState


# ════════════════════════════════════════════════════════════════════════
# 파라미터
# ════════════════════════════════════════════════════════════════════════

SAFE_Z         = 0.6
GRASP_Z_OFFSET = 0
ARRIVE_THRESH  = 0.01
ARRIVE_COUNT   = 2
GRIPPER_OPEN   = 0.044
GRIPPER_MIDDLE = 0.025
GRIPPER_CLOSE  = 0.0005
INTERP_STEPS   = 10
ALIGN_WAIT     = 6
RETRACT_WAIT   = 30

YAW_OFFSET     = 0
STUCK_STEPS    = 40
STUCK_Z_THRESH = 0.001

GOAL_POS = np.array([-0.097, -0.293, 0.0992])

ANGLE_TARGET       = 180.0
ANGLE_TOL          = 10.0
ANGLE_GAIN         = 0.005
BOLT_CHECK_TIMEOUT = 500
ANGLE_STABLE_COUNT = 10

JOINT_SEQ = [
    [0.0,   0.0,  0.483,  0.0],
    [0.0,   0.99, 1.01,   0.0],
    [-1.68, 0.99, 1.01,   0.0],
    [-1.68, 1.24, 1.24,   0.0],
    [-1.68, 1.41, 1.41,  -0.75],
    [-1.68, 1.48, 1.61,  -1.37],
]
JOINT_ARRIVE_THRESH = 0.05
JOINT_ARRIVE_COUNT  = 10
JOINT_SEQ_HOLD      = 50

# ════════════════════════════════════════════════════════════════════════


def _normalize_angle(angle_rad: float) -> float:
    while angle_rad >  np.pi:
        angle_rad -= 2 * np.pi
    while angle_rad < -np.pi:
        angle_rad += 2 * np.pi
    return angle_rad


def make_path(p_start, p_end, n=INTERP_STEPS):
    return [p_start + (p_end - p_start) * (i / n) for i in range(1, n + 1)]


class State(Enum):
    IDLE          = auto()
    JOINT_SEQ     = auto()
    BOLT_CHECK    = auto()
    ALIGN         = auto()
    DESCEND       = auto()
    GRASP         = auto()
    LIFT          = auto()
    RETRACT       = auto()
    PLACE         = auto()
    APPROACH_GOAL = auto()
    RELEASE       = auto()
    RETREAT       = auto()
    DONE          = auto()


class GraspExecutor:
    def __init__(self, robot_state: RobotState):
        self.rs = robot_state

        # 상태 변수 (원본 GraspExecutor와 동일)
        self.state                  = State.IDLE
        self.bolt_locked            = False
        self.bolt_angle             = None
        self.angle_stable_count     = 0
        self.bolt_pos               = None
        self.ee_pos                 = None
        self.bolt_yaw               = None
        self.link4_pos              = None
        self.ee_joint5              = None
        self.ee_joint3              = None
        self.snap_bolt_pos          = None
        self.snap_yaw               = None
        self.current_pos            = None

        self.joint_seq_idx          = 0
        self.joint_seq_arrive_count = 0
        self.joint_seq_hold_count   = 0
        self.descend_target_z       = None
        self.descend_timeout        = 0

        self.path                   = []
        self.path_idx               = 0
        self.arrive_count           = 0
        self.wait_count             = 0

        self.prev_ee_z              = None
        self.stuck_count            = 0
        self.joint5_locked          = False

        # 로그 쓰로틀 (key: 메시지 앞 30자, value: 마지막 출력 시각)
        self._log_timers: dict = {}

        print('[GraspExecutor] 초기화 완료')

    # ── 로그 헬퍼 ─────────────────────────────────────────────────────
    def _log(self, msg: str, throttle_sec: float = 0.0, level: str = 'INFO'):
        if throttle_sec > 0:
            key = msg[:30]
            now = time.time()
            if self._log_timers.get(key, 0) + throttle_sec > now:
                return
            self._log_timers[key] = now
        print(f'[{level}] {msg}')

    # ── RobotState 동기화 ──────────────────────────────────────────────
    def _sync_from_state(self):
        """매 step 시작 시 shared state에서 센서 데이터를 로컬로 복사"""
        s = self.rs
        if s.ee_pos is not None:
            self.ee_pos = s.ee_pos.copy()
        if s.bolt_yaw is not None:
            self.bolt_yaw = s.bolt_yaw
        if s.ee_joint5 is not None:
            self.ee_joint5 = s.ee_joint5
        if s.ee_joint3 is not None:
            self.ee_joint3 = s.ee_joint3
        if s.link4_pos is not None:
            self.link4_pos = s.link4_pos.copy()
        if s.joint_positions is not None:
            self.current_pos = list(s.joint_positions)
        if s.bolt_angle is not None:
            self.bolt_angle = s.bolt_angle
        # bolt_detected_pose → bolt_pos (bolt_locked 이면 갱신 안 함)
        if not self.bolt_locked and s.bolt_detected_pose is not None:
            self.bolt_pos = s.bolt_detected_pose.copy()

    # ── 명령 발행 헬퍼 (상태 → RobotState) ──────────────────────────
    def _pub_target(self, pos):
        self.rs.target_position = np.array([float(pos[0]), float(pos[1]), float(pos[2])])

    def _pub_gripper(self, val: float):
        self.rs.gripper_target = float(val)

    def _clear_yaw(self):
        self.rs.target_yaw = None

    def _clear_joint5(self):
        self.rs.target_joint5 = None

    def _clear_joint4(self):
        self.rs.target_joint4 = None

    def _clear_joint3(self):
        self.rs.target_joint3 = None

    def _clear_joint0(self):
        self.rs.target_joint0 = None

    def _clear_joint1(self):
        self.rs.target_joint1 = None

    def _clear_joint2(self):
        self.rs.target_joint2 = None

    def _publish_joints(self, j0, j1, j2, j3):
        """J0~J3 직접 제어, J4/J5=0 고정"""
        if self.ee_pos is not None:
            self._pub_target(self.ee_pos)
        self.rs.target_joint0 = float(j0)
        self.rs.target_joint1 = float(j1)
        self.rs.target_joint2 = float(j2)
        self.rs.target_joint3 = float(j3)
        self.rs.target_joint4 = 0.0
        self.rs.target_joint5 = 0.0

    def _lock_arm(self, yaw_val=None):
        """J3=-1.57, J4=0 고정"""
        self.rs.target_joint3 = -1.57
        self.rs.target_joint4 = 0.0

    def _lock_wrist(self, yaw_val=None):
        """J4=1.57, J5=yaw_val 고정"""
        self.rs.target_joint4 = 1.57
        target_j5 = yaw_val if yaw_val is not None else (self.snap_yaw or 0.0)
        self.rs.target_joint5 = float(_normalize_angle(target_j5 + YAW_OFFSET))

    # ── 경로 추종 ──────────────────────────────────────────────────────
    def _set_path(self, p_end, n=INTERP_STEPS):
        if self.ee_pos is None:
            return
        self.path         = make_path(self.ee_pos.copy(), p_end, n)
        self.path_idx     = 0
        self.arrive_count = 0

    def _follow_path(self, thresh=ARRIVE_THRESH) -> bool:
        if not self.path or self.path_idx >= len(self.path):
            return True
        target = self.path[self.path_idx]
        self._pub_target(target)
        dist = np.linalg.norm(self.ee_pos - target)
        if dist < thresh:
            self.arrive_count += 1
            if self.arrive_count >= ARRIVE_COUNT:
                self.path_idx += 1
                self.arrive_count = 0
        else:
            self.arrive_count = 0
        return self.path_idx >= len(self.path)

    # ── 상태 전환 ──────────────────────────────────────────────────────
    def _next(self, s: State):
        self._log(f'{self.state.name} → {s.name}')
        self.state        = s
        self.path         = []
        self.path_idx     = 0
        self.arrive_count = 0
        self.wait_count   = 0
        self.prev_ee_z    = None
        self.stuck_count  = 0
        if s == State.IDLE:
            self.bolt_locked = False
            self.bolt_pos    = None

    # ── 메인 스텝 ──────────────────────────────────────────────────────
    def step(self):
        self._sync_from_state()

        # 파지 유지 상태: 매 틱마다 그리퍼 CLOSE 유지
        if self.state in (State.GRASP, State.RETRACT, State.PLACE, State.APPROACH_GOAL):
            self._pub_gripper(GRIPPER_CLOSE)

        # ① IDLE
        if self.state == State.IDLE:
            if self.ee_pos is not None and self.current_pos is not None:
                self._log('센서 수신 완료 → JOINT_SEQ 시작')
                self.joint_seq_idx          = 0
                self.joint_seq_arrive_count = 0
                self._next(State.JOINT_SEQ)

        # ② JOINT_SEQ
        elif self.state == State.JOINT_SEQ:
            if self.joint_seq_idx >= len(JOINT_SEQ):
                self._pub_gripper(GRIPPER_MIDDLE)
                return

            target = JOINT_SEQ[self.joint_seq_idx]
            self._publish_joints(*target)

            if self.current_pos is not None:
                err = max(abs(self.current_pos[i] - target[i]) for i in range(4))
                if err < JOINT_ARRIVE_THRESH:
                    self.joint_seq_arrive_count += 1
                    if self.joint_seq_arrive_count >= JOINT_ARRIVE_COUNT:
                        self.joint_seq_hold_count += 1
                        if self.joint_seq_hold_count % 50 == 1:
                            self._log(
                                f'POS_{self.joint_seq_idx + 1} 유지 중... '
                                f'{self.joint_seq_hold_count}/{JOINT_SEQ_HOLD}')
                        if self.joint_seq_hold_count >= JOINT_SEQ_HOLD:
                            self._log(
                                f'POS_{self.joint_seq_idx + 1} 완료 '
                                f'({self.joint_seq_idx + 1}/{len(JOINT_SEQ)})')
                            self.joint_seq_idx += 1
                            self.joint_seq_arrive_count = 0
                            self.joint_seq_hold_count   = 0
                            if self.joint_seq_idx >= len(JOINT_SEQ):
                                if self.rs.bolt_detected_pose is None:
                                    self._log('볼트 미검출 - 인식 대기 중...', throttle_sec=2.0)
                                else:
                                    self._log('관절 시퀀스 완료 → ALIGN 시작')
                                    self._clear_joint0()
                                    self._clear_joint1()
                                    self._clear_joint2()
                                    self._next(State.ALIGN)
                else:
                    self.joint_seq_arrive_count = 0
                    self.joint_seq_hold_count   = 0

        # ③ BOLT_CHECK
        elif self.state == State.BOLT_CHECK:
            self._pub_gripper(GRIPPER_MIDDLE)
            self.wait_count += 1

            if self.bolt_angle is None:
                self._log(
                    f'[BOLT_CHECK] 각도 데이터 대기 중... ({self.wait_count}/{BOLT_CHECK_TIMEOUT})',
                    throttle_sec=1.0)
                if self.wait_count >= BOLT_CHECK_TIMEOUT:
                    self._log('[BOLT_CHECK] 타임아웃 → 각도 무시하고 ALIGN 진행', level='WARN')
                    self._next(State.ALIGN)
                return

            angle = self.bolt_angle
            error = angle - ANGLE_TARGET

            self._log(
                f'[BOLT_CHECK] angle={angle:.1f}°  error={error:+.1f}°  '
                f'stable={self.angle_stable_count}/{ANGLE_STABLE_COUNT}',
                throttle_sec=0.3)

            if abs(error) <= ANGLE_TOL:
                self.angle_stable_count += 1
                if self.angle_stable_count >= ANGLE_STABLE_COUNT:
                    self.rs.target_joint5 = 0.0
                    self._log(f'[BOLT_CHECK] 완료 (angle={angle:.1f}°) → DESCEND')
                    self._next(State.DESCEND)
            else:
                self.angle_stable_count = 0
                offset = error
                if offset >= 180:
                    offset -= 360
                elif offset <= -180:
                    offset += 360
                target_j5 = float(np.clip(np.radians(offset), -np.pi, np.pi))
                self.rs.target_joint5 = target_j5
                self._log(
                    f'[BOLT_CHECK] joint5 설정: {target_j5:.3f} rad '
                    f'(bolt={angle:.1f}°  error={error:+.1f}°)',
                    throttle_sec=0.3)
                if self.wait_count >= BOLT_CHECK_TIMEOUT:
                    self._log('[BOLT_CHECK] 타임아웃 → 강제 DESCEND 진행', level='WARN')
                    self._next(State.DESCEND)

        # ④ ALIGN
        elif self.state == State.ALIGN:
            self.rs.target_joint3 = -1.57
            self.rs.target_joint4 = 0.0

            target = np.array([
                self.bolt_pos[0] - 0.03,
                self.bolt_pos[1],
                self.bolt_pos[2] + 0.15
            ])

            if not self.path:
                self._set_path(target)

            path_done = self._follow_path()
            xy_error  = np.linalg.norm(self.ee_pos[:1] - target[:1])

            self._log(
                f'[ALIGN] XY오차={xy_error*100:.1f}cm  '
                f'EE=({self.ee_pos[0]:.3f},{self.ee_pos[1]:.3f},{self.ee_pos[2]:.3f})  '
                f'bolt=({self.bolt_pos[0]:.3f},{self.bolt_pos[1]:.3f})',
                throttle_sec=0.3)

            if xy_error < 0.005:
                self.wait_count += 1
                self._log(f'ALIGN 수렴 중... {self.wait_count}/{ALIGN_WAIT} XY={xy_error*100:.1f}cm',
                          throttle_sec=0.1)
                if self.wait_count >= ALIGN_WAIT:
                    self.snap_bolt_pos = self.bolt_pos.copy()
                    self.bolt_locked   = True
                    self._log(f'ALIGN 완료. 스냅={self.snap_bolt_pos.round(3)} → BOLT_CHECK')
                    self.angle_stable_count = 0
                    self._next(State.BOLT_CHECK)
            else:
                self.wait_count = 0
                if path_done:
                    self.path = []

        # ⑤ DESCEND
        elif self.state == State.DESCEND:
            if self.ee_pos is None or self.snap_bolt_pos is None:
                return

            self.rs.target_joint3 = -1.57
            self.rs.target_joint4 = 0.0

            if self.descend_target_z is None:
                self.descend_target_z = self.ee_pos[2]

            final_target_z = self.snap_bolt_pos[2] + GRASP_Z_OFFSET
            target_xy      = np.array([self.snap_bolt_pos[0] + 0.005, self.snap_bolt_pos[1]])
            xy_error       = np.linalg.norm(self.ee_pos[:2] - target_xy)
            err_z          = abs(self.ee_pos[2] - self.descend_target_z)

            if xy_error > 0.01:
                target = np.array([target_xy[0], target_xy[1], self.ee_pos[2]])
                self._log(f'XY 재정렬 중: xy={xy_error*100:.1f}cm', throttle_sec=0.2)
            else:
                if err_z < 0.01:
                    drop_speed = 0.02
                    self.descend_target_z = max(
                        self.descend_target_z - drop_speed, final_target_z)
                target = np.array([target_xy[0], target_xy[1], self.descend_target_z])
                self._log(
                    f'하강 중: Z목표={self.descend_target_z:.3f}, '
                    f'Z오차={err_z*100:.1f}cm, xy={xy_error*100:.1f}cm',
                    throttle_sec=0.2)

            self._pub_target(target)

            final_err_z = abs(self.ee_pos[2] - final_target_z)
            if xy_error < 0.005 and final_err_z < 0.01:
                self.wait_count += 1
                if self.wait_count >= 5:
                    self._log('목표 높이 도달 & XY 정렬 완료 → GRASP')
                    self._next(State.GRASP)
                    self.wait_count = 0
            else:
                self.wait_count = 0

        # ⑥ GRASP
        elif self.state == State.GRASP:
            if self.ee_pos is not None and self.bolt_pos is not None:
                target_z = self.descend_target_z
                self._lock_arm(self.snap_yaw)
                self.rs.target_joint3 = -1.57
                self._pub_target(np.array([
                    self.snap_bolt_pos[0],
                    self.snap_bolt_pos[1],
                    target_z
                ]))

                dist_xy = np.linalg.norm(self.ee_pos[:2] - self.snap_bolt_pos[:2])
                dist_z  = abs(self.ee_pos[2] - target_z)
                self.stuck_count += 1

                self._log(f'GRASP XY={dist_xy*100:.1f}cm Z={dist_z*100:.1f}cm',
                          throttle_sec=0.5)

                if dist_xy < 0.01 and dist_z < 0.01:
                    self._pub_gripper(GRIPPER_CLOSE)
                    self.wait_count += 1
                    if self.wait_count >= 20:
                        self._log('파지 완료 → LIFT')
                        self._next(State.LIFT)
                elif self.stuck_count > 100:
                    self._log('IK 한계 → DESCEND 재시도', level='WARN')
                    self._next(State.DESCEND)
                else:
                    self.wait_count = 0

        # ⑦ LIFT
        elif self.state == State.LIFT:
            self._lock_arm(self.snap_yaw)
            self._pub_gripper(GRIPPER_CLOSE)

            if not self.path:
                lift_height = 0.02
                target = np.array([
                    self.ee_pos[0],
                    self.ee_pos[1],
                    self.ee_pos[2] + lift_height
                ])
                self._set_path(target, n=50)
                self._log(f'LIFT: → {target.round(3)}')

            if self._follow_path(thresh=0.02):
                if np.linalg.norm(self.ee_pos - self.snap_bolt_pos) > 0.12:
                    self._log('볼트 놓침! 리셋.', level='WARN')
                    self._next(State.IDLE)
                else:
                    self._log('상승 완료 → RETRACT')
                    self._next(State.RETRACT)
                    self.joint5_locked = False

        # ⑧ RETRACT
        elif self.state == State.RETRACT:
            self._pub_gripper(GRIPPER_CLOSE)
            progress   = min(self.wait_count / RETRACT_WAIT, 1.0)
            self.rs.target_joint3 = float(-1.57 * (1.0 - progress))
            self.rs.target_joint4 = float( 1.57 * (1.0 - progress))
            self.rs.target_joint5 = float( 1.57 * (1.0 - progress))

            self.wait_count += 1
            if self.wait_count % 10 == 0:
                self._log(f'팔 펴는 중... {self.wait_count}/{RETRACT_WAIT}')
            if self.wait_count >= RETRACT_WAIT:
                self._log('자세 복귀 완료 → PLACE')
                self._next(State.PLACE)

        # ⑨ PLACE
        elif self.state == State.PLACE:
            if not self.path:
                target = np.array([GOAL_POS[0], GOAL_POS[1], SAFE_Z])
                self._set_path(target, n=25)

            progress = min(self.path_idx / max(len(self.path), 1), 1.0)
            self.rs.target_joint3 = float(-1.57 * (1.0 - progress))
            self.rs.target_joint4 = 0.0
            self.rs.target_joint5 = 0.0

            if self._follow_path(thresh=0.03):
                self._next(State.APPROACH_GOAL)

        # ⑩ APPROACH_GOAL
        elif self.state == State.APPROACH_GOAL:
            self.rs.target_joint3 = 0.0
            self.rs.target_joint4 = 0.0
            self.rs.target_joint5 = 0.0

            if not self.path:
                target = np.array([GOAL_POS[0], GOAL_POS[1], GOAL_POS[2] + 0.01])
                self._set_path(target, n=15)
                self._log(f'APPROACH_GOAL: 목표지점 하강 → {target.round(3)}')

            if self.path and self.path_idx < len(self.path):
                self._pub_target(self.path[self.path_idx])

            if self._follow_path(thresh=0.015):
                self._log('목표지점 도달 → RELEASE')
                self._next(State.RELEASE)

        # ⑪ RELEASE
        elif self.state == State.RELEASE:
            if self.ee_pos is not None:
                self._pub_target(self.ee_pos)
            self.rs.target_joint3 = 0.0
            self.rs.target_joint4 = 0.0
            self.rs.target_joint5 = 0.0
            self._pub_gripper(GRIPPER_OPEN)
            self.wait_count += 1
            if self.wait_count >= 20:
                self._log('그리퍼 개방 완료 → RETREAT')
                self._next(State.RETREAT)

        # ⑫ RETREAT
        elif self.state == State.RETREAT:
            self.rs.target_joint3 = 0.0
            self.rs.target_joint4 = 0.0
            self.rs.target_joint5 = 0.0

            if not self.path:
                if self.ee_pos is not None:
                    target = self.ee_pos.copy()
                    target[2] += 0.10
                    self._set_path(target, n=10)
                    self._log(f'RETREAT: 수직 상승 → {target.round(3)}')

            if self._follow_path():
                self._log('작업 완료!')
                self._clear_joint3()
                self._clear_joint4()
                self._clear_joint5()
                self._next(State.DONE)
