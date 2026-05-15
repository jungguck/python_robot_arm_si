#!/usr/bin/env python3
"""
go_home.py - 로봇을 안전하게 홈 위치로 복귀

[실행]
  conda activate trossen
  cd /home/trossen/jk_ws/src/NO_ROS_ROBOTSYSTEM
  python go_home.py

[동작]
  현재 관절 각도에서 홈(전체 0)까지 단계적으로 이동
  각 단계마다 도달 확인 후 다음 단계 진행
"""
import time
import trossen_arm
import numpy as np

ROBOT_IP  = '192.168.1.4'
GOAL_TIME = 2.0   # 각 단계별 목표 도달 시간 (느리고 안전하게)
ARRIVE_THRESH = 0.05   # rad
CHECK_HZ  = 0.05       # 50ms 간격으로 도달 확인

# 홈으로 가는 중간 경유 자세 (급격한 움직임 방지)
# JOINT_SEQ 역순으로 안전하게 펴나감
WAYPOINTS = [
    # [j0,   j1,   j2,   j3,   j4,   j5,  gripper]
    [None, None, None,  0.0,  0.0,  0.0,  None],   # 손목 먼저 펴기
    [None,  0.5,  0.5,  0.0,  0.0,  0.0,  None],   # 팔 중간 자세
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.044],  # 완전 홈 + 그리퍼 오픈
]


def get_joints(driver):
    joints  = list(driver.get_arm_positions())
    gripper = driver.get_gripper_position()
    return joints + [gripper]


def send_waypoint(driver, waypoint, current):
    """None은 현재 값 유지, 값이 있으면 해당 관절만 이동"""
    for i, val in enumerate(waypoint):
        if val is None:
            continue
        driver.set_joint_position(i, float(val), goal_time=GOAL_TIME, blocking=False)


def arrived(current, waypoint, thresh):
    for i, val in enumerate(waypoint):
        if val is None:
            continue
        if abs(current[i] - val) > thresh:
            return False
    return True


def main():
    print('[GoHome] 로봇 연결 중...')
    driver = trossen_arm.TrossenArmDriver()
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        ROBOT_IP,
        True
    )
    driver.set_joint_modes([trossen_arm.Mode.position] * 7)
    print('[GoHome] 연결 완료')

    current = get_joints(driver)
    print(f'[GoHome] 현재 관절: {[round(v, 3) for v in current]}')
    print(f'[GoHome] 홈으로 이동 시작 (GOAL_TIME={GOAL_TIME}s/단계)')

    for step, waypoint in enumerate(WAYPOINTS):
        target_str = [round(v, 3) if v is not None else '-' for v in waypoint]
        print(f'[GoHome] Step {step+1}/{len(WAYPOINTS)}: {target_str}')
        send_waypoint(driver, waypoint, current)

        # 도달 대기
        timeout = GOAL_TIME * 3
        elapsed = 0.0
        while elapsed < timeout:
            time.sleep(CHECK_HZ)
            elapsed += CHECK_HZ
            current = get_joints(driver)
            if arrived(current, waypoint, ARRIVE_THRESH):
                print(f'[GoHome] Step {step+1} 도달 완료')
                break
        else:
            print(f'[GoHome] Step {step+1} 타임아웃 — 다음 단계로 진행')

    current = get_joints(driver)
    print(f'[GoHome] 최종 관절: {[round(v, 3) for v in current]}')
    print('[GoHome] 홈 복귀 완료')


if __name__ == '__main__':
    main()
