#!/usr/bin/env python3
"""
go_to_scan_pose.py - 스캔 포즈로 이동 (NO_ROS 버전)

[스캔 포즈]
  joint 0 = -1.55
  joint 1 =  1.2
  joint 2 =  1.2
  joint 3 = -1.1
  joint 4 =  0
  joint 5 =  0

[실행]
  conda activate trossen
  cd /home/trossen/jk_ws/src/NO_ROS_ROBOTSYSTEM
  python go_to_scan_pose.py

[동작]
  1. 로봇 연결
  2. 현재 관절 각도 읽기
  3. 스캔 포즈까지 부드럽게 이동 (5초)
"""

import time
import numpy as np
import trossen_arm

ROBOT_IP = '192.168.1.4'

# 스캔 포즈 (손목 카메라가 작업 영역을 내려다보는 자세)
SCAN_POSE = np.array([-1.55, 1.25, 1.25, -1.1, 0.0, 0.0])
MOVE_TIME = 5.0  # 이동 시간 (초)


def main():
    print("📷 스캔 포즈 이동 시작...")
    
    # 로봇 연결
    print(f"🔌 로봇 연결 중... ({ROBOT_IP})")
    driver = trossen_arm.TrossenArmDriver()
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        ROBOT_IP,
        True
    )
    driver.set_joint_modes([trossen_arm.Mode.position] * 7)
    print("✅ 로봇 연결 완료")
    
    # 현재 관절 각도 읽기
    current_joints = np.array([driver.get_joint_position(i) for i in range(6)])
    print(f"📍 현재 관절: {np.degrees(current_joints).round(1)}°")
    print(f"🎯 스캔 포즈: {np.degrees(SCAN_POSE).round(1)}°")
    
    # 스캔 포즈로 이동 (blocking=True로 완료까지 대기)
    print(f"🚀 이동 시작 ({MOVE_TIME}초)...")
    for i, angle in enumerate(SCAN_POSE):
        driver.set_joint_position(i, float(angle), goal_time=MOVE_TIME, blocking=False)
    
    # 이동 완료 대기
    time.sleep(MOVE_TIME + 0.5)
    
    # 최종 위치 확인
    final_joints = np.array([driver.get_joint_position(i) for i in range(6)])
    print(f"✅ 도달 완료: {np.degrees(final_joints).round(1)}°")
    
    # 오차 확인
    error = np.abs(final_joints - SCAN_POSE)
    max_error = np.degrees(np.max(error))
    print(f"📊 최대 오차: {max_error:.2f}°")
    
    if max_error < 5.0:
        print("✅ 스캔 포즈 도달 성공!")
    else:
        print(f"⚠️  오차가 큽니다 ({max_error:.2f}°)")
    
    print("\n📷 이제 dino_sam_bolt_detector.py를 실행하세요!")


if __name__ == '__main__':
    main()
