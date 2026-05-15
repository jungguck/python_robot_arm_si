#!/usr/bin/env python3
"""
main_pipeline.py - 로봇 연결 유지 (항상 켜놓는 베이스)

[실행]
  conda activate trossen
  cd /home/trossen/jk_ws/src/NO_ROS_ROBOTSYSTEM
  python main_pipeline.py

[종료]
  Ctrl+C
"""
import time
import trossen_arm

ROBOT_IP    = '192.168.1.4'
JOINT_NAMES = ['Joint 0', 'Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Gripper']


def main():
    print(f'[Main] 로봇 연결 중... ({ROBOT_IP})')

    driver = trossen_arm.TrossenArmDriver()
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        ROBOT_IP,
        True
    )
    driver.set_joint_modes([trossen_arm.Mode.position] * 7)

    print('[Main] 연결 완료')
    print()
    for i, name in enumerate(JOINT_NAMES):
        print(f'  Joint {i} ({name})  ✓')
    print()
    print('[Main] 모든 조인트 활성화 완료')
    print('[Main] 실행 중... (종료: Ctrl+C)')
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print()
        print('[Main] 종료')


if __name__ == '__main__':
    main()
