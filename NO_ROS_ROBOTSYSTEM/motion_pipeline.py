#!/usr/bin/env python3
"""
motion_pipeline.py - 전체 파이프라인 러너 (로봇 + YOLO + 파지 시퀀스)

[스레드 구조]
  Main Thread  (100Hz): bridge.step() + executor.step()  ← 로봇 제어
  YOLO Thread  (최대속도): yolo.step()                    ← 인식만

[실행]
  conda activate trossen
  cd /home/trossen/jk_ws/src/NO_ROS_ROBOTSYSTEM
  python motion_pipeline.py

[종료]
  Ctrl+C
"""
import time
import threading

import cv2
import numpy as np

from robot_state import RobotState
from robot_bridge import RobotBridge
from yolo_pipeline import YoloPipeline
from grasp_executor import GraspExecutor


LOOP_HZ = 0.01   # 제어 루프 ≈ 100 Hz


def _yolo_worker(yolo: YoloPipeline, state: RobotState, stop: threading.Event):
    while not stop.is_set():
        yolo.step(state)


def _vis_worker(state: RobotState, stop: threading.Event):
    """카메라 뷰 표시 스레드 (30Hz)"""
    while not stop.is_set():
        top   = state.color_image
        wrist = state.wrist_image

        if top is not None:
            vis = top.copy()
            # 볼트 검출 위치 표시
            if state.bolt_detected_pose is not None:
                pos = state.bolt_detected_pose
                txt = f'bolt: X={pos[0]:.3f} Y={pos[1]:.3f} Z={pos[2]:.3f}'
                cv2.putText(vis, txt, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(vis, 'bolt: detecting...', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
            cv2.imshow('Top Camera', vis)

        if wrist is not None:
            vis = wrist.copy()
            # 볼트 각도 표시
            if state.bolt_angle is not None:
                cv2.putText(vis, f'angle: {state.bolt_angle:.1f} deg', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            else:
                cv2.putText(vis, 'angle: detecting...', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
            cv2.imshow('Wrist Camera', vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop.set()
            break

        time.sleep(1/30)


def main():
    state    = RobotState()
    bridge   = RobotBridge(state)
    yolo     = YoloPipeline()
    executor = GraspExecutor(state)

    # YOLO 별도 스레드 시작
    stop_event  = threading.Event()
    yolo_thread = threading.Thread(
        target=_yolo_worker, args=(yolo, state, stop_event), daemon=True)
    yolo_thread.start()
    print('[Motion] YOLO 스레드 시작')

    # 카메라 뷰 스레드 시작
    vis_thread = threading.Thread(
        target=_vis_worker, args=(state, stop_event), daemon=True)
    vis_thread.start()
    print('[Motion] 카메라 뷰 시작 (q: 종료)')
    print('[Motion] 제어 루프 시작. 종료: Ctrl+C')

    try:
        while True:
            t0 = time.time()

            bridge.step()     # 실제 로봇 제어 + 피드백 + 카메라
            executor.step()   # 파지 시퀀스 한 틱

            print(f'\r[Motion] State: {executor.state.name:<20}', end='', flush=True)

            elapsed = time.time() - t0
            remaining = LOOP_HZ - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print()
        print('[Motion] 종료 중...')
    finally:
        stop_event.set()
        yolo_thread.join(timeout=2.0)
        vis_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        print('[Motion] 종료 완료')


if __name__ == '__main__':
    main()
