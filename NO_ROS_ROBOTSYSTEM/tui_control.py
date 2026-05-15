#!/usr/bin/env python3
"""
tui_control.py - 터미널 UI 조인트 제어기 (MuJoCo 없이 실제 로봇 직접 제어)

[실행]
  conda activate trossen
  cd /home/trossen/jk_ws/src/NO_ROS_ROBOTSYSTEM
  python tui_control.py

[조작]
  ↑ / ↓      : 조인트 선택
  → / +  / = : 선택 조인트 + 방향 이동
  ← / -      : 선택 조인트 - 방향 이동
  [ / ]       : 스텝 크기 조절 (0.01 ~ 0.20 rad)
  h           : 홈 포지션
  q           : 종료
"""
import curses
import time
import trossen_arm

ROBOT_IP = '192.168.1.4'

JOINT_NAMES = ['Joint 0', 'Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Gripper']

JOINT_LIMITS = [
    (-3.14,  3.14),   # joint 0  (베이스 회전)
    (-1.80,  1.80),   # joint 1
    (-1.80,  1.80),   # joint 2
    (-1.80,  1.80),   # joint 3
    (-1.80,  1.80),   # joint 4
    (-3.14,  3.14),   # joint 5  (엔드이펙터 회전)
    ( 0.0,   0.044),  # gripper  (미터)
]

HOME_POS  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.022]
STEP_SIZE = [0.01, 0.02, 0.05, 0.10, 0.20]  # rad 선택지


def connect_robot():
    driver = trossen_arm.TrossenArmDriver()
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        ROBOT_IP,
        True
    )
    driver.set_joint_modes([trossen_arm.Mode.position] * 7)
    return driver


def send_joint(driver, idx: int, value: float, goal_time: float = 0.3):
    driver.set_joint_position(idx, float(value), goal_time=goal_time, blocking=False)


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def run(stdscr, driver, joint_pos):
    curses.cbreak()
    curses.noecho()
    stdscr.keypad(True)
    stdscr.nodelay(True)

    try:
        curses.curs_set(0)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN,  curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED,    curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN,   curses.COLOR_BLACK)
    except Exception:
        pass

    selected   = 0
    step_idx   = 2          # 기본 0.05 rad
    msg        = '로봇 연결 완료. 홈으로 이동 중...'

    # 홈으로 이동
    for i, v in enumerate(joint_pos):
        send_joint(driver, i, v, goal_time=2.0)
    time.sleep(2.0)
    msg = '준비 완료'

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()

        def put(row, col, text, attr=0):
            if row >= h - 1:
                return
            text = text[:w - col - 1]
            try:
                stdscr.addstr(row, col, text, attr)
            except curses.error:
                pass

        # ── 헤더 ────────────────────────────────────────────────────────
        put(0, 0, '== Trossen WXAI Joint Controller ==', curses.color_pair(4) | curses.A_BOLD)
        put(1, 0, f'  IP: {ROBOT_IP}   Step: {STEP_SIZE[step_idx]:.2f} rad   ([/] change)')

        # ── 조인트 목록 ──────────────────────────────────────────────────
        put(3, 0, f"  {'#':<3} {'Name':<10} {'rad':>8}   Range", curses.A_UNDERLINE)
        for i, (name, pos, (lo, hi)) in enumerate(zip(JOINT_NAMES, joint_pos, JOINT_LIMITS)):
            bar_len = 16
            ratio   = (pos - lo) / (hi - lo) if hi != lo else 0.5
            fill    = int(ratio * bar_len)
            bar     = '#' * fill + '.' * (bar_len - fill)
            sel     = i == selected
            prefix  = '> ' if sel else '  '
            line    = f'{prefix}{i:<3} {name:<10} {pos:>+8.3f}   [{lo:.2f},{hi:.2f}]  [{bar}]'
            attr    = curses.color_pair(1) | curses.A_BOLD if sel else 0
            put(4 + i, 0, line, attr)

        # ── 조작 안내 ────────────────────────────────────────────────────
        base = 4 + len(JOINT_NAMES) + 1
        put(base,     0, '-' * 50)
        put(base + 1, 0, '  Up/Down: select   Right/+: inc   Left/-: dec')
        put(base + 2, 0, '  h: home   q: quit   [/]: step size')
        put(base + 3, 0, f'  Status: {msg}', curses.color_pair(2))

        stdscr.refresh()

        key = stdscr.getch()
        if key == -1:
            time.sleep(0.02)
            continue

        lo, hi = JOINT_LIMITS[selected]

        if key == ord('q'):
            break

        elif key == curses.KEY_UP:
            selected = (selected - 1) % len(JOINT_NAMES)

        elif key == curses.KEY_DOWN:
            selected = (selected + 1) % len(JOINT_NAMES)

        elif key in (curses.KEY_RIGHT, ord('+'), ord('=')):
            step = STEP_SIZE[step_idx]
            joint_pos[selected] = clamp(joint_pos[selected] + step, lo, hi)
            send_joint(driver, selected, joint_pos[selected])
            msg = f'Joint {selected} → {joint_pos[selected]:+.3f} rad'

        elif key in (curses.KEY_LEFT, ord('-')):
            step = STEP_SIZE[step_idx]
            joint_pos[selected] = clamp(joint_pos[selected] - step, lo, hi)
            send_joint(driver, selected, joint_pos[selected])
            msg = f'Joint {selected} → {joint_pos[selected]:+.3f} rad'

        elif key == ord('['):
            step_idx = max(0, step_idx - 1)
            msg = f'스텝 크기: {STEP_SIZE[step_idx]:.2f} rad'

        elif key == ord(']'):
            step_idx = min(len(STEP_SIZE) - 1, step_idx + 1)
            msg = f'스텝 크기: {STEP_SIZE[step_idx]:.2f} rad'

        elif key in (ord('h'), ord('H')):
            joint_pos[:] = HOME_POS[:]
            for i, v in enumerate(joint_pos):
                send_joint(driver, i, v, goal_time=2.0)
            msg = '홈 포지션으로 이동 중...'


def main():
    print(f'[TUI] 로봇 연결 중... ({ROBOT_IP})')
    try:
        driver = connect_robot()
    except Exception as e:
        print(f'[TUI] 연결 실패: {e}')
        return

    joint_pos = HOME_POS[:]
    try:
        curses.wrapper(run, driver, joint_pos)
    except KeyboardInterrupt:
        pass
    finally:
        print('[TUI] 종료')


if __name__ == '__main__':
    main()
