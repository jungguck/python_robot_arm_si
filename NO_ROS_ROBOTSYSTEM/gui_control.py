#!/usr/bin/env python3
"""
gui_control.py - 로봇 GUI 조인트 제어기 (MuJoCo 없이 실제 로봇 직접 제어)

[실행]
  conda activate trossen
  cd /home/trossen/jk_ws/src/NO_ROS_ROBOTSYSTEM
  python gui_control.py
"""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import trossen_arm

ROBOT_IP = '192.168.1.4'

JOINT_NAMES  = ['Joint 0', 'Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Gripper']
JOINT_LIMITS = [
    (-3.14,  3.14),
    (-1.80,  1.80),
    (-1.80,  1.80),
    (-1.80,  1.80),
    (-1.80,  1.80),
    (-3.14,  3.14),
    ( 0.0,   0.044),
]
HOME_POS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.022]

GOAL_TIME = 0.3   # 슬라이더 이동 시 목표 도달 시간 (초)
HOME_TIME = 2.0


class RobotGUI:
    """
    Tkinter 기반 로봇 조인트 제어 GUI

    [Tkinter 기본 개념]
    - Tk(): 메인 윈도우 (root window)
    - Frame: 위젯들을 그룹화하는 컨테이너
    - pack(): 위젯을 부모 컨테이너에 배치 (top, bottom, left, right)
    - grid(): 격자 형태로 배치 (row, column)
    - place(): 절대 좌표로 배치 (x, y)
    """
    def __init__(self, root: tk.Tk):
        # root: Tkinter 메인 윈도우 객체
        self.root        = root
        self.driver      = None          # trossen_arm 드라이버 인스턴스
        self.connected   = False         # 로봇 연결 상태
        self.connecting  = False         # 중복 연결 방지 플래그

        # 윈도우 설정
        root.title('Trossen WXAI - Joint Controller')  # 타이틀바 텍스트
        root.resizable(False, False)                   # 크기 조절 비활성화
        root.configure(bg='#1e1e2e')                   # 배경색 (다크 테마)

        self._build_ui()  # UI 구성 메서드 호출

    # ── UI 구성 ──────────────────────────────────────────────────────────
    def _build_ui(self):
        """
        GUI 레이아웃 구성

        [구조]
        root (메인 윈도우)
        ├─ top (Frame)    : 연결 버튼 + 상태 표시
        ├─ mid (Frame)    : 조인트 슬라이더 7개
        └─ bot (Frame)    : Home/Gripper 버튼 + 스텝 설정
        """
        # S: 스타일 딕셔너리 (색상, 폰트 등)
        S = {
            'bg':      '#1e1e2e',                    # 배경색 (다크 그레이)
            'fg':      '#cdd6f4',                    # 전경색 (밝은 회색)
            'accent':  '#89b4fa',                    # 강조색 (파란색)
            'green':   '#a6e3a1',                    # 초록색
            'red':     '#f38ba8',                    # 빨간색
            'yellow':  '#f9e2af',                    # 노란색
            'surface': '#313244',                    # 표면색 (약간 밝은 회색)
            'font':    ('Consolas', 11),             # 기본 폰트
            'font_b':  ('Consolas', 11, 'bold'),     # 볼드 폰트
            'font_h':  ('Consolas', 14, 'bold'),     # 헤더 폰트
        }
        self.S = S
        r = self.root

        # ── 상단: 연결 패널 ───────────────────────────────────────────────
        # Frame: 위젯들을 담는 컨테이너
        # padx/pady: 내부 여백 (padding)
        top = tk.Frame(r, bg=S['surface'], padx=12, pady=8)
        # pack(): 위젯을 부모에 배치
        # fill='x': 가로 방향으로 꽉 채움
        # padx/pady: 외부 여백 (margin)
        top.pack(fill='x', padx=10, pady=(10, 0))

        # Label: 텍스트 표시 위젯
        # side='left': 왼쪽부터 차례로 배치
        tk.Label(top, text='Trossen WXAI  Joint Controller',
                 font=S['font_h'], bg=S['surface'], fg=S['accent']).pack(side='left')

        # Button: 클릭 가능한 버튼
        # command: 클릭 시 실행할 함수
        self.btn_connect = tk.Button(
            top, text='Connect', font=S['font_b'],
            bg='#45475a', fg=S['green'], activebackground='#585b70',
            relief='flat', padx=14, pady=4,
            command=self._on_connect  # 버튼 클릭 시 _on_connect() 호출
        )
        self.btn_connect.pack(side='right', padx=(8, 0))

        # 연결 상태 표시 레이블 (나중에 텍스트/색상 변경 위해 self에 저장)
        self.lbl_status = tk.Label(top, text='● Disconnected',
                                   font=S['font'], bg=S['surface'], fg=S['red'])
        self.lbl_status.pack(side='right', padx=8)

        # ── 중단: 조인트 슬라이더 ─────────────────────────────────────────
        # 7개 조인트(0~5번 + 그리퍼)를 위한 슬라이더 영역
        mid = tk.Frame(r, bg=S['bg'], padx=10, pady=10)
        mid.pack(fill='both', expand=True)  # expand=True: 남은 공간 모두 차지

        # 슬라이더 관련 위젯들을 저장할 리스트
        self.sliders     = []    # Scale 위젯 (슬라이더)
        self.val_labels  = []    # 현재 값 표시 Label
        self.slider_vars = []    # DoubleVar (슬라이더 값 저장)

        # 7개 조인트에 대해 반복
        for i, (name, (lo, hi)) in enumerate(zip(JOINT_NAMES, JOINT_LIMITS)):
            # 각 조인트마다 하나의 행(row) Frame 생성
            row = tk.Frame(mid, bg=S['surface'], pady=6, padx=10)
            row.pack(fill='x', pady=3)

            # 조인트 이름 레이블 (왼쪽 고정)
            tk.Label(row, text=f'{name}', width=9, anchor='w',
                     font=S['font_b'], bg=S['surface'], fg=S['accent']).pack(side='left')

            # 최소값 표시 레이블
            tk.Label(row, text=f'{lo:.2f}', width=6, anchor='e',
                     font=S['font'], bg=S['surface'], fg=S['fg']).pack(side='left')

            # DoubleVar: Tkinter 변수 (슬라이더 값과 양방향 바인딩)
            var = tk.DoubleVar(value=HOME_POS[i])  # 초기값 = HOME_POS
            self.slider_vars.append(var)

            # ttk.Scale: 슬라이더 위젯
            # variable=var: 슬라이더 값이 var와 자동 동기화
            # command: 슬라이더 값 변경 시 호출할 함수
            sl = ttk.Scale(
                row, from_=lo, to=hi, orient='horizontal',
                variable=var, length=380,
                command=lambda val, idx=i: self._on_slider(idx, val)  # 값 변경 시 콜백
            )
            sl.pack(side='left', padx=6)
            self.sliders.append(sl)

            # 최대값 표시 레이블
            tk.Label(row, text=f'{hi:.2f}', width=6, anchor='w',
                     font=S['font'], bg=S['surface'], fg=S['fg']).pack(side='left')

            # 현재 값 표시 레이블 (노란색, 나중에 업데이트)
            lbl = tk.Label(row, text=f'{HOME_POS[i]:+.3f}', width=8,
                           font=S['font_b'], bg=S['surface'], fg=S['yellow'])
            lbl.pack(side='left', padx=(4, 0))
            self.val_labels.append(lbl)

            # +/- 미세 조정 버튼
            # lambda에서 idx=i: 클로저 문제 방지 (i 값 고정)
            tk.Button(row, text='+', font=S['font_b'], width=2,
                      bg='#45475a', fg=S['green'], relief='flat',
                      command=lambda idx=i: self._step(idx, +1)  # +1 스텝 증가
                      ).pack(side='left', padx=(6, 1))
            tk.Button(row, text='-', font=S['font_b'], width=2,
                      bg='#45475a', fg=S['red'], relief='flat',
                      command=lambda idx=i: self._step(idx, -1)  # -1 스텝 감소
                      ).pack(side='left', padx=(1, 0))

        # ── 하단: 버튼 + 스텝 설정 ───────────────────────────────────────
        bot = tk.Frame(r, bg=S['surface'], padx=12, pady=8)
        bot.pack(fill='x', padx=10, pady=(0, 10))

        # 스텝 크기 설정 (라디안 단위)
        tk.Label(bot, text='Step (rad):', font=S['font'],
                 bg=S['surface'], fg=S['fg']).pack(side='left')

        # DoubleVar: 스텝 크기 저장
        self.step_var = tk.DoubleVar(value=0.05)
        # Combobox: 드롭다운 선택 위젯
        step_combo = ttk.Combobox(bot, textvariable=self.step_var,
                                  values=[0.01, 0.02, 0.05, 0.10, 0.20],
                                  width=5, state='readonly')  # readonly: 직접 입력 불가
        step_combo.pack(side='left', padx=(4, 16))

        # Home 버튼: 모든 조인트를 HOME_POS로 이동
        tk.Button(bot, text='Home', font=S['font_b'], width=8,
                  bg='#45475a', fg=S['accent'], relief='flat', pady=4,
                  command=self._on_home).pack(side='left', padx=4)

        # Gripper Open 버튼: 그리퍼를 최대값(0.044)으로
        tk.Button(bot, text='Gripper Open', font=S['font_b'],
                  bg='#45475a', fg=S['green'], relief='flat', pady=4, padx=8,
                  command=lambda: self._set_joint(6, JOINT_LIMITS[6][1])
                  ).pack(side='left', padx=4)

        # Gripper Close 버튼: 그리퍼를 0으로
        tk.Button(bot, text='Gripper Close', font=S['font_b'],
                  bg='#45475a', fg=S['red'], relief='flat', pady=4, padx=8,
                  command=lambda: self._set_joint(6, 0.0)
                  ).pack(side='left', padx=4)

        # 명령 상태 표시 레이블 (오른쪽)
        self.lbl_cmd = tk.Label(bot, text='', font=S['font'],
                                bg=S['surface'], fg=S['yellow'])
        self.lbl_cmd.pack(side='right', padx=8)

        # 슬라이더 스타일
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TScale', background=S['surface'])

    # ── 연결/해제 ────────────────────────────────────────────────────────
    def _on_connect(self):
        if self.connected:
            self.connected  = False
            self.connecting = False
            self.driver     = None
            self.lbl_status.config(text='● Disconnected', fg=self.S['red'])
            self.btn_connect.config(text='Connect')
            return

        if self.connecting:
            return   # 이미 연결 시도 중

        self.connecting = True
        self.lbl_status.config(text='● Connecting...', fg=self.S['yellow'])
        self.btn_connect.config(state='disabled')

        def _connect():
            try:
                driver = trossen_arm.TrossenArmDriver()
                driver.configure(
                    trossen_arm.Model.wxai_v0,
                    trossen_arm.StandardEndEffector.wxai_v0_leader,
                    ROBOT_IP,
                    True
                )
                driver.set_joint_modes([trossen_arm.Mode.position] * 7)
                self.driver    = driver
                self.connected = True
                self.root.after(0, lambda: [
                    self.lbl_status.config(text=f'● Connected  ({ROBOT_IP})', fg=self.S['green']),
                    self.btn_connect.config(text='Disconnect', state='normal'),
                    self.lbl_cmd.config(text='Connected. Press [Home] when ready.'),
                ])
            except Exception as e:
                err = str(e)
                self.connecting = False
                self.root.after(0, lambda: [
                    self.lbl_status.config(text='● Connect failed', fg=self.S['red']),
                    self.btn_connect.config(text='Connect', state='normal'),
                    self.lbl_cmd.config(text=f'Error: {err}'),
                ])

        threading.Thread(target=_connect, daemon=True).start()

    # ── 슬라이더 콜백 ────────────────────────────────────────────────────
    def _on_slider(self, idx: int, val: str):
        v = float(val)
        self.val_labels[idx].config(text=f'{v:+.3f}')
        self._send(idx, v, GOAL_TIME)

    # ── +/- 버튼 ─────────────────────────────────────────────────────────
    def _step(self, idx: int, direction: int):
        step   = self.step_var.get()
        lo, hi = JOINT_LIMITS[idx]
        cur    = self.slider_vars[idx].get()
        new_v  = max(lo, min(hi, cur + direction * step))
        self.slider_vars[idx].set(new_v)
        self.val_labels[idx].config(text=f'{new_v:+.3f}')
        self._send(idx, new_v, GOAL_TIME)

    # ── 특정 조인트 값 직접 설정 ─────────────────────────────────────────
    def _set_joint(self, idx: int, value: float):
        lo, hi = JOINT_LIMITS[idx]
        v = max(lo, min(hi, value))
        self.slider_vars[idx].set(v)
        self.val_labels[idx].config(text=f'{v:+.3f}')
        self._send(idx, v, GOAL_TIME)

    # ── 홈 ───────────────────────────────────────────────────────────────
    def _on_home(self):
        for i, v in enumerate(HOME_POS):
            self.slider_vars[i].set(v)
            self.val_labels[i].config(text=f'{v:+.3f}')
        if self.connected and self.driver:
            for i, v in enumerate(HOME_POS):
                self.driver.set_joint_position(i, float(v), goal_time=HOME_TIME, blocking=False)
            self.lbl_cmd.config(text='Home → moving...')

    def _go_home_silent(self):
        pass

    # ── 로봇 전송 ────────────────────────────────────────────────────────
    def _send(self, idx: int, value: float, goal_time: float):
        if not self.connected or self.driver is None:
            return
        try:
            self.driver.set_joint_position(idx, float(value), goal_time=goal_time, blocking=False)
            self.lbl_cmd.config(text=f'{JOINT_NAMES[idx]} -> {value:+.3f}')
        except Exception as e:
            self.lbl_cmd.config(text=f'Error: {e}')


def main():
    root = tk.Tk()
    app  = RobotGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
