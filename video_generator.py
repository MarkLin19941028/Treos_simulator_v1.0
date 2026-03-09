# =======================================================
# video_generator.py
# =======================================================

import numpy as np
import cv2
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

# 導入 SimulationApp 的常數
from constants import (
    FPS, WAFER_RADIUS, NOTCH_DEPTH, NOTCH_HALF_WIDTH,
    STATE_RUNNING_PROCESS, STATE_MOVING_FROM_CENTER_TO_START,
    ARM_GEOMETRIES, CHAMBER_SIZE, WATER_DROP_SIZE, WATER_ON_WAFER_SIZE
)

from simulation_engine import SimulationEngine
from models import DispenseArm

class VideoGenerator:
    """
    專門處理離屏渲染影片生成。
    """
    def __init__(self, app_instance):
        self.app = app_instance

    def _run_headless_video_generation(self, recipe, filepath, progress_widgets, play_speed_multiplier=1.0, config=None):
        """
        絕對截斷版：
        1. 排除任何引擎旗標干擾，純以計算出的總時長為準。
        2. 確保多個製程 100% 完整錄製。
        3. 支援播放倍率 (play_speed_multiplier)，實現流暢的慢動作影片導出。
        """
        # --- 1. 初始化繪圖環境 (與 Simulator 保持一致：寬 700, 高 450) ---
        fig = Figure(figsize=(7, 4.5), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'box')
        ax.set_facecolor('black')
        ax.set_xlim(-350, 350)
        ax.set_ylim(-225, 225)

        # 背景元件
        ax.add_patch(plt.Rectangle((-350, -225), 700, 450, facecolor='none', edgecolor='gray', lw=2))
        ax.add_patch(plt.Circle((0, 0), WAFER_RADIUS, facecolor='#222222', edgecolor='cyan', lw=1.5, zorder=1))
        # 晶圓中心點
        ax.add_patch(plt.Circle((0, 0), 3, color='cyan', zorder=2))
        notch_patch = plt.Polygon([[0,0],[0,0],[0,0]], closed=True, facecolor='black', edgecolor='cyan', lw=1.5, zorder=2)
        ax.add_patch(notch_patch)

        # 初始化手臂與水柱 Artist
        arms = {}
        arm_colors = {1: 'lime', 2: 'magenta', 3: 'yellow'}
        for i, geo in ARM_GEOMETRIES.items():
            arm_line, = ax.plot([], [], color='gray', lw=4, zorder=12)
            nozzle_head = plt.Circle((0, 0), 10, facecolor=arm_colors[i], zorder=13)
            ax.add_patch(nozzle_head)
            
            if i == 2:
                side_arm_line, = ax.plot([], [], color='gray', lw=4, zorder=12)
                side_nozzle_head = plt.Circle((0, 0), 10, facecolor='yellow', zorder=13)
                ax.add_patch(side_nozzle_head)
                arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], 
                                   arm_line, nozzle_head, 
                                   side_arm_length=geo.get('side_arm_length'), 
                                   side_arm_angle_offset=geo.get('side_arm_angle_offset'),
                                   side_arm_branch_dist=geo.get('side_arm_branch_dist'),
                                   side_arm_artist=side_arm_line, side_nozzle_artist=side_nozzle_head)
            else:
                arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], arm_line, nozzle_head)

        water_artists = {}
        for i in range(1, 4):
            # 提高 Alpha 值 (0.7) 增加飽和感
            falling, = ax.plot([], [], 'o', color=(0.6, 0.8, 1.0, 0.7), markersize=WATER_DROP_SIZE, zorder=10)
            # 提高 Alpha 值 (0.5) 減少透明度，強化視覺存在感
            on_wafer, = ax.plot([], [], '.', color=(0.6, 0.8, 1.0, 0.5), markersize=WATER_ON_WAFER_SIZE, zorder=3)
            water_artists[i] = {'falling': falling, 'on_wafer': on_wafer}

        status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, color='white', 
                              verticalalignment='top', family='monospace', size=11, zorder=20)

        # --- 2. 初始化引擎 (傳入 headless=True) ---
        global_water_params = self.app._get_water_params()
        water_params_dict = {i: {
            'viscosity': global_water_params['viscosity'],
            'surface_tension': global_water_params['surface_tension'],
            'evaporation_rate': global_water_params['evaporation_rate']
        } for i in [1, 2, 3]}

        engine = SimulationEngine(recipe, arms, water_params_dict, headless=True, config=config)
        
        # --- 3. VideoWriter ---
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(filepath, fourcc, float(FPS), (int(w), int(h)))

        # --- 4. 錄製循環 (與 Simulator 視窗時間完全對齊) ---
        sim_clock = 0.0
        # 根據播放倍率調整 dt。若 multiplier < 1.0，則 dt 變小，產生的幀數變多，從而實現慢動作。
        dt = (1.0 / FPS) * play_speed_multiplier
        
        # 配方總長度 (僅供 UI 進度條參考)
        recipe_net_duration = sum(p['total_duration'] for p in recipe['processes'])
        last_ui_update_time = 0.0 # 使用 time.time() 會需要 import time，這裡我們看 video_generator.py 有沒有 import time

        import time
        last_ui_update_time = time.time()

        while True:
            snapshot = engine.update(dt)
            sim_clock += dt
            
            # 更新渲染與寫入
            active_id = snapshot['active_arm_id']
            for arm_id, arm in arms.items():
                if arm_id == active_id: arm.update_artists(snapshot['nozzle_pos'], color='yellow' if snapshot['is_spraying'] else 'gray')
                else: arm.go_home()
            water_render = snapshot['water_render']
            for arm_id, artists in water_artists.items():
                d = water_render.get(arm_id, {})
                f_data = d.get('falling', np.empty((0, 2)))
                o_data = d.get('on_wafer', np.empty((0, 2)))
                
                # 支援 NumPy 陣列優化繪圖
                if isinstance(f_data, np.ndarray) and f_data.size > 0:
                    artists['falling'].set_data(f_data[:, 0], f_data[:, 1])
                else:
                    artists['falling'].set_data([], [])
                    
                if isinstance(o_data, np.ndarray) and o_data.size > 0:
                    artists['on_wafer'].set_data(o_data[:, 0], o_data[:, 1])
                else:
                    artists['on_wafer'].set_data([], [])

            notch_patch.set_xy(snapshot['notch_coords'])
            
            # 使用 snapshot['time'] 作為顯示時間 (與 Simulator 視窗左上角 Time 完全一致)
            display_text = (f"Time: {snapshot['time']:.2f}s\n"
                            f"Process: {snapshot['process_idx'] + 1}\n"
                            f"State: {snapshot['state']}\n"
                            f"Step: {snapshot['step_str']}\n"
                            f"Process Time: {snapshot['process_time_str']}\n"
                            f"RPM: {snapshot['rpm']:.0f}")
            status_text.set_text(display_text)
            
            fig.canvas.draw()
            rgba_buf = canvas.buffer_rgba()
            img = np.frombuffer(rgba_buf, dtype=np.uint8).reshape(int(h), int(w), 4)
            video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))

            # UI 進度
            if progress_widgets:
                # FPS = 每 0.5 秒更新一次 UI
                if time.time() - last_ui_update_time >= 0.5 or snapshot.get('is_finished'):
                    try:
                        p_bar, p_label = progress_widgets['bar'], progress_widgets['label']
                        # 預估總時長包含 10s 機械動作緩衝
                        est_total = recipe_net_duration + 3.0
                        p_bar['maximum'] = est_total
                        p_bar['value'] = min(snapshot['time'], est_total)
                        p_label.config(text=f"Exporting Video: {snapshot['time']:.1f}s / (Simulating...)")
                        progress_widgets['window'].update_idletasks()
                        last_ui_update_time = time.time()
                    except: pass
                
            # 檢查結束時機 (當最後一幀已繪製完成後)
            if snapshot.get('is_finished'):
                break
                
            # 安全鎖 (防止無限循環)
            if snapshot['time'] > (recipe_net_duration + 30.0): break

        video_writer.release()
        plt.close(fig)
        return True
