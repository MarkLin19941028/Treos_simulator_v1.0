import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import math
import cv2
import os
import time

from constants import *
from models import DispenseArm
from simulation_engine import SimulationEngine

class MovingPatternGenerator:
    def __init__(self, app):
        """
        app: Reference to the main SimulationApp instance
        """
        self.app = app

    def export_nozzle_pattern(self):
        parsed_recipe = self.app.parse_and_prepare_recipe()
        if not parsed_recipe: return

        user_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
            title="Export Moving Pattern Image As..."
        )
        if not user_path: return

        # 套用命名規範
        base_path, ext = os.path.splitext(user_path)
        if ext.lower() not in ['.png', '.jpg', '.jpeg']:
            ext = '.png'
        filepath_img = f"{base_path}_Moving_Pattern{ext}"
        filepath_vid = f"{base_path}_Moving_Pattern.mp4"

        progress_window = tk.Toplevel(self.app.root)
        progress_window.title("Generating Pattern & Video")
        progress_window.geometry("400x120")
        progress_window.transient(self.app.root)
        progress_window.grab_set()
        progress_window.resizable(False, False)
        ttk.Label(progress_window, text="Generating moving pattern image, please wait...", padding=10).pack()

        total_duration = sum(p['total_duration'] for p in parsed_recipe['processes'])
        if total_duration <= 0: total_duration = 1.0

        progress_label = ttk.Label(progress_window, text=f"Processing Time: 0.0s / {total_duration:.1f}s (0%)", padding=(0, 5))
        progress_label.pack()
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=350, mode="determinate", maximum=total_duration)
        progress_bar.pack(pady=10)
        progress_widgets = {'window': progress_window, 'bar': progress_bar, 'label': progress_label}

        try:
            try:
                current_multiplier = float(self.app.speed_var.get().replace('x', ''))
            except (AttributeError, ValueError):
                current_multiplier = 1.0

            self._run_headless_pattern_generation(parsed_recipe, filepath_img, filepath_vid, progress_widgets, play_speed_multiplier=current_multiplier)
            messagebox.showinfo("Success", f"Moving Pattern and Video exported successfully to:\n{filepath_img}\n{filepath_vid}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate pattern and video: {e}")
        finally:
            if progress_window.winfo_exists():
                progress_window.destroy()

    def _run_headless_pattern_generation(self, recipe, filepath_img, filepath_vid, progress_widgets=None, play_speed_multiplier=1.0):
        fig = Figure(figsize=(7, 4.5), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-350, 350)
        ax.set_ylim(-225, 225)
        ax.set_facecolor('#111111')
        ax.add_patch(plt.Circle((0, 0), WAFER_RADIUS, facecolor='#333333', edgecolor='cyan', lw=1.5, zorder=1))

        # 準備影片輸出 (VideoWriter)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_vid = cv2.VideoWriter(filepath_vid, fourcc, 30.0, (w, h))

        # 準備計算覆蓋面積的遮罩 (1mm = 1 pixel)
        grid_size = int(WAFER_RADIUS * 2) + 2
        offset = grid_size // 2
        coverage_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
        
        # 建立晶圓圓形遮罩，用來計算總有效面積
        wafer_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
        cv2.circle(wafer_mask, (offset, offset), int(WAFER_RADIUS), 1, -1)
        total_wafer_pixels = np.sum(wafer_mask)

        # Matplotlib 動態畫線的暫存容器
        arm_lines = {1: [], 2: [], 3: []}
        arm_colors = {1: 'lime', 2: 'magenta', 3: 'yellow'}

        # 建立 config 並指定模式
        pattern_config = self.app.get_current_config() # 獲取目前的物理參數
        max_speed = pattern_config.get('MAX_NOZZLE_SPEED_MMS', 250.0)

        headless_arms = {}
        for i in range(1, 3):
            geo = ARM_GEOMETRIES[i]
            if i == 2:
                headless_arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], None, None, max_nozzle_speed_mms=max_speed,
                                           side_arm_length=geo.get('side_arm_length'), side_arm_angle_offset=geo.get('side_arm_angle_offset'),
                                           side_arm_branch_dist=geo.get('side_arm_branch_dist'))
            else:
                headless_arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], None, None, max_nozzle_speed_mms=max_speed)

        pattern_config['SIMULATION_MODE'] = 'pattern_only' # 強制覆蓋為純軌跡模式

        engine = SimulationEngine(recipe, headless_arms, {}, headless=True, config=pattern_config)

        arm_trajectories = {1: [], 2: [], 3: []}
        
        # 決定高解析度的取樣率 (參考 export_simulation_report 的邏輯)
        max_rpm = 0
        for proc in recipe['processes']:
            spin = proc['spin_params']
            current_max = spin['rpm'] if spin['mode'] == 'Simple' else max(spin['start_rpm'], spin['end_rpm'])
            if current_max > max_rpm: max_rpm = current_max
        sim_fps = max(800, int(max_rpm * 4))
        
        # 影片輸出為 30 FPS
        video_fps = 30.0
        sim_dt = 1.0 / sim_fps
        video_dt = 1.0 / video_fps
        
        # 實際影片中每幀代表的模擬時間間隔
        scaled_video_dt = video_dt * play_speed_multiplier

        total_duration = sum(p['total_duration'] for p in recipe['processes'])
        if total_duration <= 0: total_duration = 1.0

        last_flows = {1: 0.0, 2: 0.0, 3: 0.0}
        last_ui_update_time = time.time()
        last_video_frame_time = -scaled_video_dt # 保證第一張立刻拍
        last_pts = {1: None, 2: None, 3: None}

        while True:
            snapshot = engine.update(sim_dt)
            if progress_widgets:
                # FPS = 每 0.5 秒更新一次 UI
                if time.time() - last_ui_update_time >= 0.5:
                    try:
                        p_bar, p_label = progress_widgets['bar'], progress_widgets['label']
                        p_bar['value'] = min(snapshot['time'], total_duration)
                        percent = (min(snapshot['time'], total_duration) / total_duration) * 100
                        p_label.config(text=f"Processing Pattern: {snapshot['time']:.1f}s / {total_duration:.1f}s ({percent:.0f}%)")
                        progress_widgets['window'].update_idletasks()
                        last_ui_update_time = time.time()
                    except: pass

            curr_arm_id = snapshot['active_arm_id']
            curr_flows = snapshot.get('nozzle_flows', {1: 0.0, 2: 0.0, 3: 0.0})

            # 處理每個噴嘴的紀錄邏輯
            curr_pts = {1: None, 2: None, 3: None}
            if curr_arm_id != 0:
                rad_wafer = math.radians(snapshot['wafer_angle'])
                cos_a, sin_a = math.cos(-rad_wafer), math.sin(-rad_wafer)
                inv_rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                nozzle_positions = snapshot['nozzle_pos']

                # Nozzle 1 (Arm 1)
                if curr_arm_id == 1:
                    flow = curr_flows[1]
                    if flow > 0:
                        pt = np.dot(inv_rot_matrix, nozzle_positions[:2])
                        curr_pts[1] = pt
                        if last_flows[1] <= 0:
                            arm_trajectories[1].append([])
                            arm_lines[1].append(ax.plot([], [], color=arm_colors[1], linewidth=NOZZLE_RADIUS_MM * 2, solid_capstyle='round', alpha=0.6, zorder=10)[0])
                        arm_trajectories[1][-1].append(pt)
                        
                        # 畫到 coverage mask 上
                        if last_pts[1] is not None:
                            p1 = (int(last_pts[1][0]) + offset, int(last_pts[1][1]) + offset)
                            p2 = (int(pt[0]) + offset, int(pt[1]) + offset)
                            cv2.line(coverage_mask, p1, p2, 1, int(NOZZLE_RADIUS_MM * 2))

                # Nozzle 2 & 3 (Arm 2)
                elif curr_arm_id == 2:
                    if isinstance(nozzle_positions, list) and len(nozzle_positions) == 2:
                        n2_pos, n3_pos = nozzle_positions
                        
                        # Nozzle 2
                        flow2 = curr_flows[2]
                        if flow2 > 0:
                            pt2 = np.dot(inv_rot_matrix, n2_pos[:2])
                            curr_pts[2] = pt2
                            if last_flows[2] <= 0:
                                arm_trajectories[2].append([])
                                arm_lines[2].append(ax.plot([], [], color=arm_colors[2], linewidth=NOZZLE_RADIUS_MM * 2, solid_capstyle='round', alpha=0.6, zorder=10)[0])
                            arm_trajectories[2][-1].append(pt2)

                            if last_pts[2] is not None:
                                p1 = (int(last_pts[2][0]) + offset, int(last_pts[2][1]) + offset)
                                p2 = (int(pt2[0]) + offset, int(pt2[1]) + offset)
                                cv2.line(coverage_mask, p1, p2, 1, int(NOZZLE_RADIUS_MM * 2))
                        
                        # Nozzle 3
                        flow3 = curr_flows[3]
                        if flow3 > 0:
                            pt3 = np.dot(inv_rot_matrix, n3_pos[:2])
                            curr_pts[3] = pt3
                            if last_flows[3] <= 0:
                                arm_trajectories[3].append([])
                                arm_lines[3].append(ax.plot([], [], color=arm_colors[3], linewidth=NOZZLE_RADIUS_MM * 2, solid_capstyle='round', alpha=0.6, zorder=10)[0])
                            arm_trajectories[3][-1].append(pt3)

                            if last_pts[3] is not None:
                                p1 = (int(last_pts[3][0]) + offset, int(last_pts[3][1]) + offset)
                                p2 = (int(pt3[0]) + offset, int(pt3[1]) + offset)
                                cv2.line(coverage_mask, p1, p2, 1, int(NOZZLE_RADIUS_MM * 2))

            last_flows = curr_flows.copy()
            last_pts = curr_pts.copy()

            # 只在時間間隔到達 scaled_video_dt 時更新 UI 畫布並寫入影片 frame
            if snapshot['time'] - last_video_frame_time >= scaled_video_dt:
                for arm_id, segments in arm_trajectories.items():
                    if len(segments) > 0 and len(segments[-1]) > 0:
                        coords = np.array(segments[-1])
                        arm_lines[arm_id][-1].set_data(coords[:, 0], coords[:, 1])

                fig.canvas.draw()
                buf = np.asarray(fig.canvas.buffer_rgba())
                frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
                out_vid.write(frame_bgr)
                last_video_frame_time = snapshot['time']

            if snapshot.get('is_finished') or snapshot['time'] > (total_duration + 30.0): break

        # 如果最後沒有寫入剛好結束的 frame，補一張
        for arm_id, segments in arm_trajectories.items():
            if len(segments) > 0 and len(segments[-1]) > 0:
                coords = np.array(segments[-1])
                arm_lines[arm_id][-1].set_data(coords[:, 0], coords[:, 1])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        out_vid.write(frame_bgr)

        out_vid.release()

        has_any_trajectory = False
        
        # 建立 Legend 用的 Handles
        from matplotlib.lines import Line2D
        legend_elements = []

        for arm_id, segments in arm_trajectories.items():
            drawn_this_arm = False
            for segment in segments:
                if len(segment) > 0:
                    has_any_trajectory = True
                    drawn_this_arm = True
            
            if drawn_this_arm:
                label = f"Nozzle {arm_id}"
                legend_elements.append(Line2D([0], [0], color=arm_colors[arm_id], lw=4, label=label))

        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', facecolor='#222222', edgecolor='gray', labelcolor='white', fontsize=9)

        # 計算覆蓋面積
        valid_coverage = cv2.bitwise_and(coverage_mask, wafer_mask)
        covered_pixels = np.sum(valid_coverage)
        if total_wafer_pixels > 0:
            coverage_percentage = (covered_pixels / total_wafer_pixels) * 100.0
        else:
            coverage_percentage = 0.0

        if has_any_trajectory:
            # 標示覆蓋率
            ax.text(0.02, 0.02, f"Coverage Area: {coverage_percentage:.2f}%", 
                    transform=ax.transAxes, color='white', fontsize=12,
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

            # 取得最後一個有效的噴嘴位置作標記
            final_pos = snapshot['nozzle_pos']
            if isinstance(final_pos, list):
                for p in final_pos:
                    ax.plot(p[0], p[1], 'o', color='white', markersize=4, zorder=15)
            else:
                ax.plot(final_pos[0], final_pos[1], 'o', color='white', markersize=4, zorder=15)

        fig.savefig(filepath_img, bbox_inches='tight', dpi=100)
        plt.close(fig)