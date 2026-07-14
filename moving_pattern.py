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

    def export_nozzle_pattern(self, filepath=None, parsed_recipe=None):
        """
        導出噴嘴移動軌跡（解耦核心：完整保留所有分析圖表，並支援手動單獨按鈕點選與批量自動化）
        """
        # 1. 處理儲存路徑的分流 (解耦 filedialog)[cite: 4]
        if filepath is None:
            user_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                title="Export Nozzle Moving Pattern As..."
            )
            if not user_path:
                return  # 使用者取消選擇
            base_path, _ = os.path.splitext(user_path)
            is_batch_mode = False
        else:
            base_path = filepath  # 批量自動化模式：直接沿用外部安全傳入的基礎路徑[cite: 4]
            is_batch_mode = True

        # 2. 處理資料來源的分流 (解耦 UI 元件相依性)[cite: 4]
        if parsed_recipe is None:
            recipe_data = self.app.parse_and_prepare_recipe()
            if not recipe_data:
                messagebox.showerror("Error", "Failed to parse current recipe variables.")
                return
        else:
            recipe_data = parsed_recipe

        # --- 核心修正一：移除對 user_path 的二次依賴，改以安全解耦的 base_path 生成所有檔案路徑 ---
        filepath_img = f"{base_path}_Moving_Pattern.png"
        filepath_vid = f"{base_path}_Moving_Pattern.mp4"
        filepath_heatmap = f"{base_path}_Time_Heatmap.png"
        filepath_csv = f"{base_path}_Time_Distribution.csv"
        filepath_radial = f"{base_path}_Radial_Distribution.png"

        # --- 核心修正二：進度條保護。在批量無人值守模式下，不創立彈出對話框，100% 防止畫面卡死 ---
        progress_widgets = None
        if not is_batch_mode:
            progress_window = tk.Toplevel(self.app.root)
            progress_window.title("Generating Pattern & Video")
            progress_window.geometry("400x120")
            progress_window.transient(self.app.root)
            progress_window.grab_set()
            progress_window.resizable(False, False)
            ttk.Label(progress_window, text="Generating moving pattern image, please wait...", padding=10).pack()

            total_duration = sum(p['total_duration'] for p in recipe_data['processes'])
            if total_duration <= 0: total_duration = 1.0

            progress_label = ttk.Label(progress_window, text=f"Processing Time: 0.0s / {total_duration:.1f}s (0%)", padding=(0, 5))
            progress_label.pack()
            progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=350, mode="determinate", maximum=total_duration)
            progress_bar.pack(pady=10)
            progress_widgets = {'window': progress_window, 'bar': progress_bar, 'label': progress_label}

        try:
            # 安全讀取全域速度參數快照
            try:
                current_multiplier = float(self.app.speed_var.get().replace('x', ''))
            except (AttributeError, ValueError):
                current_multiplier = 1.0

            # 3. 呼叫底層 Headless 運算核心 (傳入剛剛分流好的解耦變數 recipe_data)
            self._run_headless_pattern_generation(
                recipe_data, filepath_img, filepath_vid, filepath_heatmap, filepath_csv, filepath_radial, 
                progress_widgets, play_speed_multiplier=current_multiplier
            )
            
            # 手動模式下才跳出提示
            if not is_batch_mode:
                # messagebox.showinfo("Success", "Moving Pattern analyses exported successfully!")
                return

        except Exception as e:
            if not is_batch_mode:
                messagebox.showerror("Error", f"Failed to generate pattern and video: {e}")
            else:
                raise RuntimeError(f"Headless generator crashed inside moving_pattern.py: {str(e)}")
        finally:
            if not is_batch_mode and 'progress_window' in locals() and progress_window.winfo_exists():
                progress_window.destroy()

    def _run_headless_pattern_generation(self, recipe, filepath_img, filepath_vid, filepath_heatmap, filepath_csv, filepath_radial, progress_widgets=None, play_speed_multiplier=1.0):
        # 此方法內所有寫死的 plt 替換為 headless 專用的 Agg Canvas (完全保留您優秀的物理圖表分析演算法)
        fig = Figure(figsize=(7, 4.5), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-350, 350)
        ax.set_ylim(-225, 225)
        ax.set_facecolor('#111111')
        
        # 繪製圓形參考晶圓
        from matplotlib.patches import Circle
        ax.add_patch(Circle((0, 0), WAFER_RADIUS, facecolor='#333333', edgecolor='cyan', lw=1.5, zorder=1))

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

        # 建立累積覆蓋時間的矩陣
        time_accumulation_matrix = np.zeros((grid_size, grid_size), dtype=np.float32)

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
        
        # 決定高解析度的取樣率
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
                        
                        if last_pts[1] is not None:
                            p1 = (int(last_pts[1][0]) + offset, int(last_pts[1][1]) + offset)
                            p2 = (int(pt[0]) + offset, int(pt[1]) + offset)
                            temp_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
                            cv2.line(temp_mask, p1, p2, 1, int(NOZZLE_RADIUS_MM * 2))
                            time_accumulation_matrix += temp_mask * sim_dt
                            cv2.bitwise_or(coverage_mask, temp_mask, dst=coverage_mask)

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
                                temp_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
                                cv2.line(temp_mask, p1, p2, 1, int(NOZZLE_RADIUS_MM * 2))
                                time_accumulation_matrix += temp_mask * sim_dt
                                cv2.bitwise_or(coverage_mask, temp_mask, dst=coverage_mask)
                        
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
                                temp_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
                                cv2.line(temp_mask, p1, p2, 1, int(NOZZLE_RADIUS_MM * 2))
                                time_accumulation_matrix += temp_mask * sim_dt
                                cv2.bitwise_or(coverage_mask, temp_mask, dst=coverage_mask)

            last_flows = curr_flows.copy()
            last_pts = curr_pts.copy()

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

        valid_coverage = cv2.bitwise_and(coverage_mask, wafer_mask)
        covered_pixels = np.sum(valid_coverage)
        coverage_percentage = (covered_pixels / total_wafer_pixels) * 100.0 if total_wafer_pixels > 0 else 0.0

        if has_any_trajectory:
            ax.text(0.02, 0.02, f"Coverage Area: {coverage_percentage:.2f}%", 
                    transform=ax.text(0,0,'').get_transform() if hasattr(ax, 'transAxes') == False else ax.transAxes, 
                    color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

            final_pos = snapshot['nozzle_pos']
            if isinstance(final_pos, list):
                for p in final_pos: ax.plot(p[0], p[1], 'o', color='white', markersize=4, zorder=15)
            else:
                ax.plot(final_pos[0], final_pos[1], 'o', color='white', markersize=4, zorder=15)

        fig.savefig(filepath_img, bbox_inches='tight', dpi=100)
        plt.close(fig)

        # ====== 產生並儲存熱力圖 ======
        masked_time_matrix = np.where(wafer_mask == 1, time_accumulation_matrix, np.nan)
        fig_heat = Figure(figsize=(7, 6), dpi=100)
        canvas_heat = FigureCanvasAgg(fig_heat)
        ax_heat = fig_heat.add_subplot(111)
        ax_heat.set_title("Nozzle Coverage Time Distribution (s)", fontsize=14, color='white')
        ax_heat.set_facecolor('#111111')
        fig_heat.patch.set_facecolor('#222222')
        
        extent = [-WAFER_RADIUS, WAFER_RADIUS, -WAFER_RADIUS, WAFER_RADIUS]
        im = ax_heat.imshow(masked_time_matrix, extent=extent, origin='lower', cmap='jet', vmin=0)
        ax_heat.set_xlabel("X (mm)", color='white')
        ax_heat.set_ylabel("Y (mm)", color='white')
        ax_heat.tick_params(colors='white')
        
        cbar = fig_heat.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.set_label("Time (s)", color='white')
        
        fig_heat.savefig(filepath_heatmap, bbox_inches='tight', dpi=100, facecolor=fig_heat.get_facecolor())
        plt.close(fig_heat)

        # ====== 產生並儲存徑向分佈圖與 CSV ======
        y_indices, x_indices = np.indices(time_accumulation_matrix.shape)
        distances = np.sqrt((x_indices - offset)**2 + (y_indices - offset)**2)
        
        max_dist = int(WAFER_RADIUS)
        radial_bins = np.arange(0, max_dist + 1, 1)
        radial_times = np.zeros(len(radial_bins) - 1)
        
        for i in range(len(radial_bins) - 1):
            mask = (distances >= radial_bins[i]) & (distances < radial_bins[i+1]) & (wafer_mask == 1)
            radial_times[i] = np.mean(time_accumulation_matrix[mask]) if np.any(mask) else 0.0
                
        bin_centers = (radial_bins[:-1] + radial_bins[1:]) / 2.0

        with open(filepath_csv, mode='w', newline='', encoding='utf-8') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["--- Radial Distribution ---"])
            writer.writerow(["Radius (mm)", "Average Coverage Time (s)"])
            for r, t in zip(bin_centers, radial_times):
                writer.writerow([f"{r:.1f}", f"{t:.4f}"])
            
            writer.writerow([])
            writer.writerow(["--- 2D Coverage Time Matrix (s) ---"])
            writer.writerow(["Y \\ X"] + [f"{x - offset}" for x in range(grid_size)])
            for y in range(grid_size):
                row = [f"{y - offset}"] + [f"{masked_time_matrix[y, x]:.4f}" if not np.isnan(masked_time_matrix[y, x]) else "" for x in range(grid_size)]
                writer.writerow(row)

        fig_rad = Figure(figsize=(7, 4.5), dpi=100)
        canvas_rad = FigureCanvasAgg(fig_rad)
        ax_rad = fig_rad.add_subplot(111)
        
        ax_rad.plot(bin_centers, radial_times, '-', color='cyan', linewidth=2)
        ax_rad.set_title("Radial Distribution of Nozzle Coverage Time", fontsize=14, color='white')
        ax_rad.set_xlabel("Radius (mm)", color='white')
        ax_rad.set_ylabel("Average Coverage Time (s)", color='white')
        ax_rad.set_facecolor('#111111')
        fig_rad.patch.set_facecolor('#222222')
        ax_rad.tick_params(colors='white')
        ax_rad.grid(True, color='#444444', linestyle='--', alpha=0.7)
        ax_rad.set_xlim(0, max_dist)
        ax_rad.set_ylim(bottom=0)
        
        fig_rad.savefig(filepath_radial, bbox_inches='tight', dpi=100, facecolor=fig_rad.get_facecolor())
        plt.close(fig_rad)