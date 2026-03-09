import numpy as np
import math
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from numba import jit

from models import DispenseArm
from simulation_engine import SimulationEngine
from constants import (
    ARM_GEOMETRIES, WAFER_RADIUS, REPORT_FPS, 
    CHAMBER_SIZE, REPORT_INTERVAL_MM
)

# --- Numba Kernel 核心優化區 ---
@jit(nopython=True, cache=True, fastmath=True)
def fast_accumulate(coords, heatmap, dt, grid_size, range_min, range_max):
    """
    使用 Numba JIT 編譯的極速累積函數。
    一次完成：邊界檢查 -> 網格索引計算 -> 累加
    (座標系已由引擎處理為旋轉座標系，此處不再進行旋轉)
    """
    n_particles = coords.shape[0]
    # 計算每個 pixel 代表的物理尺寸 (例如 300mm / 300 grid = 1.0mm)
    bin_width = (range_max - range_min) / grid_size
    
    for i in range(n_particles):
        rx = coords[i, 0]
        ry = coords[i, 1]
        
        # 計算網格索引 (Binning)
        # 將物理座標 [-150, 150] 映射到 陣列索引 [0, 300]
        bin_x = int((rx - range_min) / bin_width)
        bin_y = int((ry - range_min) / bin_width)
        
        # 3. 邊界檢查與累加
        if 0 <= bin_x < grid_size and 0 <= bin_y < grid_size:
            # 注意：np.histogram2d 預設回傳 H[x, y]，而影像通常是 row=y, col=x
            # 這裡我們直接對應物理空間，保持與原程式邏輯一致
            heatmap[bin_x, bin_y] += dt

class AccuHeatmapGenerator:
    def __init__(self, app_instance):
        self.app = app_instance

    def generate(self, recipe, filepath, config=None, progress_widgets=None):
        """
        核心累積熱圖生成邏輯 (優化版)
        """
        if config is None:
            from simulation_config_def import get_default_config
            config = get_default_config()

        # 1. 初始化 Headless Arms
        headless_arms = {i: DispenseArm(i, geo['pivot'], geo['home'], geo['length'], None, None) 
                         for i, geo in ARM_GEOMETRIES.items()}

        # 嘗試獲取水參數
        try:
            water_params = self.app._get_water_params()
        except AttributeError:
            # Fallback 如果 app 沒有該方法 (單獨測試時)
            water_params = {'viscosity': 1.0, 'surface_tension': 72.8, 'evaporation_rate': 0.0}

        water_params_dict = {i: {
            'viscosity': water_params['viscosity'],
            'surface_tension': water_params['surface_tension'],
            'evaporation_rate': water_params['evaporation_rate']
        } for i in [1, 2, 3]}

        # 2. 實例化引擎
        engine = SimulationEngine(recipe, headless_arms, water_params_dict, headless=True, config=config)
        
        # 3. 準備熱圖矩陣
        grid_size = 300
        heatmap_accum = np.zeros((grid_size, grid_size), dtype=np.float64)
        
        report_fps = recipe.get('dynamic_report_fps', REPORT_FPS)
        dt = 1.0 / report_fps
        total_duration = sum(p['total_duration'] for p in recipe['processes'])
        sim_clock = 0.0

        # 新增：控制進度條顯示更新的頻率 (例如每 0.5 秒更新一次進度條上的文字 and 百分比)
        progress_display_interval = 0.5
        last_progress_display_time = 0.0 # 上次更新進度條顯示的時間

        # 新增：在循環開始前，為 JIT 編譯提供提示，並強制刷新 GUI
        if progress_widgets:
            progress_widgets['label'].config(text="Initializing JIT Engine for Heatmap (first run might be slow)...")
            # 確保 progress_widgets['bar'] 的最大值已經設定
            progress_widgets['bar']['maximum'] = total_duration
            progress_widgets['window'].update_idletasks() # 強制刷新 GUI

        # 定義範圍常數
        RANGE_MIN = -150.0
        RANGE_MAX = 150.0

        print(f"Starting optimized simulation... Total Duration: {total_duration}s")

        # 5. 執行模擬
        while True:
            snapshot = engine.update(dt) 
            sim_clock += dt
            
            # 判斷是否到了更新進度條顯示的時間，或者模擬已經結束
            if (sim_clock - last_progress_display_time >= progress_display_interval) or snapshot.get('is_finished'):
                if progress_widgets:
                    try:
                        p_bar = progress_widgets['bar']
                        p_label = progress_widgets['label']
                        # 確保最大值已經設定
                        p_bar['maximum'] = total_duration
                        p_bar['value'] = min(sim_clock, total_duration)
                        
                        percent = (min(sim_clock, total_duration) / total_duration) * 100
                        p_label.config(text=f"Generating Heatmap: {sim_clock:.1f}s / {total_duration:.1f}s ({percent:.0f}%)")
                        
                        # 強制刷新 GUI，讓進度條視窗有機會處理事件和繪製更新
                        progress_widgets['window'].update_idletasks()
                        
                        last_progress_display_time = sim_clock # 更新上次顯示時間
                    except tk.TclError as e: # 捕獲使用者關閉進度視窗時可能發生的錯誤
                        print(f"Heatmap progress window closed by user during GUI update: {e}, stopping generation.")
                        return False # 返回 False 表示生成被取消
                    except Exception as e:
                        print(f"Error updating Heatmap progress bar: {e}")
                        return False # 返回 False 表示生成失敗

            # --- 優化區塊開始 ---
            
            # 步驟 A: 快速提取座標 (優化：直接從引擎的 NumPy 陣列提取)
            # 現在引擎直接提供相對於晶圓的座標
            on_wafer_mask = engine.particles_state == 2 # P_ON_WAFER
            coords_array = engine.particles_pos[on_wafer_mask, :2]
            
            if coords_array.size > 0:
                # 步驟 B: 呼叫 Numba Kernel 進行極速計算
                # 這裡取代了原本的 np.dot 和 np.histogram2d
                fast_accumulate(
                    coords_array, 
                    heatmap_accum, 
                    dt, 
                    grid_size, 
                    RANGE_MIN, 
                    RANGE_MAX
                )
            # --- 優化區塊結束 ---

            if snapshot.get('is_finished') or sim_clock > (total_duration + 30.0):
                break

        self._export_results(heatmap_accum, filepath)
        return True

    def _export_results(self, heatmap_matrix, filepath):
        base_path, _ = os.path.splitext(filepath)
        heatmap_png_path = filepath
        
        real_base = base_path.replace("_Accumulation_Heatmap", "")
        heatmap_csv_path = f"{real_base}_Accumulation_RawData.csv"
        radial_png_path = f"{real_base}_Accumulation_Radial_Distribution.png"

        # 轉置矩陣以符合視覺慣例 (通常 histogram2d 的 x,y 與影像的 row,col 是轉置關係)
        # 這裡與原程式碼邏輯保持一致
        data = heatmap_matrix.T 
        
        grid_dim = int(np.sqrt(data.size))
        if data.ndim == 1: data = data.reshape((grid_dim, grid_dim))
            
        if data.size > 0:
            h_max = np.max(data)
            h_median = np.median(data[data > 0]) if np.any(data > 0) else 0.0
            h_std = np.std(data)
        else:
            h_max = h_median = h_std = 0.0

        # 繪圖部分 (Matplotlib 操作不適合 Numba，維持原樣)
        plt.figure(figsize=(11, 9), dpi=120)
        im = plt.imshow(data, origin='lower', extent=[-150, 150, -150, 150], cmap='magma', interpolation='nearest')
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Accumulated Residence Time (Seconds)')
        wafer_circle = plt.Circle((0, 0), 150, color='white', fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_artist(wafer_circle)
        plt.title("Wafer Water Accumulation Heatmap (Quantitative)", fontsize=14, pad=15)
        plt.xlabel("X Position (mm)")
        plt.ylabel("Y Position (mm)")
        
        stats_text = (f"Max Time:    {h_max:.4f} s\n"
                    f"Median(>0):  {h_median:.4f} s\n"
                    f"Std Dev:     {h_std:.4f}\n"
                    f"Resolution:  1.0 mm/pixel")
        plt.text(-145, -145, stats_text, color='white', fontsize=10, family='monospace', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

        plt.tight_layout()
        try:
            plt.savefig(heatmap_png_path, bbox_inches='tight', dpi=300)
        except Exception as e:
            print(f"Error saving image: {e}")
        plt.close()

        try:
            np.savetxt(heatmap_csv_path, data, delimiter=",", fmt='%.6f', 
                       header="Raw Accumulation Data (Seconds), Resolution: 1.0mm/pixel, Range: -150 to 150 mm")
        except Exception as e:
            print(f"Failed to write heatmap raw data to CSV: {e}")
        
        self._export_accumulation_radial_distribution(heatmap_matrix, radial_png_path)

    def _export_accumulation_radial_distribution(self, matrix, filepath):
        """
        計算並輸出隨半徑變化的累積時間分佈圖 (Radial Distribution)
        """
        grid_size = matrix.shape[0]
        center = grid_size / 2.0
        
        y, x = np.indices(matrix.shape)
        r = np.sqrt((x - center + 0.5)**2 + (y - center + 0.5)**2)
        
        r_rounded = r.astype(int)
        max_r = int(WAFER_RADIUS)
        
        radial_sum = np.zeros(max_r + 1)
        radial_count = np.zeros(max_r + 1)
        
        mask = r_rounded <= max_r
        np.add.at(radial_sum, r_rounded[mask], matrix[mask])
        np.add.at(radial_count, r_rounded[mask], 1)
        
        radial_avg = np.divide(radial_sum, radial_count, out=np.zeros_like(radial_sum), where=radial_count > 0)
        
        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(np.arange(len(radial_avg)), radial_avg, color='red', linewidth=2, label='Average Accumulation')
        plt.fill_between(np.arange(len(radial_avg)), radial_avg, alpha=0.2, color='red')
        
        plt.title("Radial Accumulation Distribution", fontsize=14, pad=15)
        plt.xlabel("Radius (mm)", fontsize=12)
        plt.ylabel("Average Accumulation Time (s)", fontsize=12)
        plt.xlim(0, max_r)
        plt.xticks(np.arange(0, max_r + 1, 10))
        plt.ylim(0, np.max(radial_avg) * 1.1 if np.max(radial_avg) > 0 else 1.0)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()