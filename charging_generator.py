import numpy as np
import math
import os
import matplotlib.pyplot as plt
import tkinter as tk
from numba import njit, prange

from simulation_engine import SimulationEngine
from models import DispenseArm
from constants import (
    ARM_GEOMETRIES, WAFER_RADIUS, REPORT_FPS, 
    GRID_SIZE, CHARGING_BASE_SPIN_DECAY,
    VACUUM_PERMITTIVITY, WATER_RELATIVE_PERMITTIVITY, DEFAULT_CONDUCTIVITY
)

# ==========================================
# Numba Kernel 1: 電荷分離 (Charge Separation)
# ==========================================
@njit(fastmath=True, cache=True)
def _numba_deposit_and_separate_charge(surface_charge, liquid_charge, film_matrix, 
                                       pos_x, pos_y, vel_x, vel_y, 
                                       radius, grid_size, dt, 
                                       dynamic_eff):
    center_offset = 150.0
    idx_x, idx_y = pos_x + center_offset, pos_y + center_offset
    r_pixel = int(math.ceil(radius))
    
    # 邊界限制
    min_i, max_i = int(max(0, idx_x - r_pixel)), int(min(grid_size - 1, idx_x + r_pixel))
    min_j, max_j = int(max(0, idx_y - r_pixel)), int(min(grid_size - 1, idx_y + r_pixel))
    
    speed = math.sqrt(vel_x**2 + vel_y**2)
    q_gen = dynamic_eff * speed * dt

    radius_sq = radius**2
    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            if (float(i) - idx_x)**2 + (float(j) - idx_y)**2 <= radius_sq:
                if film_matrix[i, j] > 1e-5:
                    # [關鍵邏輯]：表面獲得電荷 (Fixed)，液體獲得反向電荷 (Mobile)
                    surface_charge[i, j] += q_gen  
                    liquid_charge[i, j] -= q_gen

# ==========================================
# Numba Kernel 2: 表面擴散 (Surface Diffusion)
# ==========================================
@njit(fastmath=True, parallel=True, cache=True)
def _numba_diffuse_surface(surf_in, surf_out, diff_coeff, dt, grid_size):
    # alpha = D * dt / (dx^2)，此處假設 dx=1
    alpha = min(0.25, diff_coeff * dt) 
    
    for i in prange(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            # 2D Laplacian 算子
            laplacian = (surf_in[i+1, j] + surf_in[i-1, j] + 
                         surf_in[i, j+1] + surf_in[i, j-1] - 
                         4.0 * surf_in[i, j])
            surf_out[i, j] = surf_in[i, j] + alpha * laplacian

# ==========================================
# Numba Kernel 3: 電荷演化 (Relaxation & Transport)
# ==========================================
@njit(fastmath=True, parallel=True, cache=True)
def _numba_evolve_charge(charge_matrix, film_matrix, dt, 
                         conductivity, relative_permittivity, 
                         wafer_radius, spin_decay_rate):
    """
    物理機制: 介電鬆弛 (Dielectric Relaxation) 與 物理傳輸
    邏輯:
    1. 電荷會穿過液膜流向晶圓基板 (接地)。
    2. 電荷會隨著液體被甩出晶圓邊緣。
    """
    rows, cols = charge_matrix.shape
    epsilon = relative_permittivity * VACUUM_PERMITTIVITY
    
    # [關鍵物理計算 2]: 介電鬆弛時間 (Relaxation Time)
    # tau = epsilon / sigma
    # 導電率(sigma)越低，tau 越大，電荷消散越慢 (累積越多)
    sigma = max(conductivity, 1e-12) # 避免除以零
    tau = epsilon / sigma
    
    # 衰減因子 (Exponential Decay)
    relax_factor = math.exp(-dt / tau)
    
    center_offset = 150.0

    for i in prange(rows):
        for j in range(cols):
            q = charge_matrix[i, j]
            h = film_matrix[i, j]
            
            if q != 0:
                # 1. 介電鬆弛 (電荷流向 Substrate)
                # 只有當有液膜連接到地面時才會發生 (簡化模型)
                if h > 1e-6:
                    q *= relax_factor
                
                # 2. 物理甩乾 (Spin-off)
                # 電荷附著在液體上，液體被甩走，電荷也跟著走
                dx = i - center_offset
                dy = j - center_offset
                r = math.sqrt(dx*dx + dy*dy)
                
                # 簡單模擬液膜變薄帶走電荷
                if r <= wafer_radius:
                    # 邊緣甩得快
                    local_decay = spin_decay_rate * (1.0 + r/wafer_radius)
                    q *= (1.0 - local_decay * dt)
                else:
                    q = 0.0 # 離開晶圓
                
                charge_matrix[i, j] = q
            
            # 同步更新簡易膜厚 (為了計算電位用)
            if h > 0:
                dx = i - center_offset
                dy = j - center_offset
                r = math.sqrt(dx*dx + dy*dy)
                if r <= wafer_radius:
                     local_decay = spin_decay_rate * (1.0 + r/wafer_radius)
                     film_matrix[i, j] *= (1.0 - local_decay * dt)
                else:
                     film_matrix[i, j] = 0.0

class ChargingGenerator:
    def __init__(self, app_instance):
        self.app = app_instance

    def run_fast_simulation(self, recipe, config):
        """
        Fast simulation specifically for tuning, returns radial average of surface potential
        """
        cond = config.get('FLUID_CONDUCTIVITY', 5.0e-12)
        perm = config.get('FLUID_RELATIVE_PERMITTIVITY', 80.0)
        eff_base = config.get('CHARGING_EFFICIENCY', -1.0e-10)
        rpm_factor = config.get('CHARGING_RPM_FACTOR', 5.0)
        diff_coeff = config.get('SURFACE_DIFFUSION_COEFF', 0.1)
        base_spin_decay = config.get('CHARGING_BASE_SPIN_DECAY', 2.0)
        
        headless_arms = {}
        for i, geo in ARM_GEOMETRIES.items():
            if i == 2:
                headless_arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], None, None,
                                           side_arm_length=geo.get('side_arm_length'), 
                                           side_arm_angle_offset=geo.get('side_arm_angle_offset'),
                                           side_arm_branch_dist=geo.get('side_arm_branch_dist'))
            else:
                headless_arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], None, None)
                
        water_params = self.app._get_water_params() if self.app else {
            'viscosity': 1.0e-3, 'surface_tension': 72.8e-3, 'evaporation_rate': 0.0
        }
        wp_dict = {1: water_params, 2: water_params, 3: water_params}
        
        engine = SimulationEngine(recipe, headless_arms, wp_dict, headless=True, config=config)
        
        grid_size = 300
        surface_charge = np.zeros((grid_size, grid_size), dtype=np.float64)
        surface_buffer = np.zeros((grid_size, grid_size), dtype=np.float64)
        liquid_charge = np.zeros((grid_size, grid_size), dtype=np.float64)
        film_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)

        report_fps = recipe.get('dynamic_report_fps', REPORT_FPS)
        dt = 1.0 / report_fps
        total_duration = sum(p['total_duration'] for p in recipe['processes'])
        sim_clock = 0.0

        while True:
            snapshot = engine.update(dt)
            sim_clock += dt
            curr_rpm = abs(snapshot.get('rpm', 0))

            dynamic_eff = eff_base * (1.0 + (curr_rpm / 1000.0)**2 * rpm_factor)
            
            on_wafer_mask = (engine.particles_state == 2)
            if np.any(on_wafer_mask):
                indices = np.where(on_wafer_mask)[0]
                current_proc = recipe['processes'][snapshot['process_idx']]
                for idx in indices:
                    pos = engine.particles_pos[idx]
                    
                    p_arm_id = engine.particles_arm_id[idx]
                    if p_arm_id == 3:
                        actual_flow = current_proc.get('flow_rate_2', 500.0)
                    else:
                        actual_flow = current_proc.get('flow_rate', 500.0)
                    
                    flow_scale = actual_flow / 500.0
                    
                    self._simple_deposit_film(film_matrix, pos[0], pos[1], 2.0, 0.005 * flow_scale)
                    
                    vel = engine.particles_vel[idx]
                    _numba_deposit_and_separate_charge(
                        surface_charge, liquid_charge, film_matrix,
                        pos[0], pos[1], vel[0], vel[1],
                        2.0, grid_size, dt,
                        dynamic_eff * flow_scale
                    )

            _numba_diffuse_surface(surface_charge, surface_buffer, diff_coeff, dt, 300)
            surface_charge[:] = surface_buffer[:]

            rpm = snapshot.get('rpm', 0)
            current_spin_decay = base_spin_decay * (1.0 + abs(rpm)/500.0)
            
            _numba_evolve_charge(
                liquid_charge, film_matrix, dt,
                cond, perm, WAFER_RADIUS, current_spin_decay
            )

            if snapshot.get('is_finished') or sim_clock > total_duration + 2.0:
                break
                
        potential_map = self._calculate_potential(surface_charge, config)
        radial_avg = self.calculate_radial_average(potential_map)
        return radial_avg, potential_map

    def calculate_radial_average(self, matrix):
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
        return radial_avg

    def generate(self, recipe, filepath, config=None, progress_widgets=None, play_speed_multiplier=1.0):
        """
        執行電荷累積模擬 (解耦雙電層模型 Decoupled EDL Model)
        """
        # 1. 讀取設定
        if config is None:
            from simulation_config_def import get_default_config
            config = get_default_config()
        
        # 關鍵參數讀取
        cond = config.get('FLUID_CONDUCTIVITY', 5.0e-12)
        perm = config.get('FLUID_RELATIVE_PERMITTIVITY', 80.0)
        eff_base = config.get('CHARGING_EFFICIENCY', -1.0e-10) # TEOS 通常為負
        rpm_factor = config.get('CHARGING_RPM_FACTOR', 5.0)    # RPM 增強因子
        diff_coeff = config.get('SURFACE_DIFFUSION_COEFF', 0.1) # 擴散係係數
        base_spin_decay = config.get('CHARGING_BASE_SPIN_DECAY', 2.0)
        
        # 2. 初始化模擬引擎
        # 為了獨立運作，我們需要自己的 SimulationEngine 來跑粒子軌跡
        headless_arms = {}
        for i, geo in ARM_GEOMETRIES.items():
            if i == 2:
                headless_arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], None, None,
                                           side_arm_length=geo.get('side_arm_length'), 
                                           side_arm_angle_offset=geo.get('side_arm_angle_offset'),
                                           side_arm_branch_dist=geo.get('side_arm_branch_dist'))
            else:
                headless_arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], None, None)
                
        water_params = self.app._get_water_params() # 沿用主程式的水參數
        
        # 為了相容性，簡單包裝
        wp_dict = {1: water_params, 2: water_params, 3: water_params}
        
        engine = SimulationEngine(recipe, headless_arms, wp_dict, headless=True, config=config)
        
        # 3. 初始化網格 (三層矩陣)
        grid_size = 300
        self.surface_charge = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.surface_buffer = np.zeros((grid_size, grid_size), dtype=np.float64) # 擴散緩衝
        self.liquid_charge = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.film_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)

        # 影片同步設定
        VIDEO_FPS = 30.0
        record_interval = (1.0 / VIDEO_FPS) * play_speed_multiplier
        next_record_time = 0.0
        video_buffer = []

        report_fps = recipe.get('dynamic_report_fps', REPORT_FPS)
        dt = 1.0 / report_fps
        total_duration = sum(p['total_duration'] for p in recipe['processes'])
        sim_clock = 0.0

        # 新增：控制進度條顯示更新的頻率 (例如每 0.5 秒更新一次進度條上的文字 and 百分比)
        progress_display_interval = 0.5
        last_progress_display_time = 0.0 # 上次更新進度條顯示的時間

        # 新增：在循環開始前，為 JIT 編譯提供提示，並強制刷新 GUI
        if progress_widgets:
            progress_widgets['label'].config(text="Initializing JIT Engine for Charging (first run might be slow)...")
            # 確保 progress_widgets['bar'] 的最大值已經設定
            progress_widgets['bar']['maximum'] = total_duration
            progress_widgets['window'].update_idletasks() # 強制刷新 GUI
        
        print(f"Starting Charging Simulation (Cond={cond:.2e} S/m)...")

        # 4. 主迴圈
        while True:
            # 更新粒子物理
            snapshot = engine.update(dt)
            sim_clock += dt
            curr_rpm = abs(snapshot.get('rpm', 0))

            # [改良點 1]：計算隨轉速非線性成長的生成效率
            dynamic_eff = eff_base * (1.0 + (curr_rpm / 1000.0)**2 * rpm_factor)
            
            # 影片快照
            if sim_clock >= next_record_time:
                video_buffer.append({
                    'surface_charge': self.surface_charge.copy(),
                    'liquid_charge': self.liquid_charge.copy(),
                    'film': self.film_matrix.copy(),
                    'time': sim_clock
                })
                next_record_time += record_interval

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
                        p_label.config(text=f"Charging: {sim_clock:.1f}s / {total_duration:.1f}s ({percent:.0f}%)")
                        
                        # 強制刷新 GUI，讓進度條視窗有機會處理事件和繪製更新
                        progress_widgets['window'].update_idletasks()
                        
                        last_progress_display_time = sim_clock # 更新上次顯示時間
                    except tk.TclError as e: # 捕獲使用者關閉進度視窗時可能發生的錯誤
                        print(f"Charging progress window closed by user during GUI update: {e}, stopping generation.")
                        return False # 返回 False 表示生成被取消
                    except Exception as e:
                        print(f"Error updating Charging progress bar: {e}")
                        return False # 返回 False 表示生成失敗

            # --- A. 簡易液膜生成 (為了支撐電荷計算) ---
            on_wafer_mask = (engine.particles_state == 2) # P_ON_WAFER
            if np.any(on_wafer_mask):
                indices = np.where(on_wafer_mask)[0]
                current_proc = recipe['processes'][snapshot['process_idx']]
                for idx in indices:
                    pos = engine.particles_pos[idx]
                    
                    # [修正] 考慮噴嘴流量對液膜與電荷的影響
                    p_arm_id = engine.particles_arm_id[idx]
                    if p_arm_id == 3:
                        actual_flow = current_proc.get('flow_rate_2', 500.0)
                    else:
                        actual_flow = current_proc.get('flow_rate', 500.0)
                    
                    # 流量基礎係數 (以 500mL/min 為基準)
                    flow_scale = actual_flow / 500.0
                    
                    # A. 沉積液膜
                    self._simple_deposit_film(self.film_matrix, pos[0], pos[1], 2.0, 0.005 * flow_scale)
                    
                    # B. 電荷分離沉積
                    vel = engine.particles_vel[idx]
                    _numba_deposit_and_separate_charge(
                        self.surface_charge, self.liquid_charge, self.film_matrix,
                        pos[0], pos[1], vel[0], vel[1],
                        2.0, grid_size, dt,
                        dynamic_eff * flow_scale
                    )

            # --- C. 表面擴散 (平滑化) [改良點 3] ---
            _numba_diffuse_surface(self.surface_charge, self.surface_buffer, diff_coeff, dt, 300)
            self.surface_charge[:] = self.surface_buffer[:] # 更新回原矩陣

            # --- D. 演化液體電荷 (Spin-off) [改良點 4] ---
            rpm = snapshot.get('rpm', 0)
            current_spin_decay = base_spin_decay * (1.0 + abs(rpm)/500.0)
            
            _numba_evolve_charge(
                self.liquid_charge, self.film_matrix, dt,
                cond, perm, WAFER_RADIUS, current_spin_decay
            )

            if snapshot.get('is_finished') or sim_clock > total_duration + 2.0:
                break
        
        # 5. 結果輸出
        self._export_results(self.surface_charge, self.film_matrix, filepath, perm, config, video_buffer, VIDEO_FPS)
        return True

    @staticmethod
    @njit(fastmath=True)
    def _simple_deposit_film(matrix, x, y, r, val):
        cx, cy = x + 150.0, y + 150.0
        ri = int(r)
        for i in range(int(cx-ri), int(cx+ri+1)):
            for j in range(int(cy-ri), int(cy+ri+1)):
                if 0 <= i < 300 and 0 <= j < 300:
                    if (i-cx)**2 + (j-cy)**2 <= r*r:
                        matrix[i, j] += val

    def _export_results(self, charge_Q, film_H, filepath, rel_perm, config, video_buffer, fps):
        base_path, _ = os.path.splitext(filepath)
        # 檔名處理
        real_base = filepath.replace("_Charging.png", "")
        radial_png_path = f"{real_base}_Charging_Radial_Distribution.png"
        video_path = f"{real_base}_Charging_Simulation.mp4"

        # 計算電位矩陣 (基於表面電荷與等效電容)
        potential_map = self._calculate_potential(charge_Q, config)

        # 1. 輸出 Heatmap PNG
        self._export_potential_map(potential_map, filepath, config)

        # 2. 輸出 Radial Distribution
        self._export_radial_distribution(potential_map, radial_png_path)

        # 3. 輸出影片
        self._export_charging_video(video_buffer, video_path, config, fps)

    def _calculate_potential(self, surface_Q, config):
        # KPFM 量測的是乾燥後的表面殘留電位
        # V = Q_surface / C_kpfm
        kpfm_cap = config.get('KPFM_CAPACITANCE', 1.0e-10) # 用於校準伏特數值的縮放因子
        potential_map = surface_Q / kpfm_cap
        return potential_map

    def _export_potential_map(self, potential_map, filepath, current_config):
        v_max = np.max(potential_map)
        v_min = np.min(potential_map)
        abs_max = max(abs(v_max), abs(v_min))
        if abs_max == 0: abs_max = 1.0
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(potential_map.T, 
                        origin='lower', 
                        cmap='seismic_r', 
                        extent=[-150, 150, -150, 150],
                        vmin=-abs_max, vmax=abs_max)
        
        cbar = plt.colorbar(im)
        cbar.set_label('Surface Potential (Volts)')
        plt.title('Simulated Wafer Surface Potential')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')

        # 參數資訊
        params_lines = []
        from simulation_config_def import PARAMETER_DEFINITIONS
        for category, params in PARAMETER_DEFINITIONS.items():
            for key, info in params.items():
                label = info[0]
                val = current_config.get(key, info[1])
                params_lines.append(f"{label}: {val}")
        params_text = "\n".join(params_lines)
        stats_text = (
            f"Max: {v_max:.4f}V\n"
            f"Min: {v_min:.4f}V\n"
            f"Range: {abs(v_max-v_min):.4f}V\n"
            f"------------------\n"
            f"{params_text}"
        )
        plt.text(-145, -145, stats_text, color='white', fontsize=7,
                family='monospace', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
        
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()

    def _export_radial_distribution(self, matrix, filepath):
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
        plt.plot(np.arange(len(radial_avg)), radial_avg, color='red', linewidth=2, label='Avg Potential')
        plt.fill_between(np.arange(len(radial_avg)), radial_avg, alpha=0.2, color='red')
        plt.title("Radial Surface Potential Distribution", fontsize=14, pad=15)
        plt.xlabel("Radius (mm)", fontsize=12)
        plt.ylabel("Potential (Volts)", fontsize=12)
        plt.xlim(0, max_r)
        plt.grid(True, linestyle='--', alpha=0.7)

        v_max = np.max(radial_avg)
        v_min = np.min(radial_avg)
        stats_text = f"Max: {v_max:.4f}V\nMin: {v_min:.4f}V"
        plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes,
                color='red', fontsize=10, family='monospace', fontweight='bold',
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()

    def _export_charging_video(self, video_buffer, output_path, config, fps):
        import cv2
        if not video_buffer: return

        view_size = 400
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (view_size, view_size))

        mask = np.zeros((view_size, view_size), dtype=np.uint8)
        cv2.circle(mask, (view_size//2, view_size//2), view_size//2, 255, -1)

        # 預計算最終最大值作為歸一化基準
        final_potential = self._calculate_potential(video_buffer[-1]['surface_charge'], config)
        v_max_final = np.max(final_potential)
        v_min_final = np.min(final_potential)
        abs_max_global = max(abs(v_max_final), abs(v_min_final), 1e-5)

        print(f"Exporting Charging Video...")
        for frame_data in video_buffer:
            p_map = self._calculate_potential(frame_data['surface_charge'], config)
            
            # 歸一化到 0-255，且 0V 剛好在中間 (127)
            # (val - (-abs_max)) / (2 * abs_max) * 255
            norm_map = ((p_map.T + abs_max_global) / (2 * abs_max_global) * 255)
            norm_map = np.clip(norm_map, 0, 255).astype(np.uint8)
            
            # 使用 seismic_r 對應的 OpenCV 色階 (此處手動模擬或使用 COLORMAP_JET)
            color_view = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
            color_view = cv2.resize(color_view, (view_size, view_size), interpolation=cv2.INTER_LINEAR)
            color_view = cv2.bitwise_and(color_view, color_view, mask=mask)
            
            # 加上時間文字
            cv2.putText(color_view, f"Time: {frame_data['time']:.1f}s", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(color_view)

        out.release()
