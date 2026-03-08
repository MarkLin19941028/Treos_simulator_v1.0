import numpy as np
import math
import os
import matplotlib.pyplot as plt
import tkinter as tk
from numba import njit # [新增]

from models import DispenseArm
from simulation_engine import SimulationEngine
from constants import (
    ARM_GEOMETRIES, WAFER_RADIUS, REPORT_FPS, 
    PRE_ALPHA, PRE_BETA, PRE_GRID_SIZE,
    PRE_Q_REF, PRE_GAMMA_BASE
)

# --- [新增] Numba 加速核心函數 ---
@njit(fastmath=True, cache=True)
def _numba_apply_pre_kernel(matrix, center_x, center_y, contribution, radius, grid_size):
    idx_x = center_x + 150.0
    idx_y = center_y + 150.0
    
    r_pixel = int(math.ceil(radius))
    
    # 邊界檢查
    min_i = max(0, int(math.floor(idx_x - r_pixel)))
    max_i = min(grid_size - 1, int(math.ceil(idx_x + r_pixel)))
    min_j = max(0, int(math.floor(idx_y - r_pixel)))
    max_j = min(grid_size - 1, int(math.ceil(idx_y + r_pixel)))

    radius_sq = radius * radius
    
    # --- [Step 1] 計算總權重 (Total Weight) ---
    total_weight = 0.0
    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            dist_sq = (i - idx_x)**2 + (j - idx_y)**2
            if dist_sq <= radius_sq:
                # 建議改用高斯分佈，更加自然
                # Gaussian: exp(-dist^2 / (2 * sigma^2)), let sigma = radius / 2
                sigma = radius / 2.0
                w = math.exp(-dist_sq / (2 * sigma * sigma))
                total_weight += w
    
    # 避免除以零
    if total_weight <= 0.0:
        return

    # --- [Step 2] 歸一化並累加能量 ---
    normalized_contrib = contribution / total_weight
    
    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            dist_sq = (i - idx_x)**2 + (j - idx_y)**2
            if dist_sq <= radius_sq:
                sigma = radius / 2.0
                w = math.exp(-dist_sq / (2 * sigma * sigma))
                
                # 這樣確保了所有像素增加的總量 = contribution
                matrix[i, j] += normalized_contrib * w

class PREGenerator:
    def __init__(self, app_instance):
        self.app = app_instance

    def generate(self, recipe, filepath, config=None, progress_widgets=None):
        """
        Cleaning Dose 模擬邏輯 (Numba 加速版)
        """
        if config is None:
            from simulation_config_def import get_default_config
            config = get_default_config()

        pre_alpha = config.get('PRE_ALPHA', PRE_ALPHA)
        pre_beta = config.get('PRE_BETA', PRE_BETA)
        pre_grid_radius = config.get('PRE_GRID_SIZE', PRE_GRID_SIZE)
        pre_q_ref = config.get('PRE_Q_REF', PRE_Q_REF)
        pre_gamma_base = config.get('PRE_GAMMA_BASE', PRE_GAMMA_BASE)

        headless_arms = {i: DispenseArm(i, geo['pivot'], geo['home'], geo['length'], geo['p_start'], geo['p_end'], None, None) 
                         for i, geo in ARM_GEOMETRIES.items()}

        water_params = self.app._get_water_params()
        water_params_dict = {i: {
            'viscosity': water_params['viscosity'],
            'surface_tension': water_params['surface_tension'],
            'evaporation_rate': water_params['evaporation_rate']
        } for i in [1, 2, 3]}

        engine = SimulationEngine(recipe, headless_arms, water_params_dict, headless=True, config=config)
        
        grid_size = 300
        dose_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)
        
        report_fps = recipe.get('dynamic_report_fps', REPORT_FPS)
        dt = 1.0 / report_fps
        total_duration = sum(p['total_duration'] for p in recipe['processes'])
        sim_clock = 0.0

        # 新增：控制進度條顯示更新的頻率 (例如每 0.5 秒更新一次進度條上的文字 and 百分比)
        progress_display_interval = 0.5
        last_progress_display_time = 0.0 # 上次更新進度條顯示的時間

        # 新增：在循環開始前，為 JIT 編譯提供提示，並強制刷新 GUI
        if progress_widgets:
            progress_widgets['label'].config(text="Initializing JIT Engine for PRE (first run might be slow)...")
            # 確保 progress_widgets['bar'] 的最大值已經設定
            progress_widgets['bar']['maximum'] = total_duration
            progress_widgets['window'].update_idletasks() # 強制刷新 GUI

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
                        p_label.config(text=f"Dose Simulation (Accelerated): {sim_clock:.1f}s / {total_duration:.1f}s ({percent:.0f}%)")
                        
                        # 強制刷新 GUI，讓進度條視窗有機會處理事件 and 繪製更新
                        progress_widgets['window'].update_idletasks()
                        
                        last_progress_display_time = sim_clock # 更新上次顯示時間
                    except tk.TclError as e: # 捕獲使用者關閉進度視窗時可能發生的錯誤
                        print(f"PRE progress window closed by user during GUI update: {e}, stopping generation.")
                        return False # 返回 False 表示生成被取消
                    except Exception as e:
                        print(f"Error updating PRE progress bar: {e}")
                        return False # 返回 False 表示生成失敗

            current_proc = recipe['processes'][snapshot['process_idx']]
            q_actual = current_proc.get('flow_rate', pre_q_ref)
            flow_ratio = q_actual / pre_q_ref
            c_q = math.sqrt(flow_ratio) 
            g_q = 1.0 / math.sqrt(flow_ratio) if flow_ratio > 0 else 1.0 
            gamma_eff = pre_gamma_base * g_q

            current_rpm = snapshot['rpm']
            omega = (current_rpm / 60.0) * 2 * math.pi
            
            # 優化：直接從引擎的 NumPy 陣列提取 (現在引擎直接提供旋轉座標系下的座標)
            on_wafer_mask = engine.particles_state == 2 # P_ON_WAFER
            if np.any(on_wafer_mask):
                indices = np.where(on_wafer_mask)[0]
                for i in indices:
                    # 1. 取得旋轉座標系座標
                    center_x, center_y = engine.particles_pos[i, 0], engine.particles_pos[i, 1]
                    
                    r_val = math.sqrt(center_x**2 + center_y**2)
                    
                    # 2. 瞬時強度計算
                    shear_part = pre_alpha * (abs(omega) ** 1.5) * r_val
                    impact_part = pre_beta * c_q
                    k_raw = shear_part + impact_part
                    
                    # 3. 有效劑量因子
                    eta = math.exp(-gamma_eff * r_val)
                    dose_contribution = k_raw * eta * dt
                    
                    # 4. [修改] 呼叫 Numba 核心
                    _numba_apply_pre_kernel(
                        dose_matrix, 
                        center_x, center_y, 
                        dose_contribution, 
                        pre_grid_radius, 
                        grid_size
                    )

            if snapshot.get('is_finished') or sim_clock > (total_duration + 10.0):
                break

        # --- [新增] 蒙地卡羅缺陷模擬流程 ---
        pre_defect_count = int(config.get('PRE_DEFECT_COUNT', 10000))
        defectmap_cali = config.get('PRE_DEFECT_CALI', 0.5)

        if progress_widgets:
            try:
                progress_widgets['label'].config(text="Running Monte Carlo Defect Prediction...")
                progress_widgets['window'].update_idletasks()
            except:
                pass

        incoming_defects = self._generate_incoming_defects(pre_defect_count)
        final_defects = self._simulate_defect_survival(incoming_defects, dose_matrix, defectmap_cali)

        self._export_results(dose_matrix, final_defects, filepath, config=config)
        return True

    def _generate_incoming_defects(self, count):
        """
        模擬進站原始缺陷：包含座標與符合冪律分佈的粒徑 (dp)
        """
        # 產生符合 Wafer 範圍內 (150mm) 的隨機座標
        phis = np.random.uniform(0, 2 * np.pi, count)
        rs = np.sqrt(np.random.uniform(0, 150**2, count))
        xs = rs * np.cos(phis)
        ys = rs * np.sin(phis)
        
        # 粒徑分布 (dp): 模擬 Fab 常見的小粒子多、大粒子少
        # 使用 log-normal 分佈，平均直徑約在 12nm ~ 30nm 區間 (exp(2.5) ~ 12, exp(3.0) ~ 20)
        dp = np.random.lognormal(mean=2.8, sigma=0.4, size=count) 
        
        return np.stack((xs, ys, dp), axis=1)

    def _simulate_defect_survival(self, incoming_defects, dose_matrix, cali_a):
        """
        核心判定邏輯：P_survive = exp(-cali_a * Dose / D_crit)
        """
        survived = []
        grid_size = dose_matrix.shape[0]
        
        for x, y, dp in incoming_defects:
            # 座標轉索引 (150mm -> index 150, range -150~150 -> 0~300)
            ix = int(np.clip(x + 150, 0, grid_size - 1))
            iy = int(np.clip(y + 150, 0, grid_size - 1))
            
            local_dose = dose_matrix[ix, iy]
            
            # 物理阻力模型：小粒子 D_crit 越高 (越難洗)
            # 這裡假設抵抗力與粒徑成反比 (或者說清洗效率與粒徑成正比)
            d_crit = 1.0 / (math.sqrt(dp) + 1e-6)
            
            # 計算殘留機率
            p_survive = math.exp(-cali_a * local_dose / d_crit)
            
            # 蒙地卡羅隨機判定
            if np.random.random() < p_survive:
                survived.append([x, y, dp])
                
        return np.array(survived)
    
    def _export_results(self, matrix, final_defects, filepath, config=None):
        base_path, _ = os.path.splitext(filepath)
        png_path = filepath
        real_base = base_path.replace("_Cleaning_Dose", "")
        csv_path = f"{real_base}_Cleaning_Dose_RawData.csv"
        radial_png_path = f"{real_base}_Cleaning_Dose_Radial_Distribution.png"
        
        data = matrix.T

        # 提取參數用於顯示
        alpha_val = config.get('PRE_ALPHA', PRE_ALPHA) if config else PRE_ALPHA
        beta_val = config.get('PRE_BETA', PRE_BETA) if config else PRE_BETA
        gamma_base_val = config.get('PRE_GAMMA_BASE', PRE_GAMMA_BASE) if config else PRE_GAMMA_BASE
        impact_rad_val = config.get('PRE_GRID_SIZE', PRE_GRID_SIZE) if config else PRE_GRID_SIZE
        q_ref_val = config.get('PRE_Q_REF', PRE_Q_REF) if config else PRE_Q_REF

        # 1. 繪製並儲存 PNG
        plt.figure(figsize=(11, 9), dpi=120)
        im = plt.imshow(
            data,
            origin='lower',
            extent=[-150, 150, -150, 150],
            cmap='viridis',
            interpolation='bilinear'
        )
        
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Accumulated Effective Cleaning Dose (A.U.)')

        wafer_circle = plt.Circle((0, 0), 150, color='red', fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_artist(wafer_circle)

        plt.title("Wafer Cleaning Dose Distribution (Redeposition Model)", fontsize=14, pad=15)
        plt.xlabel("X Position (mm)")
        plt.ylabel("Y Position (mm)")

        if data.size > 0 and np.any(data > 0):
            valid_data = data[data > 0]
            h_max = np.max(data)
            h_min = np.min(valid_data)
            h_mean = np.mean(valid_data)
            h_uni = (h_max - h_min) / (2 * h_mean) * 100 if h_mean > 0 else 0.0
        else:
            h_max = h_min = h_mean = h_uni = 0.0

        # 獲取 Physics & System 參數
        if config is None:
            from simulation_config_def import get_default_config
            config = get_default_config()
            
        # 整理所有參數資訊
        params_lines = []
        from simulation_config_def import PARAMETER_DEFINITIONS
        for category, params in PARAMETER_DEFINITIONS.items():
            for key, info in params.items():
                label = info[0]
                val = config.get(key, info[1])
                params_lines.append(f"{label}: {val}")
        params_text = "\n".join(params_lines)

        stats_text = (
            f"Max: {h_max:.4f}\n"
            f"Min(>0): {h_min:.4f}\n"
            f"Uniformity: {h_uni:.2f}%\n"
            f"------------------\n"
            f"{params_text}"
        )
        plt.text(-145, -145, stats_text, color='white', fontsize=8,
                family='monospace', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

        plt.tight_layout()
        plt.savefig(png_path, bbox_inches='tight', dpi=300)
        plt.close()

        # 2. 儲存 CSV
        try:
            header_str = (f"Cleaning Dose Data (Redeposition Model), Q_ref: {q_ref_val}mL/min, "
                         f"Gamma_base: {gamma_base_val}, Impact Radius: {impact_rad_val}mm")
            np.savetxt(csv_path, data, delimiter=",", fmt='%.6f', header=header_str)
        except Exception as e:
            print(f"Failed to write CSV: {e}")

        # 3. 輸出徑向分佈圖 (Radial Distribution)
        self._export_radial_distribution(matrix, radial_png_path)

        # 4. [新增] 輸出缺陷圖 (Defect Map)
        pre_defect_count = int(config.get('PRE_DEFECT_COUNT', 10000))
        self._export_defect_map(final_defects, filepath, pre_defect_count)

    def _export_defect_map(self, points, filepath, total_incoming):
        base_path, _ = os.path.splitext(filepath)
        real_base = base_path.replace("_Cleaning_Dose", "")
        map_path = f"{real_base}_Defect_Map.png"
        
        plt.figure(figsize=(10, 10), dpi=150)
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_facecolor('#f8f9fa') # 淺灰背景
        
        # 畫出存活的點
        if len(points) > 0:
            # 點大小隨粒徑縮放，並限制最小/最大視覺大小
            sizes = np.clip(points[:, 2] * 0.3, 1, 50)
            plt.scatter(points[:, 0], points[:, 1], 
                        s=sizes,
                        c='red', alpha=0.6, edgecolors='none', label='Remaining Defects')
        
        # 繪製晶圓邊界
        wafer = plt.Circle((0, 0), 150, color='#007bff', fill=False, lw=2, alpha=0.5)
        ax.add_artist(wafer)
        
        # 設定座標軸範圍
        plt.xlim(-160, 160)
        plt.ylim(-160, 160)
        plt.grid(True, linestyle=':', alpha=0.3)

        # 顯示結果文字
        rem_count = len(points)
        pre_val = (1 - rem_count / total_incoming) * 100 if total_incoming > 0 else 0.0
        plt.title(f"Predicted Defect Map\nRemaining: {rem_count} / {total_incoming} (PRE: {pre_val:.2f}%)", 
                  fontsize=14, pad=10)
        
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        
        plt.tight_layout()
        plt.savefig(map_path, dpi=200)
        plt.close()

    def _export_radial_distribution(self, matrix, filepath):
        grid_size = matrix.shape[0]
        center = grid_size / 2.0
        y, x = np.indices(matrix.shape)
        r = np.sqrt((x - center + 0.5)**2 + (y - center + 0.5)**2)
        r_rounded = r.astype(int)
        max_r = 150
        radial_sum = np.zeros(max_r + 1)
        radial_count = np.zeros(max_r + 1)
        mask = r_rounded <= max_r
        np.add.at(radial_sum, r_rounded[mask], matrix[mask])
        np.add.at(radial_count, r_rounded[mask], 1)
        radial_avg = np.divide(radial_sum, radial_count, out=np.zeros_like(radial_sum), where=radial_count > 0)
        
        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(np.arange(len(radial_avg)), radial_avg, color='blue', linewidth=2, label='Average Dose')
        plt.fill_between(np.arange(len(radial_avg)), radial_avg, alpha=0.2, color='blue')
        plt.title("Radial Cleaning Dose Distribution (Redeposition Model)", fontsize=14, pad=15)
        plt.xlabel("Radius (mm)", fontsize=12)
        plt.ylabel("Average Cleaning Dose (A.U.)", fontsize=12)
        plt.xlim(0, max_r)
        plt.xticks(np.arange(0, max_r + 1, 10))
        plt.ylim(0, np.max(radial_avg) * 1.1 if np.max(radial_avg) > 0 else 1.0)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 統計資訊 (針對徑向分布資料)
        if radial_avg.size > 0 and np.any(radial_avg > 0):
            valid_r = radial_avg[radial_avg > 0]
            r_max = np.max(radial_avg)
            r_min = np.min(valid_r)
            r_mean = np.mean(valid_r)
            r_uni = (r_max - r_min) / (2 * r_mean) * 100 if r_mean > 0 else 0.0
            
            stats_text = (
                f"Max: {r_max:.4f}\n"
                f"Min(>0): {r_min:.4f}\n"
                f"Uniformity: {r_uni:.2f}%"
            )
            plt.text(0.02, 0.05, stats_text, transform=plt.gca().transAxes,
                    color='blue', fontsize=10, family='monospace', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='blue'))

        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
