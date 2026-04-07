import numpy as np
import math
import os
import matplotlib.pyplot as plt
import tkinter as tk
from numba import njit

from models import DispenseArm
from simulation_engine import SimulationEngine
from constants import (
    ARM_GEOMETRIES, WAFER_RADIUS, REPORT_FPS, 
    PRE_ALPHA, PRE_BETA, PRE_GRID_SIZE,
    PRE_Q_REF, PRE_GAMMA_BASE
)

# --- [Numba 加速核心 1]：顆粒動態行為演算法 ---
@njit(fastmath=True, cache=True)
def _numba_update_particle_states(particles, instant_matrix, rpm, dt, cali_a, trans_sens, redep_base, st_actual, wafer_radius, grid_size):
    """
    處理顆粒的實時行為：脫附、徑向位移、二次附著
    particles 矩陣結構: [N, 5] -> [x, y, dp, state, d_crit]
    state 定義: 0.0: 附著 (Attached), 1.0: 懸浮遷移 (Suspended), 2.0: 移出晶圓 (Removed)
    """
    count = particles.shape[0]
    omega = (rpm / 60.0) * 2.0 * math.pi
    # 表面張力修正因子 (以純水 72.8 mN/m 為基準)
    f_st = math.sqrt(72.8 / max(st_actual, 1.0))
    
    for i in range(count):
        state = particles[i, 3]
        if state >= 2.0: continue # 已移除則跳過
        
        px = particles[i, 0]
        py = particles[i, 1]
        
        # 取得當前網格 Dose (手動判定邊界以避免 Numba np.clip 標量報錯)
        ix = px + wafer_radius
        if ix < 0: ix = 0
        elif ix > grid_size - 1: ix = grid_size - 1
        
        iy = py + wafer_radius
        if iy < 0: iy = 0
        elif iy > grid_size - 1: iy = grid_size - 1
        
        local_dose = instant_matrix[int(ix), int(iy)]
        
        d_crit = particles[i, 4]
        r_val = math.sqrt(px**2 + py**2)
        
        # --- 狀態 0: 附著態 (判定是否被沖刷脫附) ---
        if state == 0.0:
            # 瞬時脫附機率 (Dose 越高、d_crit 越低，越容易洗起來)
            p_desorp = 1.0 - math.exp(-cali_a * local_dose / d_crit)
            if np.random.random() < p_desorp:
                particles[i, 3] = 1.0 # 轉變為懸浮遷移態
                
        # --- 狀態 1: 懸浮遷移態 (計算位移與可能的二次附著) ---
        elif state == 1.0:
            # 1. 計算徑向位移 Δr (受離心力驅動: ω^2 * r)
            # 1e-6 為經驗物理縮放常數，確保位移量符合時間尺度
            delta_r = (omega**2 * r_val * 1e-6) * trans_sens * dt
            r_new = r_val + delta_r
            
            # 邊界判定：若半徑超過 Wafer Radius 則視為移出晶圓
            if r_new > wafer_radius:
                particles[i, 3] = 2.0
                continue
                
            # 2. 更新座標 (沿原方向向量向外移動)
            ratio = r_new / max(r_val, 1e-6)
            particles[i, 0] = px * ratio
            particles[i, 1] = py * ratio
            
            # 3. 二次附著判定 (Re-deposition)
            # 物理邏輯：在清洗力弱 (Dose低) 且藥液張力高的地方容易重新黏附
            # 因子 10.0 用於放大瞬時 Dose 對附著抑制的敏感度
            p_redep = redep_base * f_st * math.exp(-local_dose * 10.0) 
            if np.random.random() < p_redep:
                particles[i, 3] = 0.0 # 重新變回附著態

# --- [Numba 加速核心 2]：Dose 能量空間分配 ---
@njit(fastmath=True, cache=True)
def _numba_apply_pre_kernel(matrix, center_x, center_y, contribution, radius, grid_size, wafer_radius):
    idx_x = center_x + wafer_radius
    idx_y = center_y + wafer_radius
    r_pixel = int(math.ceil(radius))
    
    min_i = max(0, int(math.floor(idx_x - r_pixel)))
    max_i = min(grid_size - 1, int(math.ceil(idx_x + r_pixel)))
    min_j = max(0, int(math.floor(idx_y - r_pixel)))
    max_j = min(grid_size - 1, int(math.ceil(idx_y + r_pixel)))

    radius_sq = radius * radius
    total_weight = 0.0
    
    # Step 1: 計算範圍內高斯總權重 (歸一化基準)
    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            dist_sq = (i - idx_x)**2 + (j - idx_y)**2
            if dist_sq <= radius_sq:
                sigma = radius / 2.0
                w = math.exp(-dist_sq / (2 * sigma * sigma))
                total_weight += w
    
    if total_weight <= 0.0: return

    # Step 2: 歸一化能量並累加至網格
    normalized_contrib = contribution / total_weight
    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            dist_sq = (i - idx_x)**2 + (j - idx_y)**2
            if dist_sq <= radius_sq:
                sigma = radius / 2.0
                w = math.exp(-dist_sq / (2 * sigma * sigma))
                matrix[i, j] += normalized_contrib * w

class PREGenerator:
    def __init__(self, app_instance):
        self.app = app_instance

    def _generate_incoming_defects(self, count):
        """ 產生初始進站缺陷清單 """
        phis = np.random.uniform(0, 2 * np.pi, count)
        rs = np.sqrt(np.random.uniform(0, WAFER_RADIUS**2, count))
        # 粒徑符合 Log-normal 分佈
        dp = np.random.lognormal(mean=2.8, sigma=0.4, size=count) 
        return np.stack((rs * np.cos(phis), rs * np.sin(phis), dp), axis=1)

    def generate(self, recipe, filepath, config=None, progress_widgets=None):
        """ 標準模擬入口：包含進度條與檔案輸出 """
        dose_matrix, final_defects = self._run_core_simulation(
            recipe, config, progress_widgets=progress_widgets, fast_mode=False
        )
        if dose_matrix is None: return False
        
        # 呼叫輸出方法
        self._export_results(dose_matrix, final_defects, filepath, config=config)
        return True

    def run_fast_simulation(self, recipe, config):
        """ 快速調機入口：無輸出，啟用 fast_mode 加速 """
        dose_matrix, final_defects = self._run_core_simulation(
            recipe, config, progress_widgets=None, fast_mode=True
        )
        return dose_matrix, final_defects, None

    def _run_core_simulation(self, recipe, config=None, progress_widgets=None, fast_mode=False):
        """ 核心物理模擬主迴圈 (與 EtchingAmountGenerator 結構對齊) """
        if config is None:
            from simulation_config_def import get_default_config
            config = get_default_config()

        # 1. 讀取物理參數
        pre_alpha = config.get('PRE_ALPHA', PRE_ALPHA)
        pre_beta = config.get('PRE_BETA', PRE_BETA)
        pre_grid_radius = config.get('PRE_GRID_SIZE', PRE_GRID_SIZE)
        pre_q_ref = config.get('PRE_Q_REF', PRE_Q_REF)
        pre_gamma_base = config.get('PRE_GAMMA_BASE', PRE_GAMMA_BASE)
        pre_defect_count = int(config.get('PRE_DEFECT_COUNT', 10000))
        defectmap_cali = config.get('PRE_DEFECT_CALI', 0.5)
        trans_sens = config.get('PRE_TRANS_SENSITIVITY', 1.0)
        redep_base = config.get('PRE_REDEP_COEFF', 0.05)

        # 2. 動態 FPS 計算 (用於加速)
        if fast_mode:
            max_rpm = 0
            for proc in recipe['processes']:
                spin = proc.get('spin_params', {})
                c_max = spin.get('rpm', 0) if spin.get('mode', 'Simple') == 'Simple' else max(spin.get('start_rpm', 0), spin.get('end_rpm', 0))
                if float(c_max) > max_rpm: max_rpm = float(c_max)
            report_fps = max(30, min(1000, int(max_rpm * 0.5)))
        else:
            report_fps = recipe.get('dynamic_report_fps', REPORT_FPS)
            
        recipe['dynamic_report_fps'] = report_fps
        dt = 1.0 / report_fps

        # 3. 初始化無頭手臂與物理引擎
        headless_arms = {}
        for i, geo in ARM_GEOMETRIES.items():
            if i == 2:
                headless_arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], None, None,
                                           side_arm_length=geo.get('side_arm_length'), 
                                           side_arm_angle_offset=geo.get('side_arm_angle_offset'),
                                           side_arm_branch_dist=geo.get('side_arm_branch_dist'))
            else:
                headless_arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], None, None)

        water_params = self.app._get_water_params() if hasattr(self, 'app') else {'surface_tension': 72.8}
        st_actual = water_params.get('surface_tension', 72.8)
        f_st = math.sqrt(72.8 / max(st_actual, 1.0))

        # Fast Mode 粒子縮放 (確保 Dose 總能量不變)
        fast_particle_scale = 1.0
        if fast_mode:
            max_flow = max([proc.get('flow_rate', 500.0) for proc in recipe['processes']])
            from constants import PARTICLE_SPAWN_MULTIPLIER
            original_rate = max_flow * 0.5 * PARTICLE_SPAWN_MULTIPLIER
            target_rate = max(200.0, original_rate * 0.5)
            fast_particle_scale = min(1.0, target_rate / max(original_rate, 1.0))

        engine = SimulationEngine(
            recipe, headless_arms, {i: water_params for i in [1,2,3]}, 
            headless=True, config=config, fast_mode=fast_mode, fast_particle_scale=fast_particle_scale
        )
        
        grid_size = int(WAFER_RADIUS * 2)
        dose_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)
        
        # 4. 初始化缺陷顆粒
        incoming_raw = self._generate_incoming_defects(pre_defect_count)
        particles_master = np.zeros((pre_defect_count, 5), dtype=np.float64)
        particles_master[:, :3] = incoming_raw
        particles_master[:, 4] = 1.0 / (np.sqrt(incoming_raw[:, 2]) + 1e-6)
        
        total_duration = sum(p['total_duration'] for p in recipe['processes'])
        sim_clock = 0.0
        last_progress_time = 0.0

        if progress_widgets:
            progress_widgets['label'].config(text="Running Dynamic PRE Simulation...")
            progress_widgets['bar']['maximum'] = total_duration
            progress_widgets['window'].update_idletasks()

        # 5. 模擬主迴圈
        while True:
            snapshot = engine.update(dt) 
            sim_clock += dt
            
            if progress_widgets and (sim_clock - last_progress_time >= 0.5 or snapshot.get('is_finished')):
                try:
                    p_bar, p_label = progress_widgets['bar'], progress_widgets['label']
                    p_bar['value'] = min(sim_clock, total_duration)
                    percent = (p_bar['value'] / total_duration) * 100
                    p_label.config(text=f"Transport Sim: {sim_clock:.1f}s ({percent:.0f}%)")
                    progress_widgets['window'].update_idletasks()
                    last_progress_time = sim_clock
                except: return None, None

            current_proc = recipe['processes'][snapshot['process_idx']]
            omega = (snapshot['rpm'] / 60.0) * 2.0 * math.pi
            instant_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)

            # 處理流體對 Dose 的貢獻
            on_wafer_mask = (engine.particles_state == 2)
            if np.any(on_wafer_mask):
                indices = np.where(on_wafer_mask)[0]
                for i in indices:
                    cx, cy = engine.particles_pos[i, 0], engine.particles_pos[i, 1]
                    q_actual = current_proc.get('flow_rate_2' if engine.particles_arm_id[i]==3 else 'flow_rate', 500.0)
                    
                    c_q = math.sqrt(q_actual / pre_q_ref)
                    gamma_eff = (pre_gamma_base / math.sqrt(q_actual / pre_q_ref)) / f_st
                    
                    r_val = math.sqrt(cx**2 + cy**2)
                    k_raw = (pre_alpha * (abs(omega)**1.5) * r_val + pre_beta * c_q) * f_st
                    # Dose 補償：若粒子數減少則單顆能量增加
                    dose_contrib = (k_raw * math.exp(-gamma_eff * r_val) * dt) / fast_particle_scale
                    
                    _numba_apply_pre_kernel(dose_matrix, cx, cy, dose_contrib, pre_grid_radius, grid_size, WAFER_RADIUS)
                    _numba_apply_pre_kernel(instant_matrix, cx, cy, dose_contrib, pre_grid_radius, grid_size, WAFER_RADIUS)

            # 執行 Numba 顆粒狀態更新
            _numba_update_particle_states(
                particles_master, instant_matrix, snapshot['rpm'], dt,
                defectmap_cali, trans_sens, redep_base, st_actual, WAFER_RADIUS, grid_size
            )

            if snapshot.get('is_finished') or sim_clock > (total_duration + 5.0):
                break

        final_defects = particles_master[particles_master[:, 3] < 2.0][:, :3]
        return dose_matrix, final_defects

    def _export_results(self, matrix, final_defects, filepath, config=None):
        """ 處理圖表輸出 """
        base_path, _ = os.path.splitext(filepath)
        real_base = base_path.replace("_Cleaning_Dose", "")
        png_path = filepath
        radial_png_path = f"{real_base}_Cleaning_Dose_Radial_Distribution.png"
        map_path = f"{real_base}_Defect_Map.png"
        csv_path = f"{real_base}_Cleaning_Dose_RawData.csv"
        
        data = matrix.T
        
        # 1. 繪製 Dose Heatmap
        plt.figure(figsize=(11, 9), dpi=120)
        im = plt.imshow(data, origin='lower', extent=[-WAFER_RADIUS, WAFER_RADIUS, -WAFER_RADIUS, WAFER_RADIUS], cmap='viridis', interpolation='bilinear')
        plt.colorbar(im, fraction=0.046, pad=0.04).set_label('Accumulated Dose (A.U.)')
        plt.gca().add_artist(plt.Circle((0, 0), WAFER_RADIUS, color='red', fill=False, linestyle='--', alpha=0.5))
        plt.title("Wafer Cleaning Dose Distribution (Transport Model)")
        
        # 統計數值計算 (只算 > 0)
        if data.size > 0 and np.any(data > 0):
            valid_data = data[data > 1e-6]
            h_max = np.max(data)
            h_min = np.min(valid_data) if valid_data.size > 0 else 0.0
            h_avg = np.mean(valid_data) if valid_data.size > 0 else 0.0
            h_med = np.median(valid_data) if valid_data.size > 0 else 0.0
        else: h_max = h_min = h_avg = h_med = 0.0
        
        if config is None:
            from simulation_config_def import get_default_config
            config = get_default_config()
            
        # 僅顯示 PRE 相關 Tuning Parameters
        tuning_keys = [
            'PRE_ALPHA', 'PRE_BETA', 'PRE_GRID_SIZE', 
            'PRE_GAMMA_BASE', 'PRE_DEFECT_CALI', 
            'PRE_TRANS_SENSITIVITY', 'PRE_REDEP_COEFF'
        ]
        from simulation_config_def import PARAMETER_DEFINITIONS
        params_lines = []
        for key in tuning_keys:
            if key in config:
                label = key
                for cat in PARAMETER_DEFINITIONS.values():
                    if key in cat:
                        label = cat[key][0]
                        break
                params_lines.append(f"{label}: {config[key]}")
        
        stats_text = (
            f"Max: {h_max:.4f}\nMin(>0): {h_min:.4f}\n"
            f"Average(>0): {h_avg:.4f}\nMedian(>0): {h_med:.4f}\n"
            "------------------\n" + "\n".join(params_lines)
        )
        plt.text(-145, -145, stats_text, color='white', fontsize=8, family='monospace', 
                 fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

        plt.tight_layout()
        plt.savefig(png_path, bbox_inches='tight', dpi=300)
        plt.close()

        # 2. 繪製 Defect Map (模擬真實檢測結果)
        plt.figure(figsize=(10, 10), dpi=150)
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_facecolor('#f8f9fa')
        
        initial_count = int(config.get('PRE_DEFECT_COUNT', 10000))
        remaining_count = len(final_defects)
        pre_percent = ((initial_count - remaining_count) / max(initial_count, 1)) * 100
        
        if remaining_count > 0:
            sizes = np.clip(final_defects[:, 2] * 0.3, 1, 50)
            plt.scatter(final_defects[:, 0], final_defects[:, 1], s=sizes, c='red', alpha=0.6, edgecolors='none')
        ax.add_artist(plt.Circle((0, 0), WAFER_RADIUS, color='#007bff', fill=False, lw=2, alpha=0.5))
        
        plt.title(f"Predicted Defect Map\nPRE: {pre_percent:.2f}% (Rem: {remaining_count} / {initial_count})")
        plt.savefig(map_path, dpi=200)
        plt.close()

        # 3. 輸出徑向分佈圖
        self._export_radial_distribution(matrix, radial_png_path)
        
        # 4. 輸出 CSV
        header = f"Cleaning Dose Data (Transport Model), Redep_Base: {config.get('PRE_REDEP_COEFF', 0.05)}"
        np.savetxt(csv_path, data, delimiter=",", fmt='%.6f', header=header)

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
        
        # 保存徑向分佈的 RawData
        csv_raw_path = filepath.replace(".png", "_Radial_RawData.csv")
        try:
            with open(csv_raw_path, 'w') as f:
                f.write("Radius(mm),Average Cleaning Dose(A.U.)\n")
                for r_val, avg_val in enumerate(radial_avg):
                    f.write(f"{r_val},{avg_val:.6f}\n")
        except Exception as e:
            print(f"Error saving radial raw data: {e}")

        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(np.arange(len(radial_avg)), radial_avg, color='blue', linewidth=2)
        plt.fill_between(np.arange(len(radial_avg)), radial_avg, alpha=0.2, color='blue')
        plt.title("Radial Cleaning Dose Distribution")
        plt.xlabel("Radius (mm)")
        plt.ylabel("Average Cleaning Dose (A.U.)")
        plt.xlim(0, max_r)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 統計看板 (Max, Min, Average, Median)
        if radial_avg.size > 0 and np.any(radial_avg > 0):
            valid_r = radial_avg[radial_avg > 1e-6]
            r_max = np.max(radial_avg)
            r_min = np.min(valid_r) if valid_r.size > 0 else 0.0
            r_avg = np.mean(valid_r) if valid_r.size > 0 else 0.0
            r_med = np.median(valid_r) if valid_r.size > 0 else 0.0
            
            stats_text = (
                f"Max: {r_max:.4f}\nMin(>0): {r_min:.4f}\n"
                f"Average: {r_avg:.4f}\nMedian: {r_med:.4f}"
            )
            plt.text(0.02, 0.05, stats_text, transform=plt.gca().transAxes, color='blue', 
                     fontsize=10, family='monospace', fontweight='bold', 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='blue'))

        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
