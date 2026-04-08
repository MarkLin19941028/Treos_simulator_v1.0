import numpy as np
import math
import os
import matplotlib.pyplot as plt
import tkinter as tk
from numba import njit, prange

from models import DispenseArm
from simulation_engine import SimulationEngine
from constants import (
    ARM_GEOMETRIES, WAFER_RADIUS, REPORT_FPS, 
    GRID_SIZE, ETCHING_TAU,
    ETCHING_IMPINGEMENT_BONUS,
    ETCHING_GEO_SMOOTHING, ETCHING_SATURATION_THRESHOLD,
    ETCHING_SATURATION_THICKNESS, ETCHING_BASE_SPIN_DECAY
)

@njit(fastmath=True, cache=True)
def _numba_deposit_liquid(film_matrix, conc_matrix, center_x, center_y, 
                          radius, grid_size, dt, fresh_conc=1.0, 
                          impingement_bonus=1.2):
    idx_x = center_x + 150.0
    idx_y = center_y + 150.0
    r_pixel = int(math.ceil(radius))
    min_i = max(0, int(math.floor(idx_x - r_pixel)))
    max_i = min(grid_size - 1, int(math.ceil(idx_x + r_pixel)))
    min_j = max(0, int(math.floor(idx_y - r_pixel)))
    max_j = min(grid_size - 1, int(math.ceil(idx_y + r_pixel)))
    
    radius_sq = radius * radius
    sigma = radius / 2.0
    total_weight = 0.0
    
    # Step 1: 計算總權重 (高斯歸一化)
    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            dist_sq = (i - idx_x)**2 + (j - idx_y)**2
            if dist_sq <= radius_sq:
                total_weight += math.exp(-dist_sq / (2 * sigma * sigma))
    
    if total_weight <= 0.0: return
    
    # Step 2: 根據高斯權重分配液體量
    particle_vol = impingement_bonus * dt 
    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            dist_sq = (i - idx_x)**2 + (j - idx_y)**2
            if dist_sq <= radius_sq:
                weight = math.exp(-dist_sq / (2 * sigma * sigma)) / total_weight
                added_vol = particle_vol * weight
                old_h = film_matrix[i, j]
                old_c = conc_matrix[i, j]
                new_h = old_h + added_vol
                if new_h > 1e-6:
                    new_c = (old_h * old_c + added_vol * fresh_conc) / new_h
                    film_matrix[i, j] = new_h
                    conc_matrix[i, j] = new_c

@njit(fastmath=True, parallel=True, cache=True)
def _numba_evolve_grid(etch_matrix, film_matrix, conc_matrix, 
                       dt, base_spin_decay, chem_decay_tau, 
                       saturation_h, wafer_radius, 
                       geo_smoothing, sat_threshold,
                       current_rpm, shear_coeff,
                       global_scale):
    rows, cols = etch_matrix.shape
    center_idx = 150.0
    chem_decay_factor = math.exp(-dt / chem_decay_tau)
    omega = abs(current_rpm) * 0.10472
    for i in prange(rows):
        for j in range(cols):
            h = film_matrix[i, j]
            if h > 0.0001:
                c = conc_matrix[i, j]
                dx = i - center_idx
                dy = j - center_idx
                r = math.sqrt(dx*dx + dy*dy)
                saturation_factor = math.tanh(h / saturation_h)
                v_linear = r * omega
                shear_factor = 1.0 + shear_coeff * v_linear
                delta_etch = c * saturation_factor * shear_factor * global_scale * dt
                if sat_threshold > 0:
                    delta_etch = math.tanh(delta_etch / sat_threshold) * sat_threshold
                etch_matrix[i, j] += delta_etch
                if r <= wafer_radius + 5.0:
                    r_factor = 1.0 + (geo_smoothing / 3.5) * (r / wafer_radius)
                    effective_decay = base_spin_decay * r_factor
                    film_matrix[i, j] *= (1.0 - effective_decay * dt)
                else:
                    film_matrix[i, j] = 0.0
                conc_matrix[i, j] *= chem_decay_factor
            else:
                film_matrix[i, j] = 0.0
                conc_matrix[i, j] = 0.0

class EtchingAmountGenerator:
    def __init__(self, app_instance):
        self.app = app_instance

    def generate(self, recipe, filepath, config=None, progress_widgets=None, play_speed_multiplier=1.0):
        if config is None:
            from simulation_config_def import get_default_config
            config = get_default_config()
        etch_tau = config.get('ETCHING_TAU', ETCHING_TAU)
        grid_radius = config.get('GRID_SIZE', GRID_SIZE)
        sat_h = config.get('ETCHING_SATURATION_THICKNESS', ETCHING_SATURATION_THICKNESS)
        base_spin_decay = config.get('ETCHING_BASE_SPIN_DECAY', ETCHING_BASE_SPIN_DECAY)
        imp_bonus = config.get('ETCHING_IMPINGEMENT_BONUS', ETCHING_IMPINGEMENT_BONUS)
        geo_smoothing = config.get('ETCHING_GEO_SMOOTHING', ETCHING_GEO_SMOOTHING)
        sat_threshold = config.get('ETCHING_SATURATION_THRESHOLD', ETCHING_SATURATION_THRESHOLD)
        shear_coeff = config.get('ETCHING_SHEAR_COEFF', 0.0001)
        global_scale = config.get('ETCHING_GLOBAL_SCALE', 1.0)
        # [對齊 AutoTuner 的加速邏輯] 
        max_rpm = 0
        for proc in recipe['processes']:
            spin = proc.get('spin_params', {})
            c_max = spin.get('rpm', 0) if spin.get('mode', 'Simple') == 'Simple' else max(spin.get('start_rpm', 0), spin.get('end_rpm', 0))
            if float(c_max) > max_rpm: max_rpm = float(c_max)
        
        report_fps = max(200, min(2000, int(max_rpm * 1.0)))
        recipe['dynamic_report_fps'] = report_fps
        dt = 1.0 / report_fps

        headless_arms = {}
        for i, geo in ARM_GEOMETRIES.items():
            if i == 2:
                headless_arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], None, None,
                                           side_arm_length=geo.get('side_arm_length'), 
                                           side_arm_angle_offset=geo.get('side_arm_angle_offset'),
                                           side_arm_branch_dist=geo.get('side_arm_branch_dist'))
            else:
                headless_arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], None, None)
        water_params = self.app._get_water_params()
        water_params_dict = {i: water_params for i in [1, 2, 3]}

        # 啟用 fast_mode，並設定粒子縮放比例
        max_flow = max([proc.get('flow_rate', 500.0) for proc in recipe['processes']])
        from constants import PARTICLE_SPAWN_MULTIPLIER
        original_rate = max_flow * 0.5 * PARTICLE_SPAWN_MULTIPLIER
        target_rate = max(200.0, original_rate * 0.5)
        fast_particle_scale = min(1.0, target_rate / max(original_rate, 1.0))

        engine = SimulationEngine(recipe, headless_arms, water_params_dict, headless=True, config=config, fast_mode=True, fast_particle_scale=fast_particle_scale)
        
        grid_size = 300
        etch_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)
        film_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)
        conc_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)
        VIDEO_FPS = 30.0
        record_interval = (1.0 / VIDEO_FPS) * play_speed_multiplier
        next_record_time = 0.0
        video_buffer = []
        total_duration = sum(p['total_duration'] for p in recipe['processes'])
        sim_clock = 0.0
        progress_display_interval = 0.5
        last_progress_display_time = 0.0
        if progress_widgets:
            progress_widgets['label'].config(text="Initializing JIT Engine...")
            progress_widgets['bar']['maximum'] = total_duration
            progress_widgets['window'].update_idletasks()
        while True:
            snapshot = engine.update(dt) 
            sim_clock += dt
            if sim_clock >= next_record_time:
                video_buffer.append({'etch': etch_matrix.copy(), 'film': film_matrix.copy(), 'time': sim_clock})
                next_record_time += record_interval
            if (sim_clock - last_progress_display_time >= progress_display_interval) or snapshot.get('is_finished'):
                if progress_widgets:
                    try:
                        p_bar = progress_widgets['bar']
                        p_label = progress_widgets['label']
                        p_bar['value'] = min(sim_clock, total_duration)
                        percent = (min(sim_clock, total_duration) / total_duration) * 100
                        p_label.config(text=f"Etching: {sim_clock:.1f}s / {total_duration:.1f}s ({percent:.0f}%)")
                        progress_widgets['window'].update_idletasks()
                        last_progress_display_time = sim_clock
                    except: return False
            on_wafer_mask = (engine.particles_state == 2)
            if np.any(on_wafer_mask):
                indices = np.where(on_wafer_mask)[0]
                current_proc_idx = snapshot['process_idx']
                current_proc = recipe['processes'][current_proc_idx]
                for i in indices:
                    rel_x, rel_y = engine.particles_pos[i, 0], engine.particles_pos[i, 1]
                    p_arm_id = engine.particles_arm_id[i]
                    actual_flow = current_proc.get('flow_rate_2' if p_arm_id == 3 else 'flow_rate', 500.0)
                    flow_bonus = imp_bonus * (actual_flow / 500.0)
                    
                    # 補償減少的粒子數，確保總沉積液體量不變
                    flow_bonus *= (1.0 / fast_particle_scale)
                    
                    _numba_deposit_liquid(film_matrix, conc_matrix, rel_x, rel_y, grid_radius, grid_size, dt, 1.0, flow_bonus)
            current_rpm = snapshot.get('rpm', 0)
            current_spin_decay = base_spin_decay * (1.0 + abs(current_rpm) / 500.0)
            _numba_evolve_grid(etch_matrix, film_matrix, conc_matrix, dt, current_spin_decay, etch_tau, sat_h, WAFER_RADIUS, geo_smoothing, sat_threshold, current_rpm, shear_coeff, global_scale)
            if snapshot.get('is_finished') or sim_clock > (total_duration + 3.0):
                if np.max(film_matrix) < 0.001 and snapshot.get('is_finished'): break
                elif sim_clock > (total_duration + 3.0): break
        self._export_results(etch_matrix, filepath, config=config)
        video_path = filepath.replace(".png", "_EtchingView.mp4")
        self._export_etching_video(video_buffer, video_path, max_etch=np.max(etch_matrix) if np.max(etch_matrix) > 0 else 1.0, fps=VIDEO_FPS)
        return True

    def _export_etching_video(self, video_buffer, output_path, max_etch, fps=30.0):
        import cv2
        if not video_buffer: return
        view_size = 400 
        frame_size = (view_size, view_size)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        mask = np.zeros((view_size, view_size), dtype=np.uint8)
        cv2.circle(mask, (view_size//2, view_size//2), view_size//2, 255, -1)
        for frame_data in video_buffer:
            etch_raw = frame_data['etch'].T
            etch_norm = (np.clip(etch_raw / max_etch, 0, 1) * 255).astype(np.uint8)
            etch_view = cv2.applyColorMap(etch_norm, cv2.COLORMAP_VIRIDIS)
            etch_view = cv2.resize(etch_view, (view_size, view_size), interpolation=cv2.INTER_NEAREST)
            etch_view = cv2.bitwise_and(etch_view, etch_view, mask=mask)
            out.write(etch_view)
        out.release()

    def _export_results(self, matrix, filepath, config=None):
        base_path, _ = os.path.splitext(filepath)
        real_base = base_path.replace("_Etching_Amount", "")
        csv_path = f"{real_base}_Etching_RawData.csv"
        radial_png_path = f"{real_base}_Etching_Radial_Distribution.png"
        data = matrix.T
        plt.figure(figsize=(11, 9), dpi=120)
        im = plt.imshow(data, origin='lower', extent=[-150, 150, -150, 150], cmap='viridis', interpolation='bilinear')
        plt.colorbar(im, fraction=0.046, pad=0.04).set_label('Simulated Etching Amount (A.U.)')
        plt.gca().add_artist(plt.Circle((0, 0), 150, color='red', fill=False, linestyle='--', alpha=0.5))
        plt.title("Wafer Etching Amount (Film Model)", fontsize=14, pad=15)
        plt.xlabel("X Position (mm)"); plt.ylabel("Y Position (mm)")
        
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
            
        # 僅顯示 Etching 相關 Tuning Parameters
        tuning_keys = [
            'ETCHING_GLOBAL_SCALE', 'GRID_SIZE', 'ETCHING_TAU', 
            'ETCHING_SATURATION_THICKNESS', 'ETCHING_BASE_SPIN_DECAY', 
            'ETCHING_IMPINGEMENT_BONUS', 'ETCHING_GEO_SMOOTHING', 
            'ETCHING_SATURATION_THRESHOLD', 'ETCHING_SHEAR_COEFF'
        ]
        from simulation_config_def import PARAMETER_DEFINITIONS
        params_lines = []
        for key in tuning_keys:
            if key in config:
                # 從定義中找 Label
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
        
        plt.tight_layout(); plt.savefig(filepath, bbox_inches='tight', dpi=300); plt.close()
        try: np.savetxt(csv_path, data, delimiter=",", fmt='%.6f', header="Etching Amount Data")
        except: pass
        self._export_radial_distribution(matrix, radial_png_path)

    def calculate_radial_average(self, matrix, precomputed_indices=None):
        """
        計算矩陣的徑向平均值。
        矩陣大小預期為 300x300，代表 -150mm 到 150mm。
        中心點座標為 150.0，與模擬核心對齊。
        """
        max_r = int(WAFER_RADIUS) # 150
        radial_sum = np.zeros(max_r + 1)
        radial_count = np.zeros(max_r + 1)

        if precomputed_indices is not None:
            r_rounded_mask, mask = precomputed_indices
            np.add.at(radial_sum, r_rounded_mask, matrix[mask])
        else:
            grid_size = matrix.shape[0]
            center = grid_size / 2.0
            y, x = np.indices(matrix.shape)
            # 移除 0.5 偏移，使其與物理核心 _numba_evolve_grid 的 dx = i - 150.0 對齊
            r = np.sqrt((x - center)**2 + (y - center)**2)
            r_rounded = r.astype(int)
            mask = r_rounded <= max_r
            r_rounded_mask = r_rounded[mask]
            np.add.at(radial_sum, r_rounded_mask, matrix[mask])

        # 計算 count (這裡為了極致效能，如果頻繁調用，建議在外部算好一次)
        # 我們在 AutoTuner 中會處理 count
        if precomputed_indices is not None:
             # 如果是預計算模式，count 應該已經被外部處理或在此重複計算(bincount)
             # 為了簡單，我們先用 bincount 算 count
             radial_count = np.bincount(r_rounded_mask, minlength=max_r+1)
        else:
             np.add.at(radial_count, r_rounded_mask, 1)
        
        radial_avg = np.divide(radial_sum, radial_count, out=np.zeros_like(radial_sum), where=radial_count > 0)
        return radial_avg

    def get_radial_indices_mask(self, shape):
        grid_size = shape[0]
        center = grid_size / 2.0
        y, x = np.indices(shape)
        # 移除 0.5 偏移，使其與物理核心對齊
        r = np.sqrt((x - center)**2 + (y - center)**2)
        r_rounded = r.astype(int)
        mask = r_rounded <= int(WAFER_RADIUS)
        return r_rounded[mask], mask

    def run_fast_simulation(self, recipe, config):
        """
        快速物理模擬執行，不產生影片與中間結果。
        """
        # 根據轉速動態計算 FPS，確保與正式輸出一致
        max_rpm = 0
        for proc in recipe['processes']:
            spin = proc['spin_params']
            c_max = spin['rpm'] if spin['mode'] == 'Simple' else max(spin['start_rpm'], spin['end_rpm'])
            if c_max > max_rpm: max_rpm = c_max
        recipe['dynamic_report_fps'] = max(200, min(2000, int(max_rpm * 1.0)))

        headless_arms = {}
        for i, geo in ARM_GEOMETRIES.items():
            if i == 2:
                headless_arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], None, None, side_arm_length=geo.get('side_arm_length'), side_arm_angle_offset=geo.get('side_arm_angle_offset'), side_arm_branch_dist=geo.get('side_arm_branch_dist'))
            else: headless_arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], None, None)
        water_params = self.app._get_water_params() if hasattr(self, 'app') and hasattr(self.app, '_get_water_params') else {'viscosity': 1.0, 'surface_tension': 72.8, 'evaporation_rate': 0.0}
        water_params_dict = {i: water_params for i in [1, 2, 3]}

        # [優化] 啟用 fast_mode，並設定粒子縮放比例，確保生成率下限
        max_flow = max([proc.get('flow_rate', 500.0) for proc in recipe['processes']])
        from constants import PARTICLE_SPAWN_MULTIPLIER
        original_rate = max_flow * 0.5 * PARTICLE_SPAWN_MULTIPLIER
        target_rate = max(200.0, original_rate * 0.5)
        fast_particle_scale = min(1.0, target_rate / max(original_rate, 1.0))
        engine = SimulationEngine(recipe, headless_arms, water_params_dict, headless=True, config=config, fast_mode=True, fast_particle_scale=fast_particle_scale)
        grid_size = 300
        etch_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)
        film_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)
        conc_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)
        dt = 1.0 / recipe['dynamic_report_fps']
        total_duration = sum(p['total_duration'] for p in recipe['processes'])
        sim_clock = 0.0
        etch_tau = config.get('ETCHING_TAU', ETCHING_TAU)
        grid_radius = config.get('GRID_SIZE', GRID_SIZE)
        sat_h = config.get('ETCHING_SATURATION_THICKNESS', ETCHING_SATURATION_THICKNESS)
        base_spin_decay = config.get('ETCHING_BASE_SPIN_DECAY', ETCHING_BASE_SPIN_DECAY)
        imp_bonus = config.get('ETCHING_IMPINGEMENT_BONUS', ETCHING_IMPINGEMENT_BONUS)
        geo_smoothing = config.get('ETCHING_GEO_SMOOTHING', ETCHING_GEO_SMOOTHING)
        sat_threshold = config.get('ETCHING_SATURATION_THRESHOLD', ETCHING_SATURATION_THRESHOLD)
        shear_coeff = config.get('ETCHING_SHEAR_COEFF', 0.0001)
        global_scale = config.get('ETCHING_GLOBAL_SCALE', 1.0)
        while True:
            snapshot = engine.update(dt) 
            sim_clock += dt
            on_wafer_mask = engine.particles_state == 2 
            if np.any(on_wafer_mask):
                indices = np.where(on_wafer_mask)[0]
                current_proc = recipe['processes'][snapshot['process_idx']]
                for i in indices:
                    rel_x, rel_y = engine.particles_pos[i, 0], engine.particles_pos[i, 1]
                    p_arm_id = engine.particles_arm_id[i]
                    actual_flow = current_proc.get('flow_rate_2' if p_arm_id == 3 else 'flow_rate', 500.0)
                    flow_bonus = imp_bonus * (actual_flow / 500.0)
                    
                    # [優化] 補償減少的粒子數，確保總沉積液體量不變
                    flow_bonus *= (1.0 / fast_particle_scale)
                    
                    _numba_deposit_liquid(film_matrix, conc_matrix, rel_x, rel_y, grid_radius, grid_size, dt, 1.0, flow_bonus)
            current_rpm = snapshot.get('rpm', 0)
            current_spin_decay = base_spin_decay * (1.0 + abs(current_rpm) / 500.0)
            _numba_evolve_grid(etch_matrix, film_matrix, conc_matrix, dt, current_spin_decay, etch_tau, sat_h, WAFER_RADIUS, geo_smoothing, sat_threshold, current_rpm, shear_coeff, global_scale)
            if snapshot.get('is_finished') or sim_clock > (total_duration + 3.0):
                if np.max(film_matrix) < 0.001 and snapshot.get('is_finished'): break
                elif sim_clock > (total_duration + 3.0): break
        return etch_matrix, self.calculate_radial_average(etch_matrix)

    def _export_radial_distribution(self, matrix, filepath):
        radial_avg = self.calculate_radial_average(matrix)
        
        # 保存徑向分佈的 RawData
        csv_raw_path = filepath.replace(".png", "_Radial_RawData.csv")
        try:
            with open(csv_raw_path, 'w') as f:
                f.write("Radius(mm),Average Etching Amount(A.U.)\n")
                for r_val, avg_val in enumerate(radial_avg):
                    f.write(f"{r_val},{avg_val:.6f}\n")
        except Exception as e:
            print(f"Error saving radial raw data: {e}")

        max_r = int(WAFER_RADIUS)
        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(np.arange(len(radial_avg)), radial_avg, color='blue', linewidth=2, label='Average EA')
        plt.fill_between(np.arange(len(radial_avg)), radial_avg, alpha=0.2, color='blue')
        plt.title("Radial Etching Amount Distribution", fontsize=14, pad=15)
        plt.xlabel("Radius (mm)", fontsize=12); plt.ylabel("Average Etching Amount (A.U.)", fontsize=12)
        plt.xlim(0, max_r); plt.xticks(np.arange(0, max_r + 1, 10))
        plt.ylim(0, np.max(radial_avg) * 1.1 if np.max(radial_avg) > 0 else 1.0)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 統計看板 (與要求對齊：Max, Min, Average, Median)
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
                     
        plt.tight_layout(); plt.savefig(filepath, bbox_inches='tight', dpi=300); plt.close()
