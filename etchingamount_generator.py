import numpy as np
import math
import os
import matplotlib.pyplot as plt
import tkinter as tk
from numba import njit, prange  # [修改] 引入 prange 支援平行運算

from models import DispenseArm
from simulation_engine import SimulationEngine
from constants import (
    ARM_GEOMETRIES, WAFER_RADIUS, REPORT_FPS, 
    GRID_SIZE, ETCHING_TAU,
    ETCHING_IMPINGEMENT_BONUS,
    ETCHING_GEO_SMOOTHING, ETCHING_SATURATION_THRESHOLD,
    ETCHING_SATURATION_THICKNESS, ETCHING_BASE_SPIN_DECAY
)

# --- [新增] Numba 核心 1: 粒子塗抹 (Deposition) ---
@njit(fastmath=True, cache=True)
def _numba_deposit_liquid(film_matrix, conc_matrix, center_x, center_y, 
                          radius, grid_size, dt, fresh_conc=1.0, 
                          impingement_bonus=1.2):
    """
    粒子將液體塗抹到網格上，並進行濃度混合 (CSTR Model)。
    """
    # 座標轉換：從 (-150, 150) 轉為 (0, 300)
    idx_x = center_x + 150.0
    idx_y = center_y + 150.0
    
    # 計算邊界
    r_pixel = int(math.ceil(radius))
    
    min_i = max(0, int(math.floor(idx_x - r_pixel)))
    max_i = min(grid_size - 1, int(math.ceil(idx_x + r_pixel)))
    min_j = max(0, int(math.floor(idx_y - r_pixel)))
    max_j = min(grid_size - 1, int(math.ceil(idx_y + r_pixel)))

    radius_sq = radius * radius
    
    # 定義單顆粒子在一個 dt 內攜帶的微量液體體積 (流量因子)
    # 數值大小決定了網格從乾變濕的速度
    particle_vol = impingement_bonus * dt 

    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            dist_sq = (i - idx_x)**2 + (j - idx_y)**2
            
            if dist_sq <= radius_sq:
                # 空間權重 (此處使用均勻權重，模擬液滴擴散)
                # 若希望更平滑可改為高斯權重
                weight = 1.0 
                
                added_vol = particle_vol * weight
                
                # [核心邏輯] 濃度混合 (Volume Weighted Average)
                # 新濃度 = (舊體積*舊濃度 + 新體積*新濃度) / 總體積
                old_h = film_matrix[i, j]
                old_c = conc_matrix[i, j]
                
                new_h = old_h + added_vol
                
                if new_h > 1e-6:
                    new_c = (old_h * old_c + added_vol * fresh_conc) / new_h
                    
                    # 更新網格狀態
                    film_matrix[i, j] = new_h
                    conc_matrix[i, j] = new_c

# --- [新增] Numba 核心 2: 網格演化 (Evolution) ---
@njit(fastmath=True, parallel=True, cache=True)
def _numba_evolve_grid(etch_matrix, film_matrix, conc_matrix, 
                       dt, base_spin_decay, chem_decay_tau, 
                       saturation_h, wafer_radius, 
                       geo_smoothing, sat_threshold,
                       current_rpm, shear_coeff,
                       global_scale):
    """
    全網格演化：新增相對速度 (剪切應力) 加成。
    """
    rows, cols = etch_matrix.shape
    center_idx = 150.0
    # 預計算化學衰減因子 (每幀衰減比例)
    chem_decay_factor = math.exp(-dt / chem_decay_tau)

    # 預計算角速度 omega (rad/s) = RPM * 2π / 60
    # 使用 abs() 確保反轉時速度加成依然為正
    omega = abs(current_rpm) * 0.10472

    for i in prange(rows):
        for j in range(cols):
            h = film_matrix[i, j]
            
            # 只有濕的地方才需要計算 (節省效能)
            if h > 0.0001:
                c = conc_matrix[i, j]
                
                # 計算該點離圓心的距離 r (mm)
                dx = i - center_idx
                dy = j - center_idx
                r = math.sqrt(dx*dx + dy*dy)

                # --- 1. 計算蝕刻反應 (Reaction, 含剪切加成) ---
                # 飽和機制：tanh(h / sat_h)
                # 當膜厚 h 超過 saturation_h，反應速率不再隨厚度增加 (Surface Reaction Limited)
                # 這能有效防止中心因為積水過厚而導致蝕刻量無限暴增
                saturation_factor = math.tanh(h / saturation_h)
                
                # 線速度 v = r * omega (mm/s)
                v_linear = r * omega
                # 剪切加成因子 (Shear Factor) = 1 + shear_coeff * v
                shear_factor = 1.0 + shear_coeff * v_linear

                # 本幀蝕刻量 = 濃度 * 飽和因子 * 剪切因子 * 時間
                delta_etch = c * saturation_factor * shear_factor * global_scale * dt
                
                # [新增] 飽和門檻處理 (np.tanh 限制極端值)
                if sat_threshold > 0:
                    delta_etch = math.tanh(delta_etch / sat_threshold) * sat_threshold
                
                etch_matrix[i, j] += delta_etch
                
                # --- 2. 物理甩乾 (Spin-off) ---
                if r <= wafer_radius + 5.0:
                    # 徑向甩乾模型：半徑越大，離心力越強，甩乾越快
                    # r_factor 模擬邊緣的高離心力加速乾燥
                    # 使用 geo_smoothing 調整徑向梯度 (預設 7.0 / 3.5 = 2.0)
                    r_factor = 1.0 + (geo_smoothing / 3.5) * (r / wafer_radius)
                    effective_decay = base_spin_decay * r_factor
                    
                    # 更新膜厚 (指數衰減)
                    film_matrix[i, j] *= (1.0 - effective_decay * dt)
                else:
                    # 晶圓外直接乾掉
                    film_matrix[i, j] = 0.0
                
                # --- 3. 化學老化 (Aging) ---
                # 濃度隨時間自然降低 (模擬反應消耗)
                conc_matrix[i, j] *= chem_decay_factor
                
            else:
                # 如果乾了，重置狀態，避免殘留微小數值干擾運算
                film_matrix[i, j] = 0.0
                conc_matrix[i, j] = 0.0

class EtchingAmountGenerator:
    def __init__(self, app_instance):
        self.app = app_instance

    def generate(self, recipe, filepath, config=None, progress_widgets=None, play_speed_multiplier=1.0):
        """
        核心蝕刻量模擬邏輯 (雙層網格狀態機版)
        """
        # 合併配置
        if config is None:
            from simulation_config_def import get_default_config
            config = get_default_config()

        # 提取參數
        etch_tau = config.get('ETCHING_TAU', ETCHING_TAU) # 用於化學老化
        grid_radius = config.get('GRID_SIZE', GRID_SIZE)
        sat_h = config.get('ETCHING_SATURATION_THICKNESS', ETCHING_SATURATION_THICKNESS)
        base_spin_decay = config.get('ETCHING_BASE_SPIN_DECAY', ETCHING_BASE_SPIN_DECAY)
        imp_bonus = config.get('ETCHING_IMPINGEMENT_BONUS', ETCHING_IMPINGEMENT_BONUS)
        geo_smoothing = config.get('ETCHING_GEO_SMOOTHING', ETCHING_GEO_SMOOTHING)
        sat_threshold = config.get('ETCHING_SATURATION_THRESHOLD', ETCHING_SATURATION_THRESHOLD)
        shear_coeff = config.get('ETCHING_SHEAR_COEFF', 0.0001)
        global_scale = config.get('ETCHING_GLOBAL_SCALE', 1.0)

        # 1. 初始化 Headless Arms
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
        water_params_dict = {i: {
            'viscosity': water_params['viscosity'],
            'surface_tension': water_params['surface_tension'],
            'evaporation_rate': water_params['evaporation_rate']
        } for i in [1, 2, 3]}

        # 2. 實例化引擎
        engine = SimulationEngine(recipe, headless_arms, water_params_dict, headless=True, config=config)
        
        # 3. 初始化狀態矩陣 (300x300)
        grid_size = 300
        # 最終結果 (累積蝕刻量)
        etch_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)
        # 液膜厚度矩陣 (記錄哪裡是濕的)
        film_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)
        # 濃度矩陣 (記錄藥液新鮮度)
        conc_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)

        # 影片同步設定 (解決動態 dt 問題)
        VIDEO_FPS = 30.0
        # 根據播放倍率調整錄製間隔。若 multiplier < 1.0，則間隔變小，產生的幀數變多，從而實現慢動作。
        record_interval = (1.0 / VIDEO_FPS) * play_speed_multiplier
        next_record_time = 0.0
        video_buffer = []
        
        report_fps = recipe.get('dynamic_report_fps', REPORT_FPS)
        dt = 1.0 / report_fps
        total_duration = sum(p['total_duration'] for p in recipe['processes'])
        sim_clock = 0.0

        # 新增：控制進度條顯示更新的頻率 (例如每 0.5 秒更新一次進度條上的文字和百分比)
        progress_display_interval = 0.5
        last_progress_display_time = 0.0 # 上次更新進度條顯示的時間

        # 新增：在循環開始前，為 JIT 編譯提供提示，並強制刷新 GUI
        if progress_widgets:
            progress_widgets['label'].config(text="Initializing JIT Engine for Etching (first run might be slow)...")
            # 確保 progress_widgets['bar'] 的最大值已經設定
            progress_widgets['bar']['maximum'] = total_duration
            progress_widgets['window'].update_idletasks() # 強制刷新 GUI

        # 4. 執行模擬主迴圈
        while True:
            snapshot = engine.update(dt) 
            sim_clock += dt

            # 影片緩存記錄
            if sim_clock >= next_record_time:
                video_buffer.append({
                    'etch': etch_matrix.copy(), 
                    'film': film_matrix.copy(), 
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
                        p_label.config(text=f"Etching (Film Model): {sim_clock:.1f}s / {total_duration:.1f}s ({percent:.0f}%)")
                        
                        # 強制刷新 GUI，讓進度條視窗有機會處理事件 and 繪製更新
                        progress_widgets['window'].update_idletasks()
                        
                        last_progress_display_time = sim_clock # 更新上次顯示時間
                    except tk.TclError as e: # 捕獲使用者關閉進度視窗時可能發生的錯誤
                        print(f"Etching progress window closed by user during GUI update: {e}, stopping generation.")
                        return False # 返回 False 表示生成被取消
                    except Exception as e:
                        print(f"Error updating Etching progress bar: {e}")
                        return False # 返回 False 表示生成失敗

            # --- A. 粒子塗抹 (Deposition) ---
            # 粒子作為 "水源"，將新鮮藥液塗在網格上
            on_wafer_mask = (engine.particles_state == 2) # P_ON_WAFER
            if np.any(on_wafer_mask):
                indices = np.where(on_wafer_mask)[0]
                
                # [修正] 預先獲取當前製程對象，避免在迴圈內多次查找
                current_proc_idx = snapshot['process_idx']
                current_proc = recipe['processes'][current_proc_idx]

                for i in indices:
                    # 取得相對座標
                    rel_x, rel_y = engine.particles_pos[i, 0], engine.particles_pos[i, 1]
                    
                    # [修正] 針對不同噴嘴流量進行加成補償
                    # 若為 Nozzle 3 (arm_id=3)，其流量來自 flow_rate_2
                    p_arm_id = engine.particles_arm_id[i]
                    if p_arm_id == 3:
                        actual_flow = current_proc.get('flow_rate_2', 500.0)
                    else:
                        actual_flow = current_proc.get('flow_rate', 500.0)
                    
                    # 根據流量調整沉積強度 (線性比例)
                    flow_bonus = imp_bonus * (actual_flow / 500.0)
                    
                    # 呼叫 Numba 核心進行塗抹
                    _numba_deposit_liquid(
                        film_matrix, conc_matrix, 
                        rel_x, rel_y, 
                        grid_radius, grid_size, dt, 
                        fresh_conc=1.0,
                        impingement_bonus=flow_bonus
                    )

            # --- B. 網格演化 (Evolution) ---
            # 計算全網格的反應、甩乾與老化
            
            # 讀取當下轉速 (snapshot 內含當前的動態 RPM)
            current_rpm = snapshot.get('rpm', 0)
            # 經驗公式：轉速越高，基礎甩乾率越高
            current_spin_decay = base_spin_decay * (1.0 + abs(current_rpm) / 500.0)

            # 呼叫更新後的 Numba 核心
            _numba_evolve_grid(
                etch_matrix, film_matrix, conc_matrix,
                dt, current_spin_decay, etch_tau,
                sat_h, WAFER_RADIUS,
                geo_smoothing, sat_threshold,
                current_rpm, shear_coeff,
                global_scale
            )

            if snapshot.get('is_finished') or sim_clock > (total_duration + 3.0):
                # 讓模擬多跑幾秒鐘，確保殘留在表面的液體完全反應/乾掉
                if np.max(film_matrix) < 0.001 and snapshot.get('is_finished'):
                     break
                elif sim_clock > (total_duration + 3.0):
                     break

        self._export_results(etch_matrix, filepath, config=config)
        
        # 影片輸出
        video_path = filepath.replace(".png", "_DualView.mp4")
        self._export_dual_view_video(video_buffer, video_path, 
                                     max_etch=np.max(etch_matrix) if np.max(etch_matrix) > 0 else 1.0, 
                                     sat_h=sat_h, fps=VIDEO_FPS)
        return True

    def _export_dual_view_video(self, video_buffer, output_path, max_etch, sat_h, fps=30.0):
        import cv2
        if not video_buffer: return

        view_size = 800 
        frame_size = (view_size * 2, view_size)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        mask = np.zeros((view_size, view_size), dtype=np.uint8)
        cv2.circle(mask, (view_size//2, view_size//2), view_size//2, 255, -1)

        print(f"Exporting video with Per-Frame Normalization for Liquid Film...")
        
        for frame_data in video_buffer:
            # --- 左圖：累積蝕刻 (維持最終最大值歸一化，觀察生長過程) ---
            etch_raw = frame_data['etch'].T
            etch_norm = (np.clip(etch_raw / max_etch, 0, 1) * 255).astype(np.uint8)
            etch_view = cv2.applyColorMap(etch_norm, cv2.COLORMAP_VIRIDIS)
            etch_view = cv2.resize(etch_view, (view_size, view_size), interpolation=cv2.INTER_NEAREST)
            etch_view = cv2.bitwise_and(etch_view, etch_view, mask=mask)

            # --- 右圖：動態液膜 (改為「逐幀最大值」歸一化) ---
            film_raw = frame_data['film'].T
            
            # [核心修改]：計算當前幀的最大值
            current_max_film = np.max(film_raw)
            # 設定視覺下限，避免全黑畫面時除以零
            norm_base = max(current_max_film, 0.01) 
            
            # 這樣每一幀都會重新拉伸亮度範圍
            film_norm = (np.clip(film_raw / norm_base, 0, 1) * 255).astype(np.uint8)
            film_view = cv2.applyColorMap(film_norm, cv2.COLORMAP_OCEAN)
            film_view = cv2.resize(film_view, (view_size, view_size), interpolation=cv2.INTER_NEAREST)
            film_view = cv2.bitwise_and(film_view, film_view, mask=mask)

            # 拼接並寫入
            out.write(np.hstack((etch_view, film_view)))

        out.release()

    def _export_results(self, matrix, filepath, config=None):
        base_path, _ = os.path.splitext(filepath)
        png_path = filepath
        real_base = base_path.replace("_Etching_Amount", "")
        csv_path = f"{real_base}_Etching_RawData.csv"
        radial_png_path = f"{real_base}_Etching_Radial_Distribution.png"
        
        data = matrix.T

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
        cbar.set_label('Simulated Etching Amount (A.U.)')

        wafer_circle = plt.Circle((0, 0), 150, color='red', fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_artist(wafer_circle)

        plt.title("Wafer Etching Amount (Film Model)", fontsize=14, pad=15)
        plt.xlabel("X Position (mm)")
        plt.ylabel("Y Position (mm)")

        # 統計數據
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
            np.savetxt(csv_path, data, delimiter=",", fmt='%.6f', 
                       header="Etching Amount Data (Film Model), Resolution: 1.0mm/pixel, Range: -150 to 150 mm")
        except Exception as e:
            print(f"Failed to write CSV: {e}")

        # 3. 輸出徑向分佈圖 (Radial Distribution)
        self._export_radial_distribution(matrix, radial_png_path)

    def run_fast_simulation(self, recipe, config):
        """專供 AutoTuner 使用的極速無頭模擬，只回傳最終的蝕刻矩陣與徑向分佈"""
        headless_arms = {}
        for i, geo in ARM_GEOMETRIES.items():
            if i == 2:
                headless_arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], None, None,
                                           side_arm_length=geo.get('side_arm_length'), 
                                           side_arm_angle_offset=geo.get('side_arm_angle_offset'),
                                           side_arm_branch_dist=geo.get('side_arm_branch_dist'))
            else:
                headless_arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], None, None)

        water_params = self.app._get_water_params() if hasattr(self, 'app') and hasattr(self.app, '_get_water_params') else {'viscosity': 1.0, 'surface_tension': 72.0, 'evaporation_rate': 0.1}
        water_params_dict = {i: water_params for i in [1, 2, 3]}

        engine = SimulationEngine(recipe, headless_arms, water_params_dict, headless=True, config=config)
        
        grid_size = 300
        etch_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)
        film_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)
        conc_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)

        report_fps = recipe.get('dynamic_report_fps', 30)
        dt = 1.0 / report_fps
        total_duration = sum(p['total_duration'] for p in recipe['processes'])
        sim_clock = 0.0

        # 讀取參數
        etch_tau = config.get('ETCHING_TAU', 1.5)
        base_spin_decay = config.get('ETCHING_BASE_SPIN_DECAY', 2.0)
        imp_bonus = config.get('ETCHING_IMPINGEMENT_BONUS', 1.2)
        geo_smoothing = config.get('ETCHING_GEO_SMOOTHING', 7.0)
        sat_threshold = config.get('ETCHING_SATURATION_THRESHOLD', 0.002)
        sat_h = config.get('ETCHING_SATURATION_THICKNESS', 0.5)
        shear_coeff = config.get('ETCHING_SHEAR_COEFF', 0.0001)
        global_scale = config.get('ETCHING_GLOBAL_SCALE', 1.0)
        grid_radius = config.get('GRID_SIZE', 1.5)

        while True:
            snapshot = engine.update(dt) 
            sim_clock += dt

            on_wafer_mask = engine.particles_state == 2 
            if np.any(on_wafer_mask):
                indices = np.where(on_wafer_mask)[0]
                for i in indices:
                    rel_x, rel_y = engine.particles_pos[i, 0], engine.particles_pos[i, 1]
                    _numba_deposit_liquid(film_matrix, conc_matrix, rel_x, rel_y, grid_radius, grid_size, dt, 1.0, imp_bonus)

            current_rpm = snapshot.get('rpm', 0)
            current_spin_decay = base_spin_decay * (1.0 + abs(current_rpm) / 500.0)

            _numba_evolve_grid(
                etch_matrix, film_matrix, conc_matrix,
                dt, current_spin_decay, etch_tau,
                sat_h, WAFER_RADIUS, geo_smoothing, sat_threshold,
                current_rpm, shear_coeff, global_scale
            )

            if snapshot.get('is_finished') or sim_clock > (total_duration + 3.0):
                 break

        # 計算徑向平均
        center = grid_size / 2.0
        y, x = np.indices(etch_matrix.shape)
        r = np.sqrt((x - center + 0.5)**2 + (y - center + 0.5)**2)
        r_rounded = r.astype(int)
        max_r = int(WAFER_RADIUS)
        radial_sum = np.zeros(max_r + 1)
        radial_count = np.zeros(max_r + 1)
        mask = r_rounded <= max_r
        np.add.at(radial_sum, r_rounded[mask], etch_matrix[mask])
        np.add.at(radial_count, r_rounded[mask], 1)
        radial_avg = np.divide(radial_sum, radial_count, out=np.zeros_like(radial_sum), where=radial_count > 0)

        return etch_matrix, radial_avg

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
        plt.plot(np.arange(len(radial_avg)), radial_avg, color='blue', linewidth=2, label='Average EA')
        plt.fill_between(np.arange(len(radial_avg)), radial_avg, alpha=0.2, color='blue')
        plt.title("Radial Etching Amount Distribution", fontsize=14, pad=15)
        plt.xlabel("Radius (mm)", fontsize=12)
        plt.ylabel("Average Etching Amount (A.U.)", fontsize=12)
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
