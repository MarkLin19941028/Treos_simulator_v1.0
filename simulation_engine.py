import numpy as np
import math
import random
from numba import njit, prange
from constants import *

# Particle States
P_INACTIVE = 0
P_FALLING = 1
P_ON_WAFER = 2

@njit(fastmath=True, cache=True)
def _physics_kernel(states, pos, vel, last_pos, life, time_on_wafer, path_length, arm_ids,
                    dt, omega, cos_wafer, sin_wafer,
                    viscosities, evap_rates, surface_tensions,
                    gravity, wafer_radius,
                    p_push_strength, p_core_radius,
                    st_resist_base, weber_coeff,
                    visc_damping, film_thinning_factor, drying_visc_mult,
                    rpm_evap_coeff):
    """
    [Core Physics Update - Final Version]
    1. 修正 A (能量保留): 將垂直撞擊速度轉化為水平噴濺速度 (SPLAT_RATIO)。
    2. 修正 B (運動解鎖): 移除 max(0) 的硬性門檻，改用 MIN_DRIFT_ACC 確保粒子不卡死。
    """
    n = states.shape[0]
    
    # 物理常數設定
    SPLAT_RATIO = 0.0      # 垂直速度轉換為水平初速的比例 (15%)
    MIN_DRIFT_ACC = 0.0     # 即使在中心或低離心力區，液體也會緩慢擴散的最小加速度
    
    for i in range(n):
        state = states[i]
        if state == P_INACTIVE:
            continue
            
        arm_id = arm_ids[i]
        
        # 1. 蒸發邏輯 (考慮轉速影響)
        rpm_factor = 1.0 + rpm_evap_coeff * abs(omega)
        eff_evap = evap_rates[arm_id] * rpm_factor
        
        if eff_evap > 0:
            life[i] -= eff_evap * dt
            if life[i] <= 0:
                states[i] = P_INACTIVE
                continue

        # 紀錄上一位置 (用於渲染插值)
        last_pos[i, 0] = pos[i, 0]
        last_pos[i, 1] = pos[i, 1]

        if state == P_FALLING:
            # 落下階段 (絕對座標下的自由落體)
            vel[i, 2] -= gravity * dt
            pos[i, 0] += vel[i, 0] * dt
            pos[i, 1] += vel[i, 1] * dt
            pos[i, 2] += vel[i, 2] * dt
            
            # 碰撞檢查 (碰觸晶圓表面)
            if pos[i, 2] <= 0:
                dist_sq = pos[i, 0]**2 + pos[i, 1]**2
                if dist_sq <= wafer_radius**2:
                    # 1. 座標轉換：Absolute -> Relative
                    abs_x, abs_y = pos[i, 0], pos[i, 1]
                    rel_x = abs_x * cos_wafer + abs_y * sin_wafer
                    rel_y = -abs_x * sin_wafer + abs_y * cos_wafer
                    
                    # 2. 計算相對速度 (初速繼承與旋轉補償)
                    # 粒子落下的絕對水平速度
                    v_abs_x, v_abs_y = vel[i, 0], vel[i, 1]
                    # 晶圓在碰撞點的絕對水平速度 (v = omega x r)
                    v_wafer_abs_x = -omega * abs_y
                    v_wafer_abs_y = omega * abs_x
                    
                    # 相對於晶圓的絕對速度向量
                    v_rel_abs_x = v_abs_x - v_wafer_abs_x
                    v_rel_abs_y = v_abs_y - v_wafer_abs_y
                    
                    # 轉換為相對座標系下的速度
                    v_rel_x = v_rel_abs_x * cos_wafer + v_rel_abs_y * sin_wafer
                    v_rel_y = -v_rel_abs_x * sin_wafer + v_rel_abs_y * cos_wafer
                    
                    # 3. 加上垂直撞擊產生的徑向噴濺 (SPLAT)
                    vz_impact = abs(vel[i, 2])
                    dist_val = math.sqrt(rel_x**2 + rel_y**2) + 0.1
                    nx, ny = rel_x / dist_val, rel_y / dist_val
                    
                    # 最終相對速度 = 繼承的相對速度 + 噴濺分量
                    vel[i, 0] = v_rel_x + nx * vz_impact * SPLAT_RATIO
                    vel[i, 1] = v_rel_y + ny * vz_impact * SPLAT_RATIO
                    vel[i, 2] = 0.0
                    
                    # 更新狀態與位置
                    pos[i, 0] = rel_x
                    pos[i, 1] = rel_y
                    pos[i, 2] = 0.0
                    states[i] = P_ON_WAFER
                    
                    # 更新 last_pos 以維持座標系統的一致性
                    last_pos[i, 0] = pos[i, 0]
                    last_pos[i, 1] = pos[i, 1]
                else:
                    states[i] = P_INACTIVE

        elif state == P_ON_WAFER:
            time_on_wafer[i] += dt
            
            # 獲取相對座標與相對速度
            x, y = pos[i, 0], pos[i, 1]
            vx, vy = vel[i, 0], vel[i, 1]
            dist_sq = x*x + y*y
            dist = math.sqrt(dist_sq)
            
            if dist > 1e-4:
                inv_dist = 1.0 / dist
                
                # A. 基礎離心力
                acc_cent = omega * omega * dist
                
                # B. 表面張力阻尼 (動態韋伯數修正)
                speed_sq = vx*vx + vy*vy
                weber_factor = 1.0 / (1.0 + weber_coeff * speed_sq)
                st_resist = (surface_tensions[arm_id] * st_resist_base) * weber_factor
                
                # --- [修正 B] 軟性門檻：確保即使在低速區液體也會緩慢流動 (Drift) ---
                net_outward_acc = acc_cent - st_resist
                if net_outward_acc < MIN_DRIFT_ACC:
                    net_outward_acc = MIN_DRIFT_ACC
                
                # C. 中心壓力梯度推力 (解決 Center Humping)
                if dist < p_core_radius:
                    pressure_push = p_push_strength * (1.0 - dist / p_core_radius)
                    net_outward_acc += pressure_push

                # 分解徑向加速度分量
                ax_radial = (x * inv_dist) * net_outward_acc
                ay_radial = (y * inv_dist) * net_outward_acc
                
                # D. 科氏力 (-2 * omega x v)
                # 加上抑制參數避免力道過強
                # 經驗上，在這種簡易流體模型中，0.2~0.5 的係數比較符合視覺預期
                CORIOLIS_FACTOR = 0.3 
                ax_cor =  -2 * omega * vy * CORIOLIS_FACTOR
                ay_cor = 2 * omega * vx * CORIOLIS_FACTOR
                
                # E. 總加速度更新速度
                vel[i, 0] += (ax_radial + ax_cor) * dt
                vel[i, 1] += (ay_radial + ay_cor) * dt
                
                # F. 黏滯阻尼 (Viscous Damping)
                # 結合乾燥效應與膜厚變薄因子
                dryness = 1.0 - max(0.0, life[i])
                visc_dry_factor = 1.0 + drying_visc_mult * (dryness * dryness) 
                eff_film_thinning = 1.0 + film_thinning_factor * (dist / wafer_radius)
                
                effective_visc = viscosities[arm_id] * visc_dry_factor * eff_film_thinning
                
                # [修正] 採用指數衰減，確保大 dt 下的數值穩定性，不會產生過度減速或震盪
                damping_exp = visc_damping * effective_visc * dt
                damping = math.exp(-damping_exp)
                
                vel[i, 0] *= damping
                vel[i, 1] *= damping
                
            # 更新相對位置
            pos[i, 0] += vel[i, 0] * dt
            pos[i, 1] += vel[i, 1] * dt
            
            # 更新物理路徑
            move_dist = math.sqrt((vel[i, 0] * dt)**2 + (vel[i, 1] * dt)**2)
            path_length[i] += move_dist

        # 邊界移除
        if (pos[i, 0]**2 + pos[i, 1]**2) > (wafer_radius + 20.0)**2:
            states[i] = P_INACTIVE


class SimulationEngine:
    def __init__(self, recipe, arms_dict, water_params_dict, headless=False, config=None, fast_mode=False, fast_particle_scale=0.5):
        self.recipe = recipe
        self.arms = arms_dict
        self.water_params = water_params_dict
        self.headless = headless
        self.config = config if config else {}
        self.fast_mode = fast_mode
        self.fast_particle_scale = fast_particle_scale

        self.simulation_mode = self.config.get('SIMULATION_MODE', 'full')
        self.max_nozzle_speed_mms = self.config.get('MAX_NOZZLE_SPEED_MMS', 250.0)
        self.transition_arm_speed_ratio = self.config.get('TRANSITION_ARM_SPEED_RATIO', TRANSITION_ARM_SPEED_RATIO)
        self.arm_change_pause_time = self.config.get('ARM_CHANGE_PAUSE_TIME', ARM_CHANGE_PAUSE_TIME)
        self.center_pause_time = self.config.get('CENTER_PAUSE_TIME', CENTER_PAUSE_TIME)
        
        # Physics Parameters from config
        self.p_push_strength = self.config.get('PHYSICS_PRESSURE_PUSH_STRENGTH', PHYSICS_PRESSURE_PUSH_STRENGTH)
        self.p_core_radius = self.config.get('PHYSICS_PRESSURE_CORE_RADIUS', PHYSICS_PRESSURE_CORE_RADIUS)
        self.st_resist_base = self.config.get('PHYSICS_ST_RESIST_BASE', PHYSICS_ST_RESIST_BASE)
        self.weber_coeff = self.config.get('PHYSICS_WEBER_COEFF', PHYSICS_WEBER_COEFF)
        self.visc_damping = self.config.get('PHYSICS_VISCOSITY_DAMPING', PHYSICS_VISCOSITY_DAMPING)
        self.film_thinning_factor = self.config.get('PHYSICS_FILM_THINNING_FACTOR', PHYSICS_FILM_THINNING_FACTOR)
        self.drying_visc_mult = self.config.get('PHYSICS_DRYING_VISC_MULT', PHYSICS_DRYING_VISC_MULT)
        self.rpm_evap_coeff = self.config.get('PHYSICS_RPM_EVAP_COEFF', PHYSICS_RPM_EVAP_COEFF)
        self.spray_spread_base = self.config.get('PHYSICS_SPRAY_SPREAD_BASE', PHYSICS_SPRAY_SPREAD_BASE)
        self.jet_speed_factor = self.config.get('PHYSICS_JET_SPEED_FACTOR', PHYSICS_JET_SPEED_FACTOR)

        self.max_particles = PARTICLE_MAX_COUNT
        self.particles_state = np.zeros(self.max_particles, dtype=np.int32)
        self.particles_pos = np.zeros((self.max_particles, 3), dtype=np.float64)
        self.particles_vel = np.zeros((self.max_particles, 3), dtype=np.float64)
        self.particles_last_pos = np.zeros((self.max_particles, 2), dtype=np.float64)
        self.particles_life = np.zeros(self.max_particles, dtype=np.float64)
        self.particles_time_on_wafer = np.zeros(self.max_particles, dtype=np.float64)
        self.particles_path_length = np.zeros(self.max_particles, dtype=np.float64)
        self.particles_birth_time = np.zeros(self.max_particles, dtype=np.float64)
        self.particles_arm_id = np.zeros(self.max_particles, dtype=np.int32)
        self.particles_id = np.zeros(self.max_particles, dtype=np.int32)
        
        self.next_particle_id = 0
        self._spawn_accumulator = {arm_id: 0.0 for arm_id in [1, 2, 3]} # 預留 3 個來源
        
        self.viscosities = np.ones(10, dtype=np.float64)
        self.evap_rates = np.zeros(10, dtype=np.float64)
        self.surface_tensions = np.full(10, 72.8, dtype=np.float64)
        
        for arm_id, p in water_params_dict.items():
            if arm_id < 10:
                self.viscosities[arm_id] = p.get('viscosity', 1.0)
                self.evap_rates[arm_id] = p.get('evaporation_rate', 0.0)
                self.surface_tensions[arm_id] = p.get('surface_tension', 72.8)

        self._pre_calculate_physics()
        
        self.current_notch_coords = np.array([[WAFER_RADIUS, 0],
                                              [WAFER_RADIUS-NOTCH_DEPTH, NOTCH_HALF_WIDTH],
                                              [WAFER_RADIUS-NOTCH_DEPTH, -NOTCH_HALF_WIDTH]])
        self.reset()

    @property
    def particle_systems(self):
        systems = {arm_id: [] for arm_id in [1, 2, 3]}
        for i in range(self.max_particles):
            state_val = self.particles_state[i]
            if state_val == P_INACTIVE:
                continue
            
            arm_id = self.particles_arm_id[i]
            p_dict = {
                'id': self.particles_id[i],
                'state': 'falling' if state_val == P_FALLING else 'on_wafer',
                'life': self.particles_life[i],
                'birth_time': self.particles_birth_time[i],
                'time_on_wafer': self.particles_time_on_wafer[i],
                'path_length': self.particles_path_length[i],
                'pos': self.particles_pos[i].copy(),
                'last_pos': self.particles_last_pos[i].copy()
            }
            if arm_id in systems:
                systems[arm_id].append(p_dict)
        return systems

    def reset(self):
        self.simulation_time_elapsed = 0.0
        self.current_process_index = 0
        self.time_offset_for_current_process = 0.0
        self.cumulative_physics_time = 0.0
        self.wafer_angle = 0.0
        self.is_looping_back = False
        self.current_step_label = "Init"
        self.current_rpm_value = 0.0
        
        self.particles_state.fill(P_INACTIVE)
        self.next_particle_id = 0

        first_proc = self.recipe['processes'][0]
        self.active_arm_id = first_proc['arm_id']
        
        if not self.active_arm_id or self.active_arm_id == 0:
            self.animation_state = STATE_RUNNING_PROCESS
            self.last_nozzle_pos = np.array([0.0, 0.0])
        else:
            arm = self.arms[self.active_arm_id]
            self.animation_state = STATE_ARM_MOVE_FROM_HOME
            # 支援 list (Arm 2) 或是單一 array
            home_coords = arm.angle_to_coords(arm.home_angle)
            self.last_nozzle_pos = [c.copy() for c in home_coords] if isinstance(home_coords, list) else home_coords.copy()
            self.transition_start_time = 0.0
            self.transition_start_angle = arm.home_angle
            
            if first_proc.get('start_from_center'):
                self.transition_end_angle = arm.coords_to_angle(arm.center_pos_coords)
            else:
                first_step_pos = first_proc['steps'][0]['pos'] if first_proc.get('steps') else 0
                self.transition_end_angle = arm.percent_to_angle(first_step_pos)

    def update(self, dt):
        self.prev_nozzle_pos = self.last_nozzle_pos.copy()

        # [重要] 預先紀錄位置，供渲染插值使用
        self.particles_last_pos[:, 0] = self.particles_pos[:, 0]
        self.particles_last_pos[:, 1] = self.particles_pos[:, 1]
        
        if self.current_process_index < len(self.recipe['processes']):
            current_process = self.recipe['processes'][self.current_process_index]
        else:
            current_process = {'total_duration': 0, 'steps': []}

        if self.animation_state == STATE_RUNNING_PROCESS:
            wall_time_in_proc = self.simulation_time_elapsed - self.time_offset_for_current_process
        else:
            wall_time_in_proc = 0.0

        if self.animation_state == STATE_RUNNING_PROCESS:
            self.current_rpm_value = self._get_rpm_at_time(current_process, wall_time_in_proc)
        
        current_rpm = self.current_rpm_value
        spin_dir = self.recipe.get('spin_dir', 'cw')

        if self.simulation_mode == 'full':
            if self.fast_mode:
                # [優化] AutoTune 快速模式，忽略渲染連續性，確保物理計算穩定即可
                SUB_STEPS = max(1, min(3, int(current_rpm / 300)))
            else:
                # [優化] 動態計算子步數，與 WATER_RENDER_INTERPOLATION_LIMIT 掛鉤，改善高轉速連續性
                SUB_STEPS = max(5, min(WATER_RENDER_INTERPOLATION_LIMIT, 5 + int(current_rpm / 40)))
            
            sub_dt = dt / SUB_STEPS
            
            omega = (current_rpm / 60.0) * 2 * math.pi * (-1 if spin_dir == 'cw' else 1)
            direction_mult = -1 if spin_dir == 'cw' else 1
            
            # 複製 prev_nozzle_pos，處理 list 格式
            last_sub_nozzle_pos = [c.copy() for c in self.prev_nozzle_pos] if isinstance(self.prev_nozzle_pos, list) else self.prev_nozzle_pos.copy()
            for i in range(SUB_STEPS):
                frac = (i + 1) / SUB_STEPS
                
                # 計算子步插值 (處理 list 格式)
                if isinstance(self.prev_nozzle_pos, list):
                    curr_sub_nozzle_pos = [
                        self.prev_nozzle_pos[j] + (self.last_nozzle_pos[j] - self.prev_nozzle_pos[j]) * frac
                        for j in range(len(self.prev_nozzle_pos))
                    ]
                else:
                    curr_sub_nozzle_pos = self.prev_nozzle_pos + (self.last_nozzle_pos - self.prev_nozzle_pos) * frac

                # 計算子步的晶圓角度，用於物理核心的 Impact 座標轉換
                curr_sub_angle = self.wafer_angle + (current_rpm / 60.0 * 360.0 * (i * sub_dt)) * direction_mult
                rad = math.radians(curr_sub_angle)
                cos_w, sin_w = math.cos(rad), math.sin(rad)

                # 1. 生成粒子 (絕對座標)
                if (self.active_arm_id and self.active_arm_id != 0) and (self.animation_state == STATE_RUNNING_PROCESS):
                    self._spawn_particles(self.active_arm_id, sub_dt, curr_sub_nozzle_pos, prev_sub_pos=last_sub_nozzle_pos)

                # 2. 物理步進
                _physics_kernel(
                    self.particles_state, self.particles_pos, self.particles_vel, self.particles_last_pos,
                    self.particles_life, self.particles_time_on_wafer, self.particles_path_length, self.particles_arm_id,
                    sub_dt, omega, cos_w, sin_w,
                    self.viscosities, self.evap_rates, self.surface_tensions,
                    GRAVITY_MMS2, WAFER_RADIUS,
                    self.p_push_strength, self.p_core_radius,
                    self.st_resist_base, self.weber_coeff,
                    self.visc_damping, self.film_thinning_factor, self.drying_visc_mult,
                    self.rpm_evap_coeff
                )
                last_sub_nozzle_pos = [c.copy() for c in curr_sub_nozzle_pos] if isinstance(curr_sub_nozzle_pos, list) else curr_sub_nozzle_pos.copy()

        if self.animation_state == STATE_RUNNING_PROCESS:
            if wall_time_in_proc >= current_process.get('total_duration', 0):
                self._handle_process_transition(current_process)
            elif self.active_arm_id and self.active_arm_id != 0:
                arm = self.arms.get(self.active_arm_id)
                self._calculate_physics_movement(current_process, arm, dt)
        
        elif self.animation_state == STATE_ARM_CHANGE_PAUSE:
            if self.simulation_time_elapsed - self.transition_start_time >= self.arm_change_pause_time:
                self._prepare_next_arm_move()
        elif self.animation_state == STATE_PAUSE_AT_CENTER:
            if self.simulation_time_elapsed - self.transition_start_time >= self.center_pause_time:
                self._prepare_move_center_to_start(current_process)
        else:
            self._handle_arm_transition(current_process)

        direction_mult = -1 if spin_dir == 'cw' else 1
        self.wafer_angle += (current_rpm / 60.0 * 360.0 * dt) * direction_mult
        
        rad = math.radians(self.wafer_angle)
        rot_matrix = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
        base_notch = np.array([[WAFER_RADIUS, 0], [WAFER_RADIUS-NOTCH_DEPTH, NOTCH_HALF_WIDTH], [WAFER_RADIUS-NOTCH_DEPTH, -NOTCH_HALF_WIDTH]])
        self.current_notch_coords = np.dot(base_notch, rot_matrix.T)

        if self.simulation_mode == 'full':
            # 我們需要收集 arm_id=1, 2, 3 的渲染資料，因為 Arm 2 包含了 nozzle 3
            render_data = {arm_id: self._get_render_paths(arm_id, dt, current_rpm, spin_dir) for arm_id in [1, 2, 3]}
        else:
            render_data = {}

        is_finished = False
        if self.current_process_index >= len(self.recipe['processes']) - 1:
            if self.is_looping_back:
                arm = self.arms.get(self.active_arm_id)
                if arm:
                    angle_diff = arm._get_angle_diff(self.transition_end_angle, self.transition_start_angle)
                    dist_arc = abs(angle_diff) * arm.arm_length
                    dur = max(0.05, dist_arc / (self.max_nozzle_speed_mms * self.transition_arm_speed_ratio))
                    if (self.simulation_time_elapsed - self.transition_start_time) >= dur:
                        is_finished = True
                else:
                    is_finished = True
            elif wall_time_in_proc >= current_process.get('total_duration', 0):
                if self.active_arm_id == 0:
                    is_finished = True

        if not self.headless:
            self.simulation_time_elapsed += dt
        else:
            if not is_finished:
                self.simulation_time_elapsed += dt

        # 獲取當前製程的噴嘴流量資訊
        current_flows = {1: 0.0, 2: 0.0, 3: 0.0}
        if self.animation_state == STATE_RUNNING_PROCESS:
            if self.active_arm_id == 1:
                current_flows[1] = current_process.get('flow_rate', 0.0)
            elif self.active_arm_id == 2:
                current_flows[2] = current_process.get('flow_rate', 0.0)
                current_flows[3] = current_process.get('flow_rate_2', 0.0)

        return {
            'time': self.simulation_time_elapsed,
            'state': self.animation_state,
            'active_arm_id': self.active_arm_id,
            'nozzle_pos': self.last_nozzle_pos,
            'wafer_angle': self.wafer_angle,
            'notch_coords': self.current_notch_coords,
            'rpm': current_rpm,
            'process_idx': self.current_process_index,
            'process_time_str': f"{max(0, wall_time_in_proc):.2f}s / {current_process.get('total_duration', 0):.2f}s",
            'step_str': self.current_step_label,
            'water_render': render_data,
            'is_spraying': (self.animation_state == STATE_RUNNING_PROCESS),
            'nozzle_flows': current_flows,
            'removed_particles': [],
            'is_finished': is_finished
        }

    def _calculate_physics_movement(self, process, arm, dt):
        self.cumulative_physics_time += dt
        cycle = process.get('physics_cycle_time', 0)
        segs = process.get('physics_segments', [])
        
        if cycle > 0 and segs:
            t_in = self.cumulative_physics_time % cycle
            t_acc = 0.0
            found = False
            for s in segs:
                if t_acc + s['t'] >= t_in:
                    dt_in_seg = t_in - t_acc
                    pos_pct = s['pi'] + s['vi'] * dt_in_seg + 0.5 * s['a'] * (dt_in_seg**2)
                    self.last_nozzle_pos = arm.percent_to_coords(pos_pct)
                    self.current_step_label = s['label']
                    found = True
                    break
                t_acc += s['t']
            
            if not found and segs:
                last = segs[-1]
                ds = last['t']
                self.last_nozzle_pos = arm.percent_to_coords(last['pi'] + last['vi']*ds + 0.5*last['a']*ds**2)

    def _spawn_particles(self, arm_id, dt, custom_pos=None, prev_sub_pos=None):
        """
        [修正] 生成粒子時，直接使用「絕對座標 (World Coordinates)」。
        並實施 Time Smearing 優化連續性。支援多噴嘴同時噴灑。
        """
        current_process = self.recipe['processes'][self.current_process_index]
        
        # 決定有幾個來源
        sources = [] # list of dicts: {'id', 'flow', 'start_pos', 'end_pos'}
        
        nozzle_end = custom_pos if custom_pos is not None else self.last_nozzle_pos
        nozzle_start = prev_sub_pos if prev_sub_pos is not None else self.prev_nozzle_pos

        if arm_id == 2:
            # 處理雙噴嘴
            if isinstance(nozzle_end, list) and len(nozzle_end) == 2:
                flow_1 = current_process.get('flow_rate', 500.0)
                flow_2 = current_process.get('flow_rate_2', 500.0)
                if flow_1 > 0:
                    sources.append({'id': 2, 'flow': flow_1, 'start_pos': nozzle_start[0], 'end_pos': nozzle_end[0]})
                if flow_2 > 0:
                    sources.append({'id': 3, 'flow': flow_2, 'start_pos': nozzle_start[1], 'end_pos': nozzle_end[1]})
            else:
                sources.append({'id': 2, 'flow': current_process.get('flow_rate', 500.0), 'start_pos': nozzle_start, 'end_pos': nozzle_end})
        else:
            sources.append({'id': arm_id, 'flow': current_process.get('flow_rate', 500.0), 'start_pos': nozzle_start, 'end_pos': nozzle_end})

        for source in sources:
            src_id = source['id']
            flow = source['flow']
            if flow <= 0: continue
            
            params = self.water_params.get(src_id, {})
            st_val = params.get('surface_tension', 72.8)
            spread_base = self.spray_spread_base / (st_val + 10.0) 
            
            # 使用 PARTICLE_SPAWN_MULTIPLIER 調整粒子生成密度
            expected_particles = (flow * 0.5 * PARTICLE_SPAWN_MULTIPLIER) * dt
            
            if hasattr(self, 'fast_mode') and self.fast_mode:
                expected_particles *= self.fast_particle_scale
                
            self._spawn_accumulator[src_id] += expected_particles
            
            count = int(self._spawn_accumulator[src_id])
            self._spawn_accumulator[src_id] -= count 
            
            if count <= 0: continue
            
            inactive_indices = np.where(self.particles_state == P_INACTIVE)[0]
            if len(inactive_indices) == 0: continue
            
            spawn_count = min(count, len(inactive_indices))
            target_indices = inactive_indices[:spawn_count]
            
            n_start = source['start_pos']
            n_end = source['end_pos']
            
            for i, idx in enumerate(target_indices):
                t_frac = (i + random.random()) / spawn_count
                spawn_time_offset = t_frac * dt
                
                self.particles_state[idx] = P_FALLING
                self.particles_life[idx] = 1.0
                self.particles_birth_time[idx] = self.simulation_time_elapsed + spawn_time_offset
                self.particles_time_on_wafer[idx] = 0.0
                self.particles_path_length[idx] = 0.0
                self.particles_arm_id[idx] = src_id
                self.particles_id[idx] = self.next_particle_id
                
                # [優化] 實施 Time Smearing：
                # 1. 插值水平位置 (噴嘴移動路徑)
                interp_nozzle = n_start + (n_end - n_start) * t_frac
                off = (np.random.rand(2) - 0.5) * spread_base
                
                # 2. 插值垂直位置 (下落連續性)
                # 初速設定 (向下為負)
                self.particles_vel[idx] = [0.0, 0.0, -flow * self.jet_speed_factor]
                
                # 補償計算：為了抵消後續物理步進的影響，較晚出生的粒子初始高度應稍微調高
                # z_start = H + v_z_speed * t_offset = H - vel_z * t_offset
                v_z_initial = self.particles_vel[idx, 2]
                z_offset = -v_z_initial * spawn_time_offset
                
                self.particles_pos[idx] = [
                    interp_nozzle[0] + off[0], 
                    interp_nozzle[1] + off[1], 
                    NOZZLE_Z_HEIGHT + z_offset
                ]
                self.particles_last_pos[idx] = self.particles_pos[idx, :2]
                
                self.next_particle_id += 1

    def _get_render_paths(self, arm_id, dt, rpm, spin_dir):
        """
        產生渲染路徑。
        - Falling 粒子：絕對座標，直接渲染。
        - On Wafer 粒子：相對座標，需轉回絕對座標。
        """
        f_xy = []
        o_xy = []
        
        # 用於渲染的旋轉角度 (使用當前晶圓角度)
        rad_wafer = math.radians(self.wafer_angle)
        cos_t, sin_t = math.cos(rad_wafer), math.sin(rad_wafer)
        
        mask = (self.particles_arm_id == arm_id) & (self.particles_state != P_INACTIVE)
        indices = np.where(mask)[0]
        
        for i in indices:
            state = self.particles_state[i]
            pos = self.particles_pos[i]
            
            if state == P_FALLING:
                # [修正] 落下粒子是絕對座標，直接渲染
                f_xy.append((pos[0], pos[1]))
                    
            elif state == P_ON_WAFER:
                # [修正] 晶圓上粒子是相對座標 -> 轉回絕對座標供 UI 繪製
                # P_abs = R * P_rel
                abs_x = pos[0] * cos_t - pos[1] * sin_t
                abs_y = pos[0] * sin_t + pos[1] * cos_t
                o_xy.append((abs_x, abs_y))
                
        return {
            'falling': np.array(f_xy) if f_xy else np.empty((0, 2)),
            'on_wafer': np.array(o_xy) if o_xy else np.empty((0, 2))
        }

    def _handle_process_transition(self, current_process_obj):
        arm = self.arms.get(self.active_arm_id)
        current_angle = arm.coords_to_angle(self.last_nozzle_pos) if arm else 0.0
        curr_time = self.simulation_time_elapsed

        next_idx = self.current_process_index + 1
        if next_idx >= len(self.recipe['processes']):
            if self.active_arm_id and self.active_arm_id != 0:
                self.animation_state = STATE_ARM_MOVE_TO_HOME
                self.transition_start_time = curr_time
                self.transition_start_angle = current_angle
                self.transition_end_angle = arm.home_angle
                self.is_looping_back = True
            else:
                if not self.headless:
                    self._reset_to_start()
            return

        prev_p = current_process_obj
        next_p = self.recipe['processes'][next_idx]
        self.current_process_index = next_idx

        if prev_p['arm_id'] == next_p['arm_id'] and prev_p['arm_id'] != 0:
            self.transition_start_time = curr_time
            self.transition_start_angle = current_angle
            if next_p.get('start_from_center'):
                self.animation_state = STATE_MOVING_TO_CENTER_ARC
                self.transition_end_angle = arm.coords_to_angle(arm.center_pos_coords)
            else:
                self.animation_state = STATE_ARM_MOVE_FROM_HOME
                target_pos = next_p['steps'][0]['pos'] if next_p.get('steps') else 0
                self.transition_end_angle = arm.percent_to_angle(target_pos)
        else:
            if prev_p['arm_id'] != 0 and arm:
                self.animation_state = STATE_ARM_MOVE_TO_HOME
                self.transition_start_time = curr_time
                self.transition_start_angle = current_angle
                self.transition_end_angle = arm.home_angle
            else:
                self._prepare_next_arm_move()

    def _handle_arm_transition(self, current_process):
        arm = self.arms.get(self.active_arm_id)
        if not arm: return
        
        angle_diff = arm._get_angle_diff(self.transition_end_angle, self.transition_start_angle)
        dist_arc = abs(angle_diff) * arm.arm_length
        dur = max(0.05, dist_arc / (self.max_nozzle_speed_mms * self.transition_arm_speed_ratio))
        
        t = self.simulation_time_elapsed - self.transition_start_time
        
        if t >= dur:
            self.last_nozzle_pos = arm.angle_to_coords(self.transition_end_angle)
            if self.animation_state == STATE_ARM_MOVE_TO_HOME:
                if self.is_looping_back:
                    if not self.headless:
                        self._reset_to_start()
                else:
                    if self.recipe['processes'][self.current_process_index]['arm_id'] == 0:
                        self.animation_state, self.active_arm_id, self.time_offset_for_current_process = STATE_RUNNING_PROCESS, 0, self.simulation_time_elapsed
                    else:
                        self.animation_state, self.transition_start_time = STATE_ARM_CHANGE_PAUSE, self.simulation_time_elapsed
            elif self.animation_state == STATE_ARM_MOVE_FROM_HOME:
                if current_process.get('start_from_center'):
                    self.animation_state, self.transition_start_time = STATE_MOVING_TO_CENTER_ARC, self.simulation_time_elapsed
                    self.transition_start_angle, self.transition_end_angle = arm.coords_to_angle(self.last_nozzle_pos), arm.coords_to_angle(arm.center_pos_coords)
                else:
                    self.animation_state, self.time_offset_for_current_process, self.cumulative_physics_time = STATE_RUNNING_PROCESS, self.simulation_time_elapsed, 0.0
            elif self.animation_state == STATE_MOVING_TO_CENTER_ARC:
                self.animation_state, self.transition_start_time = STATE_PAUSE_AT_CENTER, self.simulation_time_elapsed
            elif self.animation_state == STATE_MOVING_FROM_CENTER_TO_START:
                self.animation_state, self.time_offset_for_current_process, self.cumulative_physics_time = STATE_RUNNING_PROCESS, self.simulation_time_elapsed, current_process.get('sfc_start_time', 0.0)
        else:
            frac = t / dur
            self.last_nozzle_pos = arm.get_interpolated_coords(self.transition_start_angle, self.transition_end_angle, frac)

    def _prepare_next_arm_move(self):
        next_p = self.recipe['processes'][self.current_process_index]
        self.active_arm_id = next_p['arm_id']
        arm = self.arms.get(self.active_arm_id)
        if arm:
            self.animation_state, self.transition_start_time = STATE_ARM_MOVE_FROM_HOME, self.simulation_time_elapsed
            self.transition_start_angle, self.last_nozzle_pos = arm.home_angle, arm.home_pos.copy()
            if next_p.get('start_from_center'):
                self.transition_end_angle = arm.coords_to_angle(arm.center_pos_coords)
            else:
                first_pos = next_p['steps'][0]['pos'] if next_p.get('steps') else 0
                self.transition_end_angle = arm.percent_to_angle(first_pos)
        else:
            self.animation_state, self.time_offset_for_current_process = STATE_RUNNING_PROCESS, self.simulation_time_elapsed

    def _prepare_move_center_to_start(self, current_process):
        self.animation_state, self.transition_start_time = STATE_MOVING_FROM_CENTER_TO_START, self.simulation_time_elapsed
        arm = self.arms[self.active_arm_id]
        self.transition_start_angle = arm.coords_to_angle(arm.center_pos_coords)
        target_idx = current_process.get('sfc_target_idx', 0)
        if current_process.get('steps'):
            target_pos = current_process['steps'][target_idx]['pos']
            self.transition_end_angle = arm.percent_to_angle(target_pos)
        else:
            self.transition_end_angle = self.transition_start_angle

    def _pre_calculate_physics(self):
        for process in self.recipe['processes']:
            arm_id = process.get('arm_id', 0)
            if not arm_id or arm_id == 0: continue
            arm = self.arms[arm_id]
            steps = process.get('steps', [])
            if len(steps) < 2: continue
            
            def create_segments(step_list, is_forward=True):
                segs = []
                for j in range(len(step_list) - 1):
                    p_i, p_f = float(step_list[j]['pos']), float(step_list[j+1]['pos'])
                    v_i_mag = (float(step_list[j].get('speed', 0)) / 100.0) * arm.max_percent_speed
                    v_f_mag = (float(step_list[j+1].get('speed', 0)) / 100.0) * arm.max_percent_speed
                    
                    dist = p_f - p_i
                    if abs(dist) < 1e-6:
                        continue
                    
                    direction = 1.0 if dist > 0 else -1.0
                    v_i = v_i_mag * direction
                    v_f = v_f_mag * direction
                    
                    v_avg_mag = (v_i_mag + v_f_mag) / 2.0
                    if v_avg_mag < 0.1:
                        v_avg_mag = 0.1
                    
                    t_d = abs(dist) / v_avg_mag
                    accel = (v_f - v_i) / t_d if t_d > 0 else 0
                    
                    label_from = j + 1 if is_forward else len(step_list) - j
                    label_to = j + 2 if is_forward else len(step_list) - j - 1
                    
                    segs.append({
                        'pi': p_i,
                        'vi': v_i,
                        'a': accel,
                        't': max(t_d, 0.01),
                        'label': f"Step {label_from}->{label_to}"
                    })
                return segs

            f_segs = create_segments(steps, is_forward=True)
            b_segs = create_segments(steps[::-1], is_forward=False)
            
            all_s = f_segs + b_segs
            process['physics_segments'], process['physics_cycle_time'] = all_s, sum(s['t'] for s in all_s)
            
            sfc_t, sfc_idx, min_d = 0.0, 0, float('inf')
            if process.get('start_from_center'):
                for k, step in enumerate(steps):
                    d = abs(step['pos'] - arm.center_pos_percent)
                    if d < min_d: min_d, sfc_idx = d, k
                
                sfc_t = 0.0
                for m in range(min(sfc_idx, len(f_segs))):
                    sfc_t += f_segs[m]['t']
                    
            process['sfc_start_time'], process['sfc_target_idx'] = sfc_t, sfc_idx

    def _get_rpm_at_time(self, process, time_in_proc):
        spin, total_d = process.get('spin_params', {}), process.get('total_duration', 1.0)
        if spin.get('mode', 'Simple') == 'Simple': return float(spin.get('rpm', 0))
        sr, er = float(spin.get('start_rpm', 0)), float(spin.get('end_rpm', 0))
        return sr + (er - sr) * max(0.0, min(1.0, time_in_proc / total_d)) if total_d > 0 else sr

    def _reset_to_start(self): self.reset()