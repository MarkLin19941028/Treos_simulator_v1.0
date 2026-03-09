import numpy as np

# --- Constants ---
WAFER_DIAMETER = 300
WAFER_RADIUS = WAFER_DIAMETER / 2
CHAMBER_SIZE = 850
FPS = 30
REPORT_FPS = 800  # 目前已根據 recipe 動態調整，這個只是預設最低數值
NOTCH_DEPTH = 15
NOTCH_HALF_WIDTH = 7.5

# Simulation Report 參數 (Moved to simulation_config_def.py)
REPORT_INTERVAL_MM = 2

# --- Simulation Density Parameters ---
PARTICLE_MAX_COUNT = 8000         # 調降最大粒子數，顯著減輕 CPU 繪圖負擔
PARTICLE_SPAWN_MULTIPLIER = 3.0   # 粒子生成速率的乘數

# --- Physics & System Hard Parameters ---
# 1. 中心壓力梯度 (Pressure Gradient)
PHYSICS_PRESSURE_PUSH_STRENGTH = 500.0
PHYSICS_PRESSURE_CORE_RADIUS = 20.0

# 2. 表面張力 (Surface Tension)
PHYSICS_ST_RESIST_BASE = 0.5
PHYSICS_WEBER_COEFF = 0.001

# 3. 黏滯力與阻尼 (Viscosity & Damping)
PHYSICS_VISCOSITY_DAMPING = 2.0
PHYSICS_FILM_THINNING_FACTOR = 1.0
PHYSICS_DRYING_VISC_MULT = 5.0

# 4. 蒸發 (Evaporation)
PHYSICS_RPM_EVAP_COEFF = 0.005

# 5. 噴嘴生成 (Spawning)
PHYSICS_SPRAY_SPREAD_BASE = 600.0
PHYSICS_JET_SPEED_FACTOR = 0.05

# --- 視覺連貫性優化參數 ---
WATER_RENDER_INTERPOLATION_LIMIT = 30 # 提高插值點上限，改善高轉速連續性
WATER_JITTER_AMOUNT = 3.0            # 隨機微擾動幅度 (mm)
WATER_DROP_SIZE = 4                  # 噴嘴下落水滴尺寸
WATER_ON_WAFER_SIZE = 10              # 調降水滴尺寸，增加顆粒度以利連點成線

# --- Physics Constants ---
GRAVITY_MMS2 = 9800  # Gravity in mm/s^2

# --- Simulation Geometry & Speed ---
NOZZLE_RADIUS_MM = 2.0            # 噴嘴半徑 (mm)
NOZZLE_Z_HEIGHT = 15.0            # 噴嘴到晶圓的初始垂直距離 (mm)
TRANSITION_ARM_SPEED_RATIO = 0.8  # Arm 轉換狀態下的速度比例 (用於乘以 MAX_NOZZLE_SPEED_MMS)

# --- Etching Amount Simulation Constants ---
ETCHING_TAU = 0.3                 # 老化模型衰減常數 (s)
GRID_SIZE = 5.0                   # 蝕刻影響半徑 (mm)
ETCHING_IMPINGEMENT_BONUS = 1.2   # 衝擊區的強度加成倍數
ETCHING_GEO_SMOOTHING = 0.1       # 幾何釋平滑常數
ETCHING_SATURATION_THRESHOLD = 0.002 # 每一步長單個像素點的最大蝕刻貢獻飽和值
ETCHING_SATURATION_THICKNESS = 0.2   # 飽和膜厚 (mm)
ETCHING_BASE_SPIN_DECAY = 2.0        # 基礎甩乾速率 (1/s)

# --- Charging Simulation Constants ---
VACUUM_PERMITTIVITY = 8.854e-12  # F/m (真空介電常數)
WATER_RELATIVE_PERMITTIVITY = 80.0 # 相對介電常數 (水)
DEFAULT_CONDUCTIVITY = 5.0e-6    # S/m (Siemens per meter)
CHARGING_BASE_SPIN_DECAY = 2.0        # 基礎甩乾速率 (1/s)

# --- PRE (Particle Removal Efficiency) Constants ---
PRE_ALPHA = 0.001               # 剪切項係數
PRE_BETA = 0.5                    # 衝擊項保底係數
PRE_GRID_SIZE = 10.0               # 清洗影響半徑 (mm)
PRE_Q_REF = 1000.0                # 參考流量 (mL/min)
PRE_GAMMA_BASE = 0.001            # 基礎再附著係數 (1/mm)

# --- Timing & Pause ---
ARM_CHANGE_PAUSE_TIME = 1.0       # Arm 切換之間的停頓時間 (s)
CENTER_PAUSE_TIME = 0.8           # Arm 抵達晶圓中心後停頓的時間 (s)

# State Machine Constants
STATE_RUNNING_PROCESS = "RUNNING_PROCESS"
STATE_ARM_MOVE_FROM_HOME = "ARM_MOVE_FROM_HOME"
STATE_ARM_MOVE_TO_HOME = "ARM_MOVE_TO_HOME"
STATE_ARM_CHANGE_PAUSE = "ARM_CHANGE_PAUSE"
STATE_PAUSE_AT_CENTER = "PAUSE_AT_CENTER"
STATE_MOVING_TO_CENTER_ARC = "MOVING_TO_CENTER_ARC"
STATE_MOVING_FROM_CENTER_TO_START = "MOVING_FROM_CENTER_TO_START"

# --- Arm Geometric Definitions ---

ARM_GEOMETRIES = {
    1: {
        "pivot": np.array([-264.1, -189.41]), 
        "length": 325.0,
        "home": np.array([-264.1, 135.59])
    },
    2: {
        "pivot": np.array([264.1, -189.41]), 
        "length": 325.0,
        "home": np.array([264.1, 135.59]),
        "side_arm_length": 77.8,
        "side_arm_angle_offset": np.deg2rad(60), # 60 degrees offset for the side arm
        "side_arm_branch_dist": 215.5 # 分支點距離 Pivot 的距離 (mm)
    }
}
