# 定義參數結構：Key 為參數名稱，Value 為 UI 設定
# 格式: (Label顯示名稱, 預設值, 變數類型, (最小值, 最大值), 提示訊息, 是否可擬合(is_tunable))
# 變數類型: 'float', 'int'

PARAMETER_DEFINITIONS = {
    "General": {
        "MAX_NOZZLE_SPEED_MMS":       ("Max Nozzle Speed (mm/s)", 250.0, 'float', (1.0, 1000.0), "Arm 最快速度(mm/s)，EOS 為250，DNS 為366", False),
        "TRANSITION_ARM_SPEED_RATIO": ("Trans. Speed Ratio", 0.8, 'float', (0.1, 2.0), "Arm 不噴灑時移動的速度ß", False),
        "ARM_CHANGE_PAUSE_TIME":      ("Arm Change Pause (s)", 1.0, 'float', (0.0, 10.0), "Arm 切換之間的停頓時間 (s)", False),
        "CENTER_PAUSE_TIME":          ("Center Pause (s)", 0.8, 'float', (0.0, 10.0), "Arm 抵達晶圓中心後停頓的時間 (s)", False),
        "REPORT_INTERVAL_MM":         ("Report Interval (mm)", 2.0, 'float', (0.1, 50.0), "Simulation Report 徑向間隔 (mm)", False),
        "REPORT_LOG_INTERVAL":        ("Report Log Interval (s)", 0.01, 'float', (0.001, 5.0), "Simulation Report 時間記錄間隔 (s)", False),
    },
    "Etching Amount": {
        "ETCHING_GLOBAL_SCALE":       ("Global Scale", 1.0, 'float', (0.0001, 100.0), "全域縮放參數。", True),
        "GRID_SIZE":                  ("Grid Size (radius)", 5.0, 'float', (0.0000001, 150.0), "單個粒子的影響半徑 (mm)。影響渲染的解析度與路徑平滑度。", False),
        "ETCHING_TAU":                ("Etching Tau", 0.3, 'float', (0.0000001, 50.0), "化學老化常數 (s)。模擬藥液活性隨時間衰減的速度。", True),
        "ETCHING_SATURATION_THICKNESS":("Saturation Thickness", 0.002, 'float', (0.00000001, 10.0), "反應飽和與膜厚。模擬化學反應在表面完全潤濕後的飽和上限。", True),
        "ETCHING_BASE_SPIN_DECAY":    ("Base Spin Decay", 2.0, 'float', (0.00000001, 10.0), "基礎甩乾速率。模擬液體因旋轉與蒸發離開表面的速度。", True),
        "ETCHING_IMPINGEMENT_BONUS":  ("Impingement Bonus", 1.2, 'float', (1.0, 50.0), "衝擊加成倍數。噴嘴正下方新鮮藥液撞擊帶來的蝕刻增益。", True),
        "ETCHING_GEO_SMOOTHING":      ("Geo Smoothing", 0.1, 'float', (0.0, 150.0), "幾何平滑係數。配合平方項校正公式，用於微調中心點的數值。", True),
        "ETCHING_SATURATION_THRESHOLD":("Sat. Threshold", 0.0, 'float', (0.0, 10.0), "最終蝕刻量飽和閥值。用於 np.tanh 限制極端值的數學處理。", True),
        "ETCHING_SHEAR_COEFF":        ("Shear Coeff", 0.0001, 'float', (0.0, 0.1), "剪切應力加成係數。模擬高速流動時邊界層變薄、傳質速率增加的現象。", True),
    },
    "Particle Removal": {
        "PRE_ALPHA":                  ("Alpha (Shear)", 0.001, 'float', (0.0, 1.0), "剪切項係數", True),
        "PRE_BETA":                   ("Beta (Impact)", 0.5, 'float', (0.0, 10.0), "衝擊項保底係數", True),
        "PRE_GRID_SIZE":              ("PRE Grid Size (mm)", 5.0, 'float', (1.0, 30.0), "清洗影響半徑 (mm)", False),
        "PRE_Q_REF":                  ("Q Ref (mL/min)", 1000.0, 'float', (100.0, 5000.0), "參考流量 (mL/min)", False),
        "PRE_GAMMA_BASE":             ("Gamma Base", 0.001, 'float', (0.0, 1.0), "基礎再附著係數 (1/mm)", True),
        "PRE_DEFECT_COUNT":           ("Initial Defect Count", 10000, 'int', (10, 100000), "模擬進站初始缺陷數量", False),
        "PRE_DEFECT_CALI":            ("Defect Map Cali", 0.5, 'float', (0.01, 10.0), "缺陷殘留機率校正係數", True),
    },
    "Charging Simulation": {
        # 1. 流體基礎電性
        "FLUID_CONDUCTIVITY":         ("Conductivity (S/m)", 5.0e-12, 'float', (1.0e-16, 10.0), "藥液導電率。低導電率 (如 1e-12) 較易觀察到電荷累積。", True),
        "FLUID_RELATIVE_PERMITTIVITY":("Rel. Permittivity", 80.0, 'float', (1.0, 100.0), "相對介電常數。水約 80，IPA 約 18。", False),
        
        # 2. 電荷分離與轉速修正 (解決轉速矛盾)
        "CHARGING_EFFICIENCY":        ("Base Efficiency", -1.0e-7, 'float', (-1.0, 1.0), "基礎生成係數。TEOS 表面通常設為負值。", True),
        "CHARGING_RPM_FACTOR":        ("RPM Scaling Factor", 5.0, 'float', (0.0, 50.0), "轉速增強因子。模擬高轉速剪切力導致的非線性電荷分離。", True),
        
        # 3. 表面電荷物理 (解決平滑度與擴散)
        "SURFACE_DIFFUSION_COEFF":    ("Surface Diffusion", 0.1, 'float', (0.0, 1.0), "表面電荷橫向擴散係數。數值越高，KPFM 曲線越平滑。", True),
        "DIFFUSION_STABILITY_LIMIT":  ("Stability Limit (α)", 0.25, 'float', (0.1, 0.25), "數值穩定性限制 (Fourier Number)。維持 0.25 以確保不震盪。", False),
        
        # 4. 製程與量測映射
        "CHARGING_SPRAY_RADIUS":      ("Spray Radius (mm)", 2.0, 'float', (0.5, 10.0), "噴嘴水滴撞擊產生的電荷分離有效半徑。", False),
        "CHARGING_BASE_SPIN_DECAY":   ("Liquid Spin Decay", 2.0, 'float', (1e-8, 10.0), "液體甩乾速率。僅影響 Mobile Charge 的流失速度。", True),
        "KPFM_CAPACITANCE":           ("KPFM Equiv. Cap", 1.0e-10, 'float', (1e-12, 1e-8), "KPFM 等效電容因子。用於將模擬庫倫值映射到實驗伏特值。", True),
    },
    "Advanced Physics Parameters": {
        "PHYSICS_PRESSURE_PUSH_STRENGTH": ("Pressure Push", 5.0, 'float', (0.0, 5000.0), "中心區域推力強度 (解決中心堆積)", True),
        "PHYSICS_PRESSURE_CORE_RADIUS":   ("Core Radius (mm)", 80.0, 'float', (1.0, 150.0), "中心推力影響半徑 (mm)", False),
        "PHYSICS_ST_RESIST_BASE":         ("ST Resist Base", 0.3, 'float', (0.0, 10.0), "表面張力基礎阻力係數", True),
        "PHYSICS_WEBER_COEFF":            ("Weber Coeff", 0.01, 'float', (0.0, 0.1), "韋伯數係數 (速度對張力的削弱)", True),
        "PHYSICS_VISCOSITY_DAMPING":      ("Visc. Damping", 2.0, 'float', (0.0, 10.0), "基礎阻尼係數 (控制整體流動)", True),
        "PHYSICS_FILM_THINNING_FACTOR":   ("Thinning Factor", 1.0, 'float', (0.0, 10.0), "膜厚變薄阻力係數 (Emslie 模型)", True),
        "PHYSICS_DRYING_VISC_MULT":       ("Drying Visc Mult", 5.0, 'float', (1.0, 20.0), "乾燥時的黏度倍率", True),
        "PHYSICS_RPM_EVAP_COEFF":         ("RPM Evap Coeff", 0.005, 'float', (0.0, 0.1), "轉速依賴蒸發係數", True),
        "PHYSICS_SPRAY_SPREAD_BASE":      ("Spray Spread", 600.0, 'float', (100.0, 2000.0), "噴嘴擴散基數 (霧化程度)", False),
        "PHYSICS_JET_SPEED_FACTOR":       ("Jet Speed Factor", 0.05, 'float', (0.0, 1.0), "噴嘴垂直初速係數", True),
    }
}

def get_default_config():
    """回傳一個扁平化的預設配置字典"""
    config = {}
    for section in PARAMETER_DEFINITIONS.values():
        for key, val in section.items():
            config[key] = val[1]
    return config
