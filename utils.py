import math
import numpy as np
from constants import WAFER_RADIUS, NOZZLE_RADIUS_MM

def calculate_water_velocity(flow_rate_ml_per_min):
    """
    根據流量(mL/min)計算水柱的初始速度(mm/s)。
    1 mL = 1 cm^3 = 1000 mm^3
    """
    # 這裡建議優先使用從 constants 匯入的 NOZZLE_RADIUS_MM，而非硬編碼
    # 如果 constants 沒定義，則預設為 2.0
    r = NOZZLE_RADIUS_MM if 'NOZZLE_RADIUS_MM' in globals() else 2.0
    
    flow_rate_mm3_per_s = flow_rate_ml_per_min * 1000.0 / 60.0
    nozzle_area_mm2 = math.pi * (r ** 2)
    
    # 避免除以零
    if nozzle_area_mm2 == 0:
        return 0.0
        
    velocity_mm_per_s = flow_rate_mm3_per_s / nozzle_area_mm2
    return velocity_mm_per_s

def calculate_water_counts_by_radius(drops_coordinates, wafer_radius=WAFER_RADIUS, interval_size=9):
    """
    根據水滴在晶圓上的座標，計算不同半徑區間內的水滴數量。
    此函數常用於匯出 CSV 報表時的統計。
    """
    num_intervals = math.ceil(wafer_radius / interval_size)
    counts = {}
    column_headers = []

    # 建立統計區間標題
    for i in range(num_intervals):
        start_radius = i * interval_size
        end_radius_label = start_radius + interval_size - 1
        header = f"Water cts_{start_radius}-{end_radius_label}mm"
        counts[header] = 0
        column_headers.append(header)

    if drops_coordinates is None or len(drops_coordinates) == 0:
        return counts

    # 統計落點
    for coord in drops_coordinates:
        # 支援 tuple 或 numpy array 格式的座標
        x, y = coord[0], coord[1]
        distance = math.sqrt(x**2 + y**2)
        
        if distance <= wafer_radius:
            interval_index = math.floor(distance / interval_size)
            # 安全檢查，確保索引不超出範圍
            if interval_index < len(column_headers):
                target_header = column_headers[interval_index]
                counts[target_header] += 1

    return counts