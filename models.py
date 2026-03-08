import math
import numpy as np

class DispenseArm:
    def __init__(self, arm_id, pivot, home, length, 
                 arm_artist=None, nozzle_artist=None, max_nozzle_speed_mms=250.0,
                 side_arm_length=None, side_arm_angle_offset=None, side_arm_branch_dist=None,
                 side_arm_artist=None, side_nozzle_artist=None, wafer_radius=150.0):
        self.id = arm_id
        # 強制轉為 float numpy array，避免數據類型混亂
        self.pivot_pos = np.array(pivot, dtype=float)
        self.home_pos = np.array(home, dtype=float)
        self.arm_length = float(length)
        
        # Side arm 參數 (Arm 2 專用)
        self.side_arm_length = float(side_arm_length) if side_arm_length is not None else None
        self.side_arm_angle_offset = float(side_arm_angle_offset) if side_arm_angle_offset is not None else None
        self.side_arm_branch_dist = float(side_arm_branch_dist) if side_arm_branch_dist is not None else None
        
        # 綁定繪圖物件 (Headless 模式下可能為 None)
        self.arm_line = arm_artist
        self.nozzle_head = nozzle_artist
        self.side_arm_line = side_arm_artist
        self.side_nozzle_head = side_nozzle_artist
        
        # --- 幾何計算區域：計算 Wafer 邊緣交點 (-100% 和 100%) ---
        # 求解: X^2 + Y^2 = R^2 以及 (X-Px)^2 + (Y-Py)^2 = L^2
        # 我們知道 Pivot 到原點的距離 D = sqrt(Px^2 + Py^2)
        # 在我們的設定中，D 大約等於 L (325)。因此原點 (0,0) 在圓上，即中心點
        Px, Py = self.pivot_pos
        R = wafer_radius
        D_sq = Px**2 + Py**2
        D = math.sqrt(D_sq)
        
        # 交點計算 (兩圓交點公式)
        # d = D, r1 = R, r2 = self.arm_length
        # a = (r1^2 - r2^2 + d^2) / (2*d)
        a = (R**2 - self.arm_length**2 + D_sq) / (2 * D)
        h_sq = R**2 - a**2
        
        if h_sq < 0:
            # 避免精度誤差導致微小的負數
            h_sq = 0
            
        h = math.sqrt(h_sq)
        
        P2x = 0 + a * (-Px) / D # 因為原點是 (0,0)，向量是從原點指向 Pivot 的反方向嗎？
        P2y = 0 + a * (-Py) / D
        # 正確的 P2 點：P2 = O + a * (P - O) / D = (a * Px / D, a * Py / D)
        P2x = a * Px / D
        P2y = a * Py / D
        
        # 交點 1 和 2
        intersect1_x = P2x + h * Py / D
        intersect1_y = P2y - h * Px / D
        
        intersect2_x = P2x - h * Py / D
        intersect2_y = P2y + h * Px / D
        
        self.p_start = np.array([intersect1_x, intersect1_y])
        self.p_end = np.array([intersect2_x, intersect2_y])
        
        self.theta_start = math.atan2(self.p_start[1] - self.pivot_pos[1], self.p_start[0] - self.pivot_pos[0])
        self.theta_end = math.atan2(self.p_end[1] - self.pivot_pos[1], self.p_end[0] - self.pivot_pos[0])
        self.home_angle = math.atan2(self.home_pos[1] - self.pivot_pos[1], self.home_pos[0] - self.pivot_pos[0])
        
        # 確保 theta_start 到 theta_end 的掃掠方向經過中心點
        center_angle = math.atan2(0 - self.pivot_pos[1], 0 - self.pivot_pos[0])
        
        # 排序 theta_start 和 theta_end 以匹配手臂從外向內或特定的方向
        # 通常我們希望 -100% 是一側，+100% 是另一側。
        # 我們強制讓 p_start 對應到更右側或特定的方向，或者讓角度差保持一致
        diff_1 = self._get_angle_diff(self.theta_start, center_angle)
        diff_2 = self._get_angle_diff(self.theta_end, center_angle)
        
        if diff_1 > diff_2:
            self.theta_start, self.theta_end = self.theta_end, self.theta_start
            self.p_start, self.p_end = self.p_end, self.p_start

        # 針對 Arm 1 的特殊需求，將正負方向顛倒
        if self.id == 1:
            self.theta_start, self.theta_end = self.theta_end, self.theta_start
            self.p_start, self.p_end = self.p_end, self.p_start

        self.center_pos_coords = self.angle_to_coords(center_angle)
        
        # 對於圓弧來說，中心角度剛好是兩側交點的中心
        # 所以 center_pos_percent 就是 0
        self.center_pos_percent = 0.0

        # 計算最大速度百分比 (用於物理計算)
        self.update_max_speed(max_nozzle_speed_mms)

    def update_max_speed(self, new_max_speed_mms):
        """動態更新手臂的最大百分比速度 (用於物理運算與 UI 同步)"""
        angle_diff = self._get_angle_diff(self.theta_end, self.theta_start)
        arc_length = self.arm_length * abs(angle_diff)
        self.max_percent_speed = (new_max_speed_mms / arc_length) * 200 if arc_length > 0 else 0

    def _get_angle_diff(self, a1, a2):
        """計算兩個角度之間的最小差值 (-pi 到 pi)"""
        return (a1 - a2 + math.pi) % (2 * math.pi) - math.pi

    def angle_to_coords(self, angle):
        """輸入角度(弧度)，回傳 (x, y) 座標
        如果包含 side arm，回傳 [main_coords, side_coords]
        否則回傳單一座標 (為保持向下相容性，這裡設計回傳 np.array 或是 list of np.array)
        """
        x = self.pivot_pos[0] + self.arm_length * math.cos(angle)
        y = self.pivot_pos[1] + self.arm_length * math.sin(angle)
        main_coords = np.array([x, y])
        
        if self.side_arm_length is not None and self.side_arm_angle_offset is not None:
            # 計算 side arm nozzle 的位置
            side_angle = angle + self.side_arm_angle_offset
            # side arm 從 main_coords 延伸出去，或是直接從 pivot 出發？
            # 根據架構圖，它似乎從主幹的某個點分岔出來，或者從同一個 pivot 延伸但長度不同？
            # 「連接Nozzle3的Side arm的長度 = 7.78」
            # 這裡我們假設它是從主要臂(主 nozzle 點附近，或某個節點)延伸，但圖中看起來像 Y 字。
            # 為簡單且精確起見，我們先假設它從主噴嘴的位置延伸回去，或者從 pivot 直接算一個等效向量。
            # 看圖，Side arm 是連接到主 Arm 的。
            # 我們假設連接點距離主噴嘴 d 的位置。為簡化，我們直接從 pivot 計算其最終位置向量：
            # 等效作法：假設分岔點距離 pivot 為 L - d，然後旋轉一定角度。
            # 為了讓它像您圖中的 Y 字型：分岔點由 side_arm_branch_dist 決定
            branch_dist = self.side_arm_branch_dist if self.side_arm_branch_dist is not None else (self.arm_length - 80.0)
            bx = self.pivot_pos[0] + branch_dist * math.cos(angle)
            by = self.pivot_pos[1] + branch_dist * math.sin(angle)
            
            sx = bx + self.side_arm_length * math.cos(side_angle)
            sy = by + self.side_arm_length * math.sin(side_angle)
            side_coords = np.array([sx, sy])
            return [main_coords, side_coords]
            
        return main_coords

    def percent_to_angle(self, percent):
        """輸入路徑百分比 (-100~100)，回傳角度"""
        original_ratio = (percent + 100) / 200.0
        angle_diff = self._get_angle_diff(self.theta_end, self.theta_start)
        return self.theta_start + original_ratio * angle_diff

    def percent_to_coords(self, percent):
        """輸入路徑百分比，回傳 (x, y) 座標"""
        return self.angle_to_coords(self.percent_to_angle(percent))

    def coords_to_angle(self, coords):
        """輸入座標，回傳相對於轉軸的角度"""
        if isinstance(coords, list):
            main_coords = coords[0]
        else:
            main_coords = coords
        return math.atan2(main_coords[1] - self.pivot_pos[1], main_coords[0] - self.pivot_pos[0])

    def get_interpolated_coords(self, start_angle, end_angle, progress_ratio):
        """計算過渡期間的插值座標"""
        ratio = max(0.0, min(1.0, progress_ratio))
        angle_diff = self._get_angle_diff(end_angle, start_angle)
        current_angle = start_angle + ratio * angle_diff
        return self.angle_to_coords(current_angle)

    def update_artists(self, coords, color=None):
        """更新 GUI 上的圖形元件位置與顏色"""
        # coords 可能是單一座標或 [main_coords, side_coords]
        if isinstance(coords, list):
            main_coords = coords[0]
            side_coords = coords[1]
        else:
            main_coords = coords
            side_coords = None

        if self.nozzle_head is not None and self.arm_line is not None:
            self.nozzle_head.center = (main_coords[0], main_coords[1])
            if color:
                self.nozzle_head.set_facecolor(color)
            
            self.arm_line.set_data([self.pivot_pos[0], main_coords[0]], [self.pivot_pos[1], main_coords[1]])
            self.nozzle_head.set_visible(True)
            self.arm_line.set_visible(True)
            
        if side_coords is not None and self.side_nozzle_head is not None and self.side_arm_line is not None:
            self.side_nozzle_head.center = (side_coords[0], side_coords[1])
            if color:
                self.side_nozzle_head.set_facecolor(color)
            
            # 找到分岔點
            branch_dist = self.side_arm_branch_dist if self.side_arm_branch_dist is not None else (self.arm_length - 80.0)
            angle = math.atan2(main_coords[1] - self.pivot_pos[1], main_coords[0] - self.pivot_pos[0])
            bx = self.pivot_pos[0] + branch_dist * math.cos(angle)
            by = self.pivot_pos[1] + branch_dist * math.sin(angle)
            
            self.side_arm_line.set_data([bx, side_coords[0]], [by, side_coords[1]])
            self.side_nozzle_head.set_visible(True)
            self.side_arm_line.set_visible(True)
        elif self.side_nozzle_head is not None:
            self.side_nozzle_head.set_visible(False)
            if self.side_arm_line:
                self.side_arm_line.set_visible(False)

    def go_home(self):
        """讓手臂回到 Home 點並變灰"""
        angle = math.atan2(self.home_pos[1] - self.pivot_pos[1], self.home_pos[0] - self.pivot_pos[0])
        coords = self.angle_to_coords(angle)
        self.update_artists(coords, color='gray')

    def get_artists(self):
        """回傳所有的繪圖物件列表，供 Blitting 動畫使用"""
        artists = []
        if self.arm_line: artists.append(self.arm_line)
        if self.nozzle_head: artists.append(self.nozzle_head)
        if self.side_arm_line: artists.append(self.side_arm_line)
        if self.side_nozzle_head: artists.append(self.side_nozzle_head)
        return artists
