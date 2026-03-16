import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import pchip_interpolate, PchipInterpolator
import optuna
import threading
import os
import json

# 匯入您的配置與模擬器
from simulation_config_def import PARAMETER_DEFINITIONS, get_default_config
from etchingamount_generator import EtchingAmountGenerator

class AutoTunerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Etching Physics Auto Tuner (Recipe-Aware)")
        self.root.geometry("900x700")
        
        # 數據存儲
        self.exp_radius = None
        self.exp_values = None
        self.spline_target = None
        self.current_recipe = None
        self.target_radius_range = np.linspace(0, 145, 146)
        
        self.base_config = get_default_config()
        self.param_vars = {} 
        
        self._create_widgets()

    def _create_widgets(self):
        # --- 1. 實驗數據與 Recipe 載入區 ---
        frame_load = ttk.LabelFrame(self.root, text="1. 載入實驗數據與 Recipe", padding=10)
        frame_load.pack(fill="x", padx=10, pady=5)
        
        # CSV 載入
        self.lbl_csv = ttk.Label(frame_load, text="實驗 CSV: 未載入", foreground="gray")
        self.lbl_csv.grid(row=0, column=0, padx=5, sticky="w")
        btn_csv = ttk.Button(frame_load, text="載入實驗 CSV (.csv)", command=self.load_csv)
        btn_csv.grid(row=0, column=1, padx=5, pady=2)
        
        # Recipe 載入
        self.lbl_recipe = ttk.Label(frame_load, text="製程 Recipe: 未載入", foreground="gray")
        self.lbl_recipe.grid(row=1, column=0, padx=5, sticky="w")
        btn_recipe = ttk.Button(frame_load, text="載入對應 Recipe (.txt)", command=self.load_recipe)
        btn_recipe.grid(row=1, column=1, padx=5, pady=2)

        # --- 2. 參數調教設定區 ---
        frame_params = ttk.LabelFrame(self.root, text="2. 選擇調教參數 (Etching Amount 自定義參數)", padding=10)
        frame_params.pack(fill="both", expand=True, padx=10, pady=5)
        
        canvas = tk.Canvas(frame_params)
        scrollbar = ttk.Scrollbar(frame_params, orient="vertical", command=canvas.yview)
        self.scroll_frame = ttk.Frame(canvas)

        self.scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 表頭
        headers = ["啟用", "參數名稱", "預設值", "搜尋下限", "搜尋上限", "起始猜測"]
        for col, text in enumerate(headers):
            ttk.Label(self.scroll_frame, text=text, font=("Arial", 9, "bold")).grid(row=0, column=col, padx=8, pady=5)

        # 自動抓取 Etching 相關參數
        row = 1
        for category, params in PARAMETER_DEFINITIONS.items():
            if "Etching" in category:
                for key, info in params.items():
                    label_name, default_val = info[0], info[1]
                    
                    var_enabled = tk.BooleanVar(value=False)
                    ttk.Checkbutton(self.scroll_frame, variable=var_enabled).grid(row=row, column=0)
                    ttk.Label(self.scroll_frame, text=label_name).grid(row=row, column=1, sticky="w")
                    ttk.Label(self.scroll_frame, text=str(default_val)).grid(row=row, column=2)
                    
                    ent_min = ttk.Entry(self.scroll_frame, width=10); ent_min.insert(0, str(default_val * 0.2))
                    ent_max = ttk.Entry(self.scroll_frame, width=10); ent_max.insert(0, str(default_val * 5.0))
                    ent_guess = ttk.Entry(self.scroll_frame, width=10); ent_guess.insert(0, str(default_val))
                    
                    ent_min.grid(row=row, column=3, padx=5); ent_max.grid(row=row, column=4, padx=5); ent_guess.grid(row=row, column=5, padx=5)
                    
                    self.param_vars[key] = {'enabled': var_enabled, 'min': ent_min, 'max': ent_max, 'guess': ent_guess}
                    row += 1

        # --- 3. 執行控制區 ---
        frame_run = ttk.Frame(self.root, padding=10)
        frame_run.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(frame_run, text="最佳化次數 (Trials):").pack(side="left", padx=5)
        self.ent_trials = ttk.Entry(frame_run, width=8); self.ent_trials.insert(0, "50"); self.ent_trials.pack(side="left", padx=5)
        
        self.btn_run = ttk.Button(frame_run, text="🚀 開始自動分析與調教", command=self.start_tuning_thread)
        self.btn_run.pack(side="right", padx=5)
        
        self.lbl_status = ttk.Label(self.root, text="準備就緒", font=("Arial", 10), foreground="blue")
        self.lbl_status.pack(pady=5)

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path: return
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            data = data[data[:, 0].argsort()]
            self.exp_radius, self.exp_values = data[:, 0], data[:, 1]
            
            # [修正點]：
            # 1. k=3 改為 k=2 (二次樣條) 或 k=1 (線性) 可以減少邊界震盪。
            # 2. 增加 s 權重，讓曲線不強制通過點，而是趨向平滑。
            self.spline_target = UnivariateSpline(self.exp_radius, self.exp_values, k=1.0, s=0.0)
            
            #self.spline_target = PchipInterpolator(self.exp_radius, self.exp_values)
            
            self.lbl_csv.config(text=f"實驗 CSV: {os.path.basename(path)}", foreground="black")
        except Exception as e: messagebox.showerror("錯誤", f"CSV 載入失敗: {e}")

    def load_recipe(self):
        path = filedialog.askopenfilename(filetypes=[("Recipe Files", "*.txt")])
        if not path: return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 解析 .txt Recipe 轉為模擬器格式
            imported_procs = []
            curr_proc = None
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'): continue
                if line.startswith('[') and line.endswith(']'):
                    section = line[1:-1]
                    if section.startswith('PROCESS_'):
                        curr_proc = {'steps_data': {}}
                        imported_procs.append(curr_proc)
                    else: curr_proc = None
                    continue
                if '=' not in line: continue
                key, val = [x.strip() for x in line.split('=', 1)]
                if curr_proc is not None:
                    if key.startswith('step_'):
                        p = key.split('_'); s_num, s_key = int(p[1]), p[2]
                        if s_num not in curr_proc['steps_data']: curr_proc['steps_data'][s_num] = {}
                        curr_proc['steps_data'][s_num][s_key] = val
                    else: curr_proc[key] = val

            # 格式化為 Simulation Engine 格式
            formatted = {'dynamic_report_fps': 30, 'processes': []}
            for p in imported_procs:
                arm_str = p.get('dispense_arm', 'Arm 1')
                try:
                    if 'Arm' in arm_str:
                        arm_id = int(arm_str.replace('Arm', '').strip())
                    else:
                        arm_id = 0
                except:
                    arm_id = 1

                p_dict = {
                    'dispense_arm': arm_str,
                    'arm_id': arm_id,
                    'flow_rate': float(p.get('flow_rate', 500)),
                    'total_duration': float(p.get('total_duration', 10)),
                    'spin_mode': p.get('spin_mode', 'Simple'),
                    'steps': []
                }
                
                # 處理 spin parameters 轉為 simulation_engine 的格式
                spin_params = {'mode': p_dict['spin_mode']}
                if p_dict['spin_mode'] == 'Simple': 
                    p_dict['simple_rpm'] = float(p.get('simple_rpm', 500))
                    spin_params['rpm'] = float(p.get('simple_rpm', 500))
                else:
                    p_dict['start_rpm'] = float(p.get('start_rpm', 0))
                    p_dict['end_rpm'] = float(p.get('end_rpm', 500))
                    spin_params['start_rpm'] = float(p.get('start_rpm', 0))
                    spin_params['end_rpm'] = float(p.get('end_rpm', 500))
                p_dict['spin_params'] = spin_params
                
                num_steps = int(p.get('steps', 0))
                for i in range(1, num_steps + 1):
                    p_dict['steps'].append({'pos': float(p['steps_data'].get(i, {}).get('pos', 0)), 
                                            'speed': float(p['steps_data'].get(i, {}).get('speed', 0))})
                formatted['processes'].append(p_dict)

            self.current_recipe = formatted
            self.lbl_recipe.config(text=f"製程 Recipe: {os.path.basename(path)}", foreground="black")
        except Exception as e: messagebox.showerror("錯誤", f"Recipe 載入失敗: {e}")

    def start_tuning_thread(self):
        if self.exp_radius is None or self.current_recipe is None:
            messagebox.showwarning("警告", "請載入實驗數據 CSV 以及對應的 Recipe .txt 檔案！")
            return
        active = {k: v for k, v in self.param_vars.items() if v['enabled'].get()}
        if not active:
            messagebox.showwarning("警告", "請至少勾選一個調教參數。")
            return
        
        self.btn_run.config(state="disabled")
        self.lbl_status.config(text="最佳化進行中...", foreground="red")
        threading.Thread(target=self.run_optimization, args=(active,), daemon=True).start()

    def run_optimization(self, active_params):
        trials = int(self.ent_trials.get())
        search_space = {k: (float(v['min'].get()), float(v['max'].get())) for k, v in active_params.items()}
        initial_guess = {k: float(v['guess'].get()) for k, v in active_params.items()}
        v_target_smooth = self.spline_target(self.target_radius_range)

        def objective(trial):
            config = self.base_config.copy()
            for key, (low, high) in search_space.items():
                config[key] = trial.suggest_float(key, low, high)
            
            generator = EtchingAmountGenerator(None)
            try:
                _, sim_radial = generator.run_fast_simulation(self.current_recipe, config)
                sim_matched = np.interp(self.target_radius_range, np.arange(len(sim_radial)), sim_radial)
                return np.mean((sim_matched - v_target_smooth)**2)
            except: return float('inf')

        study = optuna.create_study(direction='minimize')
        study.enqueue_trial(initial_guess)
        study.optimize(objective, n_trials=trials)

        best_config = self.base_config.copy()
        best_config.update(study.best_params)
        _, best_sim_radial = EtchingAmountGenerator(None).run_fast_simulation(self.current_recipe, best_config)
        
        self.root.after(0, self.finish_tuning, study.best_params, study.best_value, best_sim_radial, v_target_smooth)

    def finish_tuning(self, best_params, best_mse, best_sim_radial, v_target_smooth):
        self.lbl_status.config(text=f"完成！MSE: {best_mse:.6f}", foreground="green")
        self.btn_run.config(state="normal")
        
        # [核心修正]：將找到的最佳參數自動填回「起始猜測」欄位
        for k, v in best_params.items():
            if k in self.param_vars:
                # 取得該參數對應的 Entry 插件
                guess_entry = self.param_vars[k]['guess']
                
                # 先清空舊內容
                guess_entry.delete(0, tk.END)
                
                # 根據數值大小決定格式 (避免科學記號太長)
                if v < 0.001:
                    formatted_v = f"{v:.8f}"
                else:
                    formatted_v = f"{v:.4f}"
                
                # 插入新找到的最佳值
                guess_entry.insert(0, formatted_v)

        # 顯示訊息彈窗
        msg = "最佳參數已更新至『起始猜測』欄位：\n\n"
        for k, v in best_params.items():
            msg += f"{k}: {v:.6e}\n"
        messagebox.showinfo("調教成功", msg)

        # --- 繪圖 (保持原樣) ---
        plt.figure(figsize=(10, 6))
        plt.scatter(self.exp_radius, self.exp_values, color='red', label='Exp Data (Discrete)', zorder=5, s=50)
        plt.plot(self.target_radius_range, v_target_smooth, 'orange', linestyle='--', label='Spline Target', zorder=2)
        plt.plot(np.arange(len(best_sim_radial)), best_sim_radial, 'blue', linewidth=2, label='Optimized Sim')
        plt.title(f"Comparison (MSE: {best_mse:.6f})")
        plt.xlabel("Radius (mm)")
        plt.ylabel("Etching Amount")
        
        # [核心修正]：手動設定 X 軸範圍，讓它從 -5 開始
        # 這樣 0mm 的紅點就不會被 Y 軸擋住
        plt.xlim(-5, 155) 
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoTunerGUI(root)
    root.mainloop()