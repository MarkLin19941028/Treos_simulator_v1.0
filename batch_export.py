import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import threading
import time
import csv

# 建立完全執行緒安全的虛擬視窗 Mock 物件，阻斷底層引擎的 GUI 刷新
class HeadlessMockWindow:
    def update_idletasks(self): pass
    def update(self): pass

class BatchExportWindow:
    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app  # 引用主程式的 instance
        
        # 狀態控制變數
        self.recipe_paths = []
        self.output_dir = tk.StringVar(value="Not selected")
        self.is_running = False
        self.cancel_requested = False
        
        # 儲存背景運算需要的同步共享容器
        self.current_parsed_recipe = None
        self.current_config = None
        
        # 輸出的項目勾選變數
        self.chk_report = tk.BooleanVar(value=True)
        self.chk_video = tk.BooleanVar(value=False)
        self.chk_pattern = tk.BooleanVar(value=False)
        self.chk_heatmap = tk.BooleanVar(value=False)
        
        # 建立視窗
        self.window = tk.Toplevel(self.parent)
        self.window.title("Batch Export Console")
        self.window.geometry("550x450")
        self.window.transient(self.parent)
        self.window.grab_set()
        self.window.resizable(False, False)
        
        self.create_widgets()
        
    def create_widgets(self):
        # 1. Recipe 選擇區塊
        recipe_frame = ttk.LabelFrame(self.window, text="1. Select Recipes", padding=10)
        recipe_frame.pack(fill="x", padx=10, pady=5)
        
        self.btn_select_recipes = ttk.Button(recipe_frame, text="Browse Recipes (Multi)", command=self.browse_recipes)
        self.btn_select_recipes.pack(side="left", padx=5)
        
        self.lbl_recipe_count = ttk.Label(recipe_frame, text="No recipes selected", foreground="gray")
        self.lbl_recipe_count.pack(side="left", padx=10)
        
        # 2. 輸出資料夾選擇區塊
        dir_frame = ttk.LabelFrame(self.window, text="2. Select Output Directory", padding=10)
        dir_frame.pack(fill="x", padx=10, pady=5)
        
        self.btn_select_dir = ttk.Button(dir_frame, text="Browse Folder", command=self.browse_output_dir)
        self.btn_select_dir.pack(side="left", padx=5)
        
        lbl_dir = ttk.Label(dir_frame, textvariable=self.output_dir, foreground="blue", wraplength=400)
        lbl_dir.pack(side="left", padx=10)
        
        # 3. 輸出內容勾選區塊
        content_frame = ttk.LabelFrame(self.window, text="3. Select Export Contents", padding=10)
        content_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Checkbutton(content_frame, text="Simulation Report (.csv)", variable=self.chk_report).grid(row=0, column=0, sticky="w", padx=10, pady=5)
        ttk.Checkbutton(content_frame, text="Generate Video (.mp4)", variable=self.chk_video).grid(row=0, column=1, sticky="w", padx=10, pady=5)
        ttk.Checkbutton(content_frame, text="Moving Pattern (.csv, .png, .mp4)", variable=self.chk_pattern).grid(row=1, column=0, sticky="w", padx=10, pady=5)
        ttk.Checkbutton(content_frame, text="Accumulation Heatmap (.png)", variable=self.chk_heatmap).grid(row=1, column=1, sticky="w", padx=10, pady=5)
        
        # 4. 進度顯示與控制區塊
        control_frame = ttk.Frame(self.window, padding=10)
        control_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.btn_action = ttk.Button(control_frame, text="Start Batch Export", width=25, command=self.toggle_batch_process)
        self.btn_action.pack(pady=5)
        
        self.lbl_status = ttk.Label(control_frame, text="Status: Idle", font=("Segoe UI", 9, "bold"))
        self.lbl_status.pack(anchor="w", pady=(5, 0))
        
        self.progress_bar = ttk.Progressbar(control_frame, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill="x", pady=5)
        
        self.txt_log = tk.Text(control_frame, height=6, font=("Consolas", 9), state="disabled")
        self.txt_log.pack(fill="both", expand=True)

    def log(self, message):
        """執行緒安全的 Log 印出工具"""
        def safe_log():
            self.txt_log.config(state="normal")
            self.txt_log.insert(tk.END, message + "\n")
            self.txt_log.see(tk.END)
            self.txt_log.config(state="disabled")
        self.window.after(0, safe_log)

    def browse_recipes(self):
        files = filedialog.askopenfilenames(
            filetypes=[
                ("All Supported Recipes", "*.csv *.txt *.json"),
                ("CSV Recipe Files", "*.csv"),
                ("Text Recipe Files", "*.txt"),
                ("JSON Recipe Files", "*.json"),
                ("All Files", "*.*")
            ],
            title="Select One or Multiple Recipes"
        )
        if files:
            self.recipe_paths = list(files)
            self.lbl_recipe_count.config(text=f"{len(self.recipe_paths)} recipes loaded", foreground="black")
            self.log(f"[Loaded] Selected {len(self.recipe_paths)} recipe files.")

    def browse_output_dir(self):
        folder = filedialog.askdirectory(title="Select Destination Directory")
        if folder:
            self.output_dir.set(folder)
            self.log(f"[Target] Output directory set to: {folder}")

    def toggle_batch_process(self):
        if not self.is_running:
            if not self.recipe_paths:
                messagebox.showwarning("Warning", "Please select at least one recipe file.")
                return
            if self.output_dir.get() == "Not selected" or not os.path.exists(self.output_dir.get()):
                messagebox.showwarning("Warning", "Please select a valid output directory.")
                return
            if not (self.chk_report.get() or self.chk_video.get() or self.chk_pattern.get() or self.chk_heatmap.get()):
                messagebox.showwarning("Warning", "Please check at least one content item to export.")
                return
            
            # 提前在主執行緒將所需的全域 GUI 參數讀出
            try:
                if hasattr(self.main_app, 'speed_var') and self.main_app.speed_var:
                    self.speed_multiplier = float(self.main_app.speed_var.get().replace('x', ''))
                else:
                    self.speed_multiplier = 1.0
            except:
                self.speed_multiplier = 1.0
            
            self.run_report = self.chk_report.get()
            self.run_video = self.chk_video.get()
            self.run_pattern = self.chk_pattern.get()
            self.run_heatmap = self.chk_heatmap.get()
            self.target_dir = self.output_dir.get()
            
            # 開始執行
            self.is_running = True
            self.cancel_requested = False
            self.btn_action.config(text="Cancel")
            self.lock_ui_inputs(True)
            
            self.progress_bar["maximum"] = len(self.recipe_paths)
            self.progress_bar["value"] = 0
            
            self.worker_thread = threading.Thread(target=self.execute_batch_export, daemon=True)
            self.worker_thread.start()
        else:
            if messagebox.askyesno("Confirm", "Are you sure you want to cancel the batch process?"):
                self.cancel_requested = True
                self.lbl_status.config(text="Status: Canceling...", foreground="orange")
                self.log("[Interrupt] Cancellation requested by user. Terminating loop...")

    def lock_ui_inputs(self, lock=True):
        state = "disabled" if lock else "normal"
        self.btn_select_recipes.config(state=state)
        self.btn_select_dir.config(state=state)

    def execute_batch_export(self):
        total_recipes = len(self.recipe_paths)
        out_dir = self.target_dir
        
        for idx, path in enumerate(self.recipe_paths):
            if self.cancel_requested:
                break
                
            filename = os.path.basename(path)
            base_name, _ = os.path.splitext(filename)
            
            self.window.after(0, lambda f=filename, i=idx: self.lbl_status.config(
                text=f"Processing ({i+1}/{total_recipes}): {f}", foreground="blue"
            ))
            self.log(f"\n[Start] ({idx+1}/{total_recipes}) Processing: {filename}")
            
            try:
                # 1. 同步加載配方檔案到 UI 變數快照中 (交給主執行緒安全執行)[cite: 4]
                evt = threading.Event()
                def safe_gui_load_and_parse():
                    try:
                        self.main_app.recipe_manager.import_recipe(path)
                        self.current_parsed_recipe = self.main_app.parse_and_prepare_recipe()
                        self.current_config = self.main_app.get_current_config()
                    finally:
                        evt.set()
                
                self.window.after(0, safe_gui_load_and_parse)
                evt.wait()  # 阻塞背景，確保主執行緒快照完畢[cite: 4]
                
                parsed_recipe = self.current_parsed_recipe
                current_config = self.current_config
                
                if not parsed_recipe:
                    self.log(f"[Error] Failed to parse recipe variables for: {filename}")
                    continue
                
                mock_win = HeadlessMockWindow()
                dummy_widgets = {
                    'window': mock_win,
                    'bar': type('DummyBar', (object,), {'__setitem__': lambda s, k, v: None, 'get': lambda s, k: 0.0})(),
                    'label': type('DummyLabel', (object,), {'config': lambda s, **kw: None})()
                }

                # ---- 執行導出項目 ----
                
                # A. Simulation Report
                if self.run_report and not self.cancel_requested:
                    self.log(f" > Exporting Simulation Report (.csv)...")
                    max_rpm = max([p['spin_params']['rpm'] if p['spin_params']['mode'] == 'Simple' else max(p['spin_params']['start_rpm'], p['spin_params']['end_rpm']) for p in parsed_recipe['processes']])
                    parsed_recipe['dynamic_report_fps'] = max(800, int(max_rpm * 4))
                    
                    report_data, particle_data, _ = self.main_app._run_headless_simulation(parsed_recipe, progress_widgets=None)
                    
                    if report_data:
                        rep_path = os.path.join(out_dir, f"{base_name}_Simulation_Report.csv")
                        with open(rep_path, 'w', newline='', encoding='utf-8') as f:
                            w = csv.DictWriter(f, fieldnames=report_data[0].keys())
                            w.writeheader()
                            w.writerows(report_data)
                    if particle_data:
                        part_path = os.path.join(out_dir, f"{base_name}_Particle_Calculation.csv")
                        processed_part = []
                        for p in particle_data:
                            if p['time_on_wafer'] > 0:
                                processed_part.append({
                                    'Particle ID': p['id'],
                                    'Residence Time (s)': f"{p['time_on_wafer']:.4f}",
                                    'Path Length (mm)': f"{p['path_length']:.4f}",
                                    'Average Velocity (mm/s)': f"{(p['path_length'] / p['time_on_wafer']):.4f}"
                                })
                        with open(part_path, 'w', newline='', encoding='utf-8') as f:
                            w = csv.DictWriter(f, fieldnames=['Particle ID', 'Residence Time (s)', 'Path Length (mm)', 'Average Velocity (mm/s)'])
                            w.writeheader()
                            w.writerows(processed_part)

                # B. Generate Video
                if self.run_video and not self.cancel_requested:
                    self.log(f" > Calling video_generator.py (.mp4)...")
                    vid_path = os.path.join(out_dir, f"{base_name}_Simulation_Video.mp4")
                    
                    self.main_app.video_generator._run_headless_video_generation(
                        parsed_recipe, vid_path, progress_widgets=dummy_widgets,
                        play_speed_multiplier=self.speed_multiplier, config=current_config
                    )

                # C. Moving Pattern (核心修正：直接調用已解耦優化後的外部方法，100% 完整產出所有圖表)[cite: 4]
                if self.run_pattern and not self.cancel_requested:
                    self.log(f" > Calling moving_pattern.py (CSV, PNG graphs, MP4)...")
                    pat_path_base = os.path.join(out_dir, base_name)
                    
                    evt_pattern = threading.Event()
                    def safe_pattern_export():
                        try:
                            # 傳入快照後的純數據結構與指定基礎路徑，完美跨平台解耦！[cite: 4]
                            self.main_app.moving_pattern_generator.export_nozzle_pattern(
                                filepath=pat_path_base, 
                                parsed_recipe=parsed_recipe
                            )
                        except Exception as ex:
                            self.log(f"   >> [Pattern Export Failed] {str(ex)}")
                        finally:
                            evt_pattern.set()
                            
                    self.window.after(0, safe_pattern_export)
                    evt_pattern.wait()

                # D. Accumulation Heatmap
                if self.run_heatmap and not self.cancel_requested:
                    self.log(f" > Calling accu_heatmap_generator.py...")
                    hmp_path = os.path.join(out_dir, f"{base_name}_Accumulation_Heatmap.png")
                    
                    max_rpm = max([p['spin_params']['rpm'] if p['spin_params']['mode'] == 'Simple' else max(p['spin_params']['start_rpm'], p['spin_params']['end_rpm']) for p in parsed_recipe['processes']])
                    parsed_recipe['dynamic_report_fps'] = max(800, int(max_rpm * 4))
                    
                    from accu_heatmap_generator import AccuHeatmapGenerator
                    generator = AccuHeatmapGenerator(self.main_app)
                    generator.generate(
                        recipe=parsed_recipe, 
                        filepath=hmp_path, 
                        config=current_config, 
                        progress_widgets=dummy_widgets
                    )

                self.log(f"[Success] Completed export for: {filename}")
                
            except Exception as e:
                self.log(f"[Error] Failed on {filename}: {str(e)}")
            
            # 更新進度條
            self.window.after(0, lambda i=idx: self.progress_bar.configure(value=i + 1))

        # 批次結束後的 UI 復原
        def batch_finished_ui():
            self.is_running = False
            self.lock_ui_inputs(False)
            self.btn_action.config(text="Start Batch Export")
            if self.cancel_requested:
                self.lbl_status.config(text="Status: Cancelled", foreground="red")
                messagebox.showwarning("Cancelled", "Batch export was cancelled.")
            else:
                self.lbl_status.config(text="Status: Finished All", foreground="green")
                # messagebox.showinfo("Finished", f"Successfully processed {total_recipes} recipes!")
                
        self.window.after(0, batch_finished_ui)