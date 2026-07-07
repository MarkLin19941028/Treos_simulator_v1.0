import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import threading
import time
import csv

class BatchExportWindow:
    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app  # 引用主程式的 instance
        
        # 狀態控制變數
        self.recipe_paths = []
        self.output_dir = tk.StringVar(value="Not selected")
        self.is_running = False
        self.cancel_requested = False
        
        # 批次排隊處理索引
        self.current_index = 0
        
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
        ttk.Checkbutton(content_frame, text="Moving Pattern (.csv)", variable=self.chk_pattern).grid(row=1, column=0, sticky="w", padx=10, pady=5)
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
        """主執行緒安全的直接 Log 記錄"""
        self.txt_log.config(state="normal")
        self.txt_log.insert(tk.END, message + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state="disabled")

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
            # 驗證輸入條件
            if not self.recipe_paths:
                messagebox.showwarning("Warning", "Please select at least one recipe file.")
                return
            if self.output_dir.get() == "Not selected" or not os.path.exists(self.output_dir.get()):
                messagebox.showwarning("Warning", "Please select a valid output directory.")
                return
            if not (self.chk_report.get() or self.chk_video.get() or self.chk_pattern.get() or self.chk_heatmap.get()):
                messagebox.showwarning("Warning", "Please check at least one content item to export.")
                return
            
            # 緩存主前台勾選參數
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
            
            # 初始化進度條
            self.progress_bar["maximum"] = len(self.recipe_paths)
            self.progress_bar["value"] = 0
            self.current_index = 0
            
            # 核心修正：利用 Tkinter after 事件鏈，一條龍在主執行緒安全執行分析，絕不閃退
            self.window.after(100, self.process_next_recipe)
        else:
            if messagebox.askyesno("Confirm", "Are you sure you want to cancel the batch process?"):
                self.cancel_requested = True
                self.lbl_status.config(text="Status: Canceling...", foreground="orange")

    def lock_ui_inputs(self, lock=True):
        state = "disabled" if lock else "normal"
        self.btn_select_recipes.config(state=state)
        self.btn_select_dir.config(state=state)

    def process_next_recipe(self):
        """一條龍在 Tkinter Main Thread 內調用外部 Generator 的核心方法"""
        total_recipes = len(self.recipe_paths)
        
        # 檢查是否取消或結束
        if self.cancel_requested or self.current_index >= total_recipes:
            self.finish_batch_export()
            return
            
        path = self.recipe_paths[self.current_index]
        filename = os.path.basename(path)
        base_name, _ = os.path.splitext(filename)
        out_dir = self.target_dir
        
        self.lbl_status.config(text=f"Processing ({self.current_index + 1}/{total_recipes}): {filename}", foreground="blue")
        self.log(f"\n[Start] ({self.current_index + 1}/{total_recipes}) Running: {filename}")
        
        try:
            # 1. 載入配方並解析變數
            self.main_app.recipe_manager.import_recipe(path)
            parsed_recipe = self.main_app.parse_and_prepare_recipe()
            current_config = self.main_app.get_current_config()
            
            if not parsed_recipe:
                self.log(f"[Error] Failed to parse recipe structure for: {filename}")
                self.next_loop()
                return

            # 2. 建立前台進度條的本地對接存根 (此時因為在主執行緒，可以安全刷新)
            widgets = {
                'window': self.window,
                'bar': self.progress_bar,
                'label': self.lbl_status
            }

            # ---- 調用你現有的擴充 Generator 進行數據處理 ----
            
            # A. Simulation Report
            if self.run_report and not self.cancel_requested:
                self.log(f" > Running simulation report...")
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

            # B. Generate Video (安全調用你的 video_generator.py)
            if self.run_video and not self.cancel_requested:
                self.log(f" > Running video_generator.py...")
                vid_path = os.path.join(out_dir, f"{base_name}_Simulation_Video.mp4")
                self.main_app.video_generator._run_headless_video_generation(
                    parsed_recipe, vid_path, progress_widgets=widgets,
                    play_speed_multiplier=self.speed_multiplier, config=current_config
                )

            # C. Moving Pattern (安全調用你的 moving_pattern.py)
            if self.run_pattern and not self.cancel_requested:
                self.log(f" > Running moving_pattern.py...")
                pat_path = os.path.join(out_dir, f"{base_name}_Moving_Pattern.csv")
                if hasattr(self.main_app.moving_pattern_generator, 'generate_headless'):
                    self.main_app.moving_pattern_generator.generate_headless(parsed_recipe, pat_path)
                elif hasattr(self.main_app.moving_pattern_generator, 'export_nozzle_pattern'):
                    # 如果原先是透過手動選擇存檔，這處可能需要你確認外部檔案是否有接收指定路徑的參數
                    try:
                        self.main_app.moving_pattern_generator.export_nozzle_pattern(filepath=pat_path)
                    except TypeError:
                        self.export_nozzle_pattern_headless(parsed_recipe, pat_path)
                else:
                    self.export_nozzle_pattern_headless(parsed_recipe, pat_path)

            # D. Accumulation Heatmap (安全調用你的 accu_heatmap_generator.py)
            if self.run_heatmap and not self.cancel_requested:
                self.log(f" > Running accu_heatmap_generator.py...")
                hmp_path = os.path.join(out_dir, f"{base_name}_Accumulation_Heatmap.png")
                max_rpm = max([p['spin_params']['rpm'] if p['spin_params']['mode'] == 'Simple' else max(p['spin_params']['start_rpm'], p['spin_params']['end_rpm']) for p in parsed_recipe['processes']])
                parsed_recipe['dynamic_report_fps'] = max(800, int(max_rpm * 4))
                
                from accu_heatmap_generator import AccuHeatmapGenerator
                generator = AccuHeatmapGenerator(self.main_app)
                generator.generate(
                    recipe=parsed_recipe, 
                    filepath=hmp_path, 
                    config=current_config, 
                    progress_widgets=widgets
                )

            self.log(f"[Success] Completed export for: {filename}")
            
        except Exception as e:
            self.log(f"[Error Exception] Failed on {filename}: {str(e)}")
            
        self.next_loop()

    def next_loop(self):
        """遞增索引並排定下一個非同步任務"""
        self.current_index += 1
        self.progress_bar["value"] = self.current_index
        # 10 毫秒後在主執行緒調用下一個配方，讓 Tkinter 有時間刷新視窗畫面而不當機
        self.window.after(10, self.process_next_recipe)

    def finish_batch_export(self):
        """結束導出後的 UI 重置復原"""
        self.is_running = False
        self.lock_ui_inputs(False)
        self.btn_action.config(text="Start Batch Export")
        
        if self.cancel_requested:
            self.lbl_status.config(text="Status: Cancelled", foreground="red")
            self.log("\n[Terminated] Batch process was aborted by user.")
            messagebox.showwarning("Cancelled", "Batch export has been cancelled.")
        else:
            self.lbl_status.config(text="Status: Finished All", foreground="green")
            self.log("\n[Success] All tasks completed successfully.")
            messagebox.showinfo("Finished", f"Successfully processed all recipes!")

    def export_nozzle_pattern_headless(self, parsed_recipe, filepath):
        """Headless 移動路徑本地導出版 (降級備用備份)"""
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Process Step', 'Time Point (s)', 'Nozzle Position (%)', 'Arm ID'])
                for i, proc in enumerate(parsed_recipe['processes']):
                    arm_id = proc['arm_id']
                    if arm_id == 0: continue
                    total_t = proc['total_duration']
                    steps = proc['steps']
                    if not steps: continue
                    dt = total_t / len(steps)
                    for j, step in enumerate(steps):
                        writer.writerow([i + 1, f"{(j * dt):.2f}", step['pos'], arm_id])
        except Exception as e:
            self.log(f"   >> [Pattern Backup Save Error] {str(e)}")