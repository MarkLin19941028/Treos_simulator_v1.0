import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import UnivariateSpline
import optuna
import threading
import os

# Import your configurations and simulator
from simulation_config_def import PARAMETER_DEFINITIONS, get_default_config
from etchingamount_generator import EtchingAmountGenerator

class AutoTunerGUI:
    def __init__(self, root, main_app=None):
        self.root = root
        self.main_app = main_app
        self.root.title("Etching Physics Auto Tuner (Recipe-Aware)")
        self.root.geometry("1200x800")
        
        # Data Storage
        self.exp_radius = None
        self.exp_values = None
        self.spline_target = None
        self.current_recipe = None
        self.target_radius_range = np.linspace(0, 150, 151)
        
        self.base_config = get_default_config()
        self.param_vars = {} 
        
        self._create_widgets()

    def _create_widgets(self):
        # Use PanedWindow to separate left control and right plot
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill="both", expand=True)

        left_frame = ttk.Frame(self.paned)
        self.paned.add(left_frame, weight=1)

        # --- 1. Experimental Data Loading ---
        frame_load = ttk.LabelFrame(left_frame, text="1. Load Experimental Data (Recipe Auto-Fetched)", padding=10)
        frame_load.pack(fill="x", padx=10, pady=5)
        
        # CSV Load
        self.lbl_csv = ttk.Label(frame_load, text="Exp CSV: Not Loaded", foreground="gray")
        self.lbl_csv.grid(row=0, column=0, padx=5, sticky="w")
        btn_csv = ttk.Button(frame_load, text="Load Exp CSV (.csv)", command=self.load_csv)
        btn_csv.grid(row=0, column=1, padx=5, pady=2)
        
        self.lbl_recipe_status = ttk.Label(frame_load, text="Recipe Status: Auto-Fetched on Run", foreground="blue")
        self.lbl_recipe_status.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # --- 2. Parameter Tuning Settings ---
        frame_params = ttk.LabelFrame(left_frame, text="2. Select Tuning Parameters (Etching Custom Params)", padding=10)
        frame_params.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Container for Canvas and Scrollbars
        params_container = ttk.Frame(frame_params)
        params_container.pack(fill="both", expand=True)

        canvas = tk.Canvas(params_container)
        v_scrollbar = ttk.Scrollbar(params_container, orient="vertical", command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(frame_params, orient="horizontal", command=canvas.xview)
        
        self.scroll_frame = ttk.Frame(canvas)
        self.scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")

        # Headers
        headers = ["Enable", "Parameter Name", "Default", "Min Bound", "Max Bound", "Initial Guess"]
        for col, text in enumerate(headers):
            ttk.Label(self.scroll_frame, text=text, font=("Arial", 9, "bold")).grid(row=0, column=col, padx=8, pady=5)

        # Auto-fetch Etching related parameters
        row = 1
        for category, params in PARAMETER_DEFINITIONS.items():
            if "Etching" in category:
                for key, info in params.items():
                    label_name, default_val = info[0], info[1]
                    
                    var_enabled = tk.BooleanVar(value=False)
                    ttk.Checkbutton(self.scroll_frame, variable=var_enabled).grid(row=row, column=0)
                    ttk.Label(self.scroll_frame, text=label_name).grid(row=row, column=1, sticky="w")
                    ttk.Label(self.scroll_frame, text=str(default_val)).grid(row=row, column=2)
                    
                    ent_min = ttk.Entry(self.scroll_frame, width=10); ent_min.insert(0, str(default_val * 0.01))
                    ent_max = ttk.Entry(self.scroll_frame, width=10); ent_max.insert(0, str(default_val * 100.0))
                    ent_guess = ttk.Entry(self.scroll_frame, width=10); ent_guess.insert(0, str(default_val))
                    
                    ent_min.grid(row=row, column=3, padx=5); ent_max.grid(row=row, column=4, padx=5); ent_guess.grid(row=row, column=5, padx=5)
                    
                    self.param_vars[key] = {'enabled': var_enabled, 'min': ent_min, 'max': ent_max, 'guess': ent_guess}
                    row += 1

        # --- 3. Execution Control Area ---
        frame_run = ttk.Frame(left_frame, padding=10)
        frame_run.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(frame_run, text="Tuning Trials:").pack(side="left", padx=5)
        self.ent_trials = ttk.Entry(frame_run, width=8); self.ent_trials.insert(0, "5"); self.ent_trials.pack(side="left", padx=5)
        
        self.btn_run = ttk.Button(frame_run, text="🚀 Start Auto Tuning", command=self.start_tuning_thread)
        self.btn_run.pack(side="right", padx=5)
        
        # Progress and Status
        self.progress_frame = ttk.Frame(left_frame, padding=5)
        self.progress_frame.pack(fill="x", padx=10)
        
        self.lbl_progress = ttk.Label(self.progress_frame, text="Progress: 0 / 0", font=("Arial", 10))
        self.lbl_progress.pack(side="left", padx=5)
        
        self.lbl_status = ttk.Label(self.progress_frame, text="Ready", font=("Arial", 10), foreground="blue")
        self.lbl_status.pack(side="right", padx=5)

        # Export to Recipe Button
        self.btn_export_to_recipe = ttk.Button(left_frame, text="📥 Export Parameters to Recipe Editor", command=self.export_to_recipe)
        self.btn_export_to_recipe.pack(fill="x", padx=15, pady=10)

        # --- Right plot area ---
        right_frame = ttk.Frame(self.paned)
        self.paned.add(right_frame, weight=2)
        
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Real-time Tuning Comparison")
        self.ax.set_xlabel("Radius (mm)")
        self.ax.set_ylabel("Etching Amount")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path: return
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            data = data[data[:, 0].argsort()]
            self.exp_radius, self.exp_values = data[:, 0], data[:, 1]
            
            # Use UnivariateSpline for smooth target
            self.spline_target = UnivariateSpline(self.exp_radius, self.exp_values, k=1.0, s=0.0)
            
            self.lbl_csv.config(text=f"Exp CSV: {os.path.basename(path)}", foreground="black")
        except Exception as e: messagebox.showerror("Error", f"Failed to load CSV: {e}")

    def fetch_latest_data(self):
        """Auto-fetch all content from Recipe Editor"""
        if self.main_app:
            self.current_recipe = self.main_app.parse_and_prepare_recipe()
            self.base_config = self.main_app.get_current_config()
            return True if self.current_recipe else False
        return False

    def start_tuning_thread(self):
        # Auto-fetch latest data before starting
        if not self.fetch_latest_data():
            messagebox.showwarning("Warning", "Failed to fetch recipe. Please check your Recipe Editor settings.")
            return

        if self.exp_radius is None:
            messagebox.showwarning("Warning", "Please load experimental CSV data first!")
            return
            
        active = {k: v for k, v in self.param_vars.items() if v['enabled'].get()}
        if not active:
            messagebox.showwarning("Warning", "Please select at least one parameter to tune.")
            return
        
        trials = int(self.ent_trials.get())
        self.lbl_progress.config(text=f"Progress: 0 / {trials}")
        self.btn_run.config(state="disabled")
        self.lbl_status.config(text="Optimization in progress...", foreground="red")
        threading.Thread(target=self.run_optimization, args=(active,), daemon=True).start()

    def run_optimization(self, active_params):
        trials = int(self.ent_trials.get())
        search_space = {k: (float(v['min'].get()), float(v['max'].get())) for k, v in active_params.items()}
        initial_guess = {k: float(v['guess'].get()) for k, v in active_params.items()}
        
        # Extract all UI guesses (both active and inactive)
        all_ui_guesses = {}
        for k, v in self.param_vars.items():
            try:
                all_ui_guesses[k] = float(v['guess'].get())
            except ValueError:
                pass
                
        v_target_smooth = self.spline_target(self.target_radius_range)
        
        # 預計算徑向平均所需的 Mask
        # 確保傳入 self.main_app 以便正確獲取 water_params
        generator = EtchingAmountGenerator(self.main_app)
        precomputed_indices = generator.get_radial_indices_mask((300, 300))

        def objective(trial):
            # 建立當前 trial 的配置，並強制同步全域參數
            config = self.base_config.copy()
            
            # Apply all UI guesses first, so inactive parameters match the UI
            config.update(all_ui_guesses)
            
            # Apply the trial suggestions for the active parameters
            for key, (low, high) in search_space.items():
                config[key] = trial.suggest_float(key, low, high)
            
            # 核心優化：如果 Global Scale 在 Tuning 中，確保其被正確應用
            try:
                # 執行快速模擬
                # 為了加速，我們可以手動調整 dynamic_report_fps (例如降低解析度)
                # 這裡暫時保持與 generator 一致以確保曲線相同，但優化 calculate_radial_average
                max_rpm = 0
                for proc in self.current_recipe['processes']:
                    spin = proc['spin_params']
                    c_max = spin['rpm'] if spin['mode'] == 'Simple' else max(spin['start_rpm'], spin['end_rpm'])
                    if c_max > max_rpm: max_rpm = c_max
                
                # 移除覆蓋的 FPS 以確保與 etchingamount_generator 的物理曲線一致
                # self.current_recipe['dynamic_report_fps'] = max(400, int(max_rpm * 2))
                
                # 手動執行模擬邏輯以使用 precomputed_indices
                etch_matrix, _ = generator.run_fast_simulation(self.current_recipe, config)
                sim_radial = generator.calculate_radial_average(etch_matrix, precomputed_indices=precomputed_indices)
                
                # sim_radial 索引 0 對應半徑 0mm，索引 150 對應 150mm
                sim_radius_axis = np.arange(len(sim_radial))
                sim_matched = np.interp(self.target_radius_range, sim_radius_axis, sim_radial)
                
                # Update progress and plot immediately after every trial
                self.root.after(0, lambda: self.lbl_progress.config(text=f"Progress: {trial.number + 1} / {trials}"))
                
                # Throttle plot updates to avoid freezing GUI and speed up optimization
                # Update plot only every 5 trials, or on the last trial
                if (trial.number + 1) % 2 == 0 or (trial.number + 1) == trials:
                    self.root.after(0, self.update_plot, sim_radial, v_target_smooth, trial.number)
                
                return np.mean((sim_matched - v_target_smooth)**2)
            except Exception as e: 
                print(f"Trial Error: {e}")
                return float('inf')

        study = optuna.create_study(direction='minimize')
        study.enqueue_trial(initial_guess)
        study.optimize(objective, n_trials=trials)

        best_config = self.base_config.copy()
        best_config.update(all_ui_guesses)
        best_config.update(study.best_params)
        
        best_etch_matrix, _ = generator.run_fast_simulation(self.current_recipe, best_config)
        best_sim_radial = generator.calculate_radial_average(best_etch_matrix, precomputed_indices=precomputed_indices)
        
        self.root.after(0, self.finish_tuning, study.best_params, study.best_value, best_sim_radial, v_target_smooth)

    def update_plot(self, sim_radial, v_target_smooth, trial_num=None):
        """Update the right-side real-time plot"""
        self.ax.clear()
        self.ax.scatter(self.exp_radius, self.exp_values, color='red', label='Exp Data', s=30, alpha=0.6, zorder=5)
        self.ax.plot(self.target_radius_range, v_target_smooth, color='orange', linestyle='--', label='Spline Target', zorder=2)
        
        # 確保 X 軸正確映射到半徑 (mm)
        sim_radius_axis = np.arange(len(sim_radial))
        self.ax.plot(sim_radius_axis, sim_radial, color='blue', linewidth=2, label='Current Sim', zorder=3)
        
        title = "Tuning Comparison"
        if trial_num is not None:
            title += f" (Trial: {trial_num + 1})"
        self.ax.set_title(title)
        self.ax.set_xlabel("Radius (mm)")
        self.ax.set_ylabel("Etching Amount (A.U.)")
        self.ax.set_xlim(-5, 155)
        self.ax.legend(loc='upper left')
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.canvas.draw()

    def export_to_recipe(self):
        """Export parameters back to Recipe Editor"""
        if not self.main_app:
            messagebox.showerror("Error", "Main application reference not found.")
            return
            
        exported_count = 0
        for key, vars in self.param_vars.items():
            try:
                val = float(vars['guess'].get())
                if key in self.main_app.config_vars:
                    self.main_app.config_vars[key].set(str(val))
                    exported_count += 1
            except ValueError:
                continue
                
        if exported_count > 0:
            messagebox.showinfo("Export Successful", f"Successfully exported {exported_count} parameters to the Recipe Editor (Physics & System tab).")
        else:
            messagebox.showwarning("Export Warning", "No matching parameters were found to export.")

    def finish_tuning(self, best_params, best_mse, best_sim_radial, v_target_smooth):
        trials = int(self.ent_trials.get())
        self.lbl_progress.config(text=f"Progress: {trials} / {trials}")
        self.lbl_status.config(text=f"Finished! MSE: {best_mse:.6f}", foreground="green")
        self.btn_run.config(state="normal")
        
        # Fill best parameters back to "Initial Guess" fields
        for k, v in best_params.items():
            if k in self.param_vars:
                guess_entry = self.param_vars[k]['guess']
                guess_entry.delete(0, tk.END)
                if v < 0.001:
                    formatted_v = f"{v:.8f}"
                else:
                    formatted_v = f"{v:.4f}"
                guess_entry.insert(0, formatted_v)

        # Update final plot
        self.update_plot(best_sim_radial, v_target_smooth)
        self.ax.set_title(f"Final Comparison (MSE: {best_mse:.6f})")
        self.canvas.draw()

        # Show notification
        msg = "Optimal parameters have been updated to the 'Initial Guess' fields:\n\n"
        for k, v in best_params.items():
            msg += f"{k}: {v:.6e}\n"
        messagebox.showinfo("Tuning Success", msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoTunerGUI(root)
    root.mainloop()
