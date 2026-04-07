import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import UnivariateSpline
import optuna
import threading
import os
import math

# Import your configurations and simulator
from simulation_config_def import PARAMETER_DEFINITIONS, get_default_config
from etchingamount_generator import EtchingAmountGenerator
from PRE_generator import PREGenerator
from charging_generator import ChargingGenerator

class AutoTunerGUI:
    def __init__(self, root, main_app=None):
        self.root = root
        self.main_app = main_app
        self.root.title("Advanced Simulator")
        self.root.geometry("1400x800")
        
        self.base_config = get_default_config()
        self.current_recipe = None
        self.stop_tuning_flags = {"Etching": False, "PRE": False, "Charging": False}
        self.is_tuning = {"Etching": False, "PRE": False, "Charging": False}
        
        # State for each tab
        self.tabs_state = {
            "Etching": {
                "exp_radius": None, "exp_values": None, "spline_target": None, "target_radius_range": np.linspace(0, 150, 151),
                "param_vars": {}, "plot_title": "Etching Amount Comparison", "ylabel": "Etching Amount"
            },
            "PRE": {
                "target_mode": tk.StringVar(value="count"), # "count" or "pre"
                "target_count": tk.StringVar(value="50"), 
                "target_pre": tk.StringVar(value="99.5"), 
                "total_incoming": 10000, "param_vars": {}, "plot_title": "Defect Map Comparison", "ylabel": "Y (mm)"
            },
            "Charging": {
                "exp_radius": None, "exp_values": None, "spline_target": None, "target_radius_range": np.linspace(0, 150, 151),
                "param_vars": {}, "plot_title": "Surface Potential Comparison", "ylabel": "Potential (Volts)"
            }
        }
        
        self._create_widgets()

    def _create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)
        
        # 1. Etching Tab
        self.frame_etching = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_etching, text="Etching Amount")
        self._build_tab(self.frame_etching, "Etching")
        
        # 2. PRE Tab
        self.frame_pre = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_pre, text="Particle Removal")
        self._build_pre_tab(self.frame_pre)
        
        # 3. Charging Tab
        self.frame_charging = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_charging, text="Charging")
        self._build_tab(self.frame_charging, "Charging")

    def _build_tab(self, parent_frame, tab_name):
        paned = ttk.PanedWindow(parent_frame, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True)

        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)

        # 1. Experimental Data Loading
        frame_load = ttk.LabelFrame(left_frame, text=f"1. Load Experimental Data", padding=10)
        frame_load.pack(fill="x", padx=10, pady=5)
        
        lbl_csv = ttk.Label(frame_load, text="Exp CSV: Not Loaded", foreground="gray")
        lbl_csv.grid(row=0, column=0, padx=5, sticky="w")
        btn_csv = ttk.Button(frame_load, text="Load Exp CSV (.csv)", command=lambda: self.load_csv(tab_name, lbl_csv))
        btn_csv.grid(row=0, column=1, padx=5, pady=2)
        
        lbl_recipe_status = ttk.Label(frame_load, text="Recipe Status: Auto-Fetched on Run", foreground="blue")
        lbl_recipe_status.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # 2. Parameter Tuning Settings
        frame_params = ttk.LabelFrame(left_frame, text=f"2. Select Tuning Parameters", padding=10)
        frame_params.pack(fill="both", expand=True, padx=10, pady=5)
        
        params_container = ttk.Frame(frame_params)
        params_container.pack(fill="both", expand=True)

        canvas = tk.Canvas(params_container)
        v_scrollbar = ttk.Scrollbar(params_container, orient="vertical", command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(params_container, orient="horizontal", command=canvas.xview)
        
        scroll_frame = ttk.Frame(canvas)
        
        # When scroll_frame changes size, update the scroll region
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # When canvas changes size, resize the scroll_frame to fit
        def on_canvas_configure(e):
            canvas.itemconfig(frame_id, width=e.width)
            
        frame_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.bind("<Configure>", on_canvas_configure)
        
        # Enable Mouse Wheel Scrolling
        def _on_mousewheel(event):
            # For Windows/macOS
            if event.num == 5 or event.delta < 0:
                canvas.yview_scroll(1, "units")
            elif event.num == 4 or event.delta > 0:
                canvas.yview_scroll(-1, "units")
                
        canvas.bind("<MouseWheel>", _on_mousewheel)
        # For Linux
        canvas.bind("<Button-4>", _on_mousewheel)
        canvas.bind("<Button-5>", _on_mousewheel)
        
        # Ensure scroll events are caught when hovering over inner widgets
        scroll_frame.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        scroll_frame.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        scroll_frame.bind("<Enter>", lambda e: canvas.bind_all("<Button-4>", _on_mousewheel), add="+")
        scroll_frame.bind("<Leave>", lambda e: canvas.unbind_all("<Button-4>"), add="+")
        scroll_frame.bind("<Enter>", lambda e: canvas.bind_all("<Button-5>", _on_mousewheel), add="+")
        scroll_frame.bind("<Leave>", lambda e: canvas.unbind_all("<Button-5>"), add="+")

        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Pack scrollbars and canvas properly
        h_scrollbar.pack(side="bottom", fill="x")
        v_scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        headers = ["Enable", "Parameter Name", "Default", "Min Bound", "Max Bound", "Initial Guess"]
        for col, text in enumerate(headers):
            ttk.Label(scroll_frame, text=text, font=("Arial", 9, "bold")).grid(row=0, column=col, padx=2, pady=2)

        row = 1
        filter_keyword = tab_name if tab_name != "Charging" else "Charging"
        
        tunable_params = []
        fixed_params = []

        for category, params in PARAMETER_DEFINITIONS.items():
            if filter_keyword in category or (tab_name == "Charging" and "Fluid" in category):
                for key, info in params.items():
                    if info[5]: # is_tunable
                        tunable_params.append((key, info))
                    else:
                        fixed_params.append((key, info))

        # Render Tunable Group
        if tunable_params:
            ttk.Label(scroll_frame, text="--- Tunable Parameters ---", font=("Arial", 9, "bold"), foreground="blue").grid(row=row, column=0, columnspan=6, pady=(10, 5), sticky="w")
            row += 1
            for key, info in tunable_params:
                label_name, default_val, var_type, limit_range, description, is_tunable = info
                var_enabled = tk.BooleanVar(value=False)
                ttk.Checkbutton(scroll_frame, variable=var_enabled).grid(row=row, column=0, pady=(2, 0))
                
                lbl_param = ttk.Label(scroll_frame, text=label_name)
                lbl_param.grid(row=row, column=1, sticky="w", pady=(2, 0))
                ttk.Label(scroll_frame, text=str(default_val)).grid(row=row, column=2, pady=(2, 0))
                
                min_val, max_val = limit_range
                ent_min = ttk.Entry(scroll_frame, width=8); ent_min.insert(0, str(min_val))
                ent_max = ttk.Entry(scroll_frame, width=8); ent_max.insert(0, str(max_val))
                ent_guess = ttk.Entry(scroll_frame, width=8); ent_guess.insert(0, str(default_val))
                
                ent_min.grid(row=row, column=3, padx=2, pady=(2, 0)); ent_max.grid(row=row, column=4, padx=2, pady=(2, 0)); ent_guess.grid(row=row, column=5, padx=2, pady=(2, 0))
                self.tabs_state[tab_name]["param_vars"][key] = {'enabled': var_enabled, 'min': ent_min, 'max': ent_max, 'guess': ent_guess}
                
                lbl_desc = ttk.Label(scroll_frame, text=description, font=("Arial", 9), foreground="gray", wraplength=450)
                lbl_desc.grid(row=row+1, column=1, columnspan=5, sticky="w", pady=(0, 2))
                row += 2

        # Render Fixed Group
        if fixed_params:
            ttk.Label(scroll_frame, text="--- Fixed (Physical) Parameters ---", font=("Arial", 9, "bold"), foreground="brown").grid(row=row, column=0, columnspan=6, pady=(10, 5), sticky="w")
            row += 1
            for key, info in fixed_params:
                label_name, default_val, var_type, limit_range, description, is_tunable = info
                # Fixed parameters are not enabled for tuning
                var_enabled = tk.BooleanVar(value=False)
                
                ttk.Label(scroll_frame, text="Fixed", font=("Arial", 8, "italic"), foreground="gray").grid(row=row, column=0, pady=(2, 0))
                
                lbl_param = ttk.Label(scroll_frame, text=label_name)
                lbl_param.grid(row=row, column=1, sticky="w", pady=(2, 0))
                ttk.Label(scroll_frame, text=str(default_val)).grid(row=row, column=2, pady=(2, 0))
                
                # Still show entries but maybe emphasize they are initial settings
                ent_min = ttk.Entry(scroll_frame, width=8, state="disabled"); ent_min.insert(0, "N/A")
                ent_max = ttk.Entry(scroll_frame, width=8, state="disabled"); ent_max.insert(0, "N/A")
                ent_guess = ttk.Entry(scroll_frame, width=8); ent_guess.insert(0, str(default_val))
                
                ent_min.grid(row=row, column=3, padx=2, pady=(2, 0)); ent_max.grid(row=row, column=4, padx=2, pady=(2, 0)); ent_guess.grid(row=row, column=5, padx=2, pady=(2, 0))
                self.tabs_state[tab_name]["param_vars"][key] = {'enabled': var_enabled, 'min': ent_min, 'max': ent_max, 'guess': ent_guess}
                
                lbl_desc = ttk.Label(scroll_frame, text=description, font=("Arial", 9), foreground="gray", wraplength=450)
                lbl_desc.grid(row=row+1, column=1, columnspan=5, sticky="w", pady=(0, 2))
                row += 2

        # 3. Execution Control
        frame_run = ttk.Frame(left_frame, padding=10)
        frame_run.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(frame_run, text="Tuning Trials:").pack(side="left", padx=5)
        ent_trials = ttk.Entry(frame_run, width=8); ent_trials.insert(0, "100"); ent_trials.pack(side="left", padx=5)
        self.tabs_state[tab_name]["ent_trials"] = ent_trials
        
        btn_run = ttk.Button(frame_run, text="🚀 Start Auto Tuning", command=lambda: self.toggle_tuning(tab_name), state="disabled")
        btn_run.pack(side="right", padx=5)
        self.tabs_state[tab_name]["btn_run"] = btn_run
        
        progress_frame = ttk.Frame(left_frame, padding=5)
        progress_frame.pack(fill="x", padx=10)
        
        lbl_progress = ttk.Label(progress_frame, text="Progress: 0 / 0", font=("Arial", 10))
        lbl_progress.pack(side="left", padx=5)
        self.tabs_state[tab_name]["lbl_progress"] = lbl_progress
        
        lbl_status = ttk.Label(progress_frame, text="Ready", font=("Arial", 10), foreground="blue")
        lbl_status.pack(side="right", padx=5)
        self.tabs_state[tab_name]["lbl_status"] = lbl_status

        btn_export = ttk.Button(left_frame, text="📥 Export Simulation Result", command=lambda: self.export_simulation_result(tab_name))
        btn_export.pack(fill="x", padx=15, pady=10)

        # Right Plot Area
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)
        
        fig = Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title(self.tabs_state[tab_name]["plot_title"])
        ax.set_xlabel("Radius (mm)")
        ax.set_ylabel(self.tabs_state[tab_name]["ylabel"])
        ax.grid(True, linestyle='--', alpha=0.6)
        self.tabs_state[tab_name]["ax"] = ax
        self.tabs_state[tab_name]["fig"] = fig
        
        canvas_plot = FigureCanvasTkAgg(fig, master=right_frame)
        canvas_plot.get_tk_widget().pack(fill="both", expand=True)
        self.tabs_state[tab_name]["canvas"] = canvas_plot

    def _build_pre_tab(self, parent_frame):
        tab_name = "PRE"
        paned = ttk.PanedWindow(parent_frame, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True)

        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)

        # 1. PRE Target
        frame_target = ttk.LabelFrame(left_frame, text="1. Set Tuning Target", padding=10)
        frame_target.pack(fill="x", padx=10, pady=5)
        
        target_mode_var = self.tabs_state[tab_name]["target_mode"]
        
        # Radio button for Target Count
        rb_count = ttk.Radiobutton(frame_target, text="Target Remaining Defects:", variable=target_mode_var, value="count")
        rb_count.grid(row=0, column=0, sticky="w", pady=2)
        ent_count = ttk.Entry(frame_target, textvariable=self.tabs_state[tab_name]["target_count"], width=10)
        ent_count.grid(row=0, column=1, padx=5, pady=2)
        
        # Radio button for Target PRE (%)
        rb_pre = ttk.Radiobutton(frame_target, text="Target PRE (%):", variable=target_mode_var, value="pre")
        rb_pre.grid(row=1, column=0, sticky="w", pady=2)
        ent_pre = ttk.Entry(frame_target, textvariable=self.tabs_state[tab_name]["target_pre"], width=10)
        ent_pre.grid(row=1, column=1, padx=5, pady=2)
        
        lbl_recipe_status = ttk.Label(frame_target, text="Recipe Status: Auto-Fetched on Run", foreground="blue")
        lbl_recipe_status.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        def _on_target_mode_change(*args):
            try:
                # Try to get dynamically from UI input
                total_in = float(self.tabs_state[tab_name]["param_vars"]["PRE_DEFECT_COUNT"]['guess'].get())
            except:
                total_in = 10000.0

            mode = target_mode_var.get()
            if mode == "count":
                ent_count.config(state="normal")
                ent_pre.config(state="disabled")
                try:
                    count = float(self.tabs_state[tab_name]["target_count"].get())
                    pre = ((total_in - count) / total_in) * 100
                    self.tabs_state[tab_name]["target_pre"].set(f"{pre:.2f}")
                except ValueError: pass
            else:
                ent_pre.config(state="normal")
                ent_count.config(state="disabled")
                try:
                    pre = float(self.tabs_state[tab_name]["target_pre"].get())
                    count = total_in * (1 - pre/100)
                    self.tabs_state[tab_name]["target_count"].set(f"{int(count)}")
                except ValueError: pass

        target_mode_var.trace_add("write", _on_target_mode_change)
        
        # Trigger initial state
        _on_target_mode_change()

        # 2. Parameter Settings
        frame_params = ttk.LabelFrame(left_frame, text="2. Select Tuning Parameters", padding=10)
        frame_params.pack(fill="both", expand=True, padx=10, pady=5)
        
        params_container = ttk.Frame(frame_params)
        params_container.pack(fill="both", expand=True)

        canvas = tk.Canvas(params_container)
        v_scrollbar = ttk.Scrollbar(params_container, orient="vertical", command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(params_container, orient="horizontal", command=canvas.xview)
        
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        def on_canvas_configure_pre(e):
            canvas.itemconfig(frame_id_pre, width=e.width)
            
        frame_id_pre = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.bind("<Configure>", on_canvas_configure_pre)
        
        # Enable Mouse Wheel Scrolling
        def _on_mousewheel_pre(event):
            # For Windows/macOS
            if event.num == 5 or event.delta < 0:
                canvas.yview_scroll(1, "units")
            elif event.num == 4 or event.delta > 0:
                canvas.yview_scroll(-1, "units")
                
        canvas.bind("<MouseWheel>", _on_mousewheel_pre)
        # For Linux
        canvas.bind("<Button-4>", _on_mousewheel_pre)
        canvas.bind("<Button-5>", _on_mousewheel_pre)

        scroll_frame.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel_pre))
        scroll_frame.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        scroll_frame.bind("<Enter>", lambda e: canvas.bind_all("<Button-4>", _on_mousewheel_pre), add="+")
        scroll_frame.bind("<Leave>", lambda e: canvas.unbind_all("<Button-4>"), add="+")
        scroll_frame.bind("<Enter>", lambda e: canvas.bind_all("<Button-5>", _on_mousewheel_pre), add="+")
        scroll_frame.bind("<Leave>", lambda e: canvas.unbind_all("<Button-5>"), add="+")

        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        h_scrollbar.pack(side="bottom", fill="x")
        v_scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        headers = ["Enable", "Parameter Name", "Default", "Min Bound", "Max Bound", "Initial Guess"]
        for col, text in enumerate(headers):
            ttk.Label(scroll_frame, text=text, font=("Arial", 9, "bold")).grid(row=0, column=col, padx=2, pady=2)

        row = 1
        tunable_params = []
        fixed_params = []

        for category, params in PARAMETER_DEFINITIONS.items():
            if "Particle Removal" in category:
                for key, info in params.items():
                    if info[5]: # is_tunable
                        tunable_params.append((key, info))
                    else:
                        fixed_params.append((key, info))

        # Render Tunable Group
        if tunable_params:
            ttk.Label(scroll_frame, text="--- Tunable Parameters ---", font=("Arial", 9, "bold"), foreground="blue").grid(row=row, column=0, columnspan=6, pady=(10, 5), sticky="w")
            row += 1
            for key, info in tunable_params:
                label_name, default_val, var_type, limit_range, description, is_tunable = info
                var_enabled = tk.BooleanVar(value=False)
                ttk.Checkbutton(scroll_frame, variable=var_enabled).grid(row=row, column=0, pady=(2, 0))
                
                lbl_param = ttk.Label(scroll_frame, text=label_name)
                lbl_param.grid(row=row, column=1, sticky="w", pady=(2, 0))
                ttk.Label(scroll_frame, text=str(default_val)).grid(row=row, column=2, pady=(2, 0))
                
                min_val, max_val = limit_range
                ent_min = ttk.Entry(scroll_frame, width=8); ent_min.insert(0, str(min_val))
                ent_max = ttk.Entry(scroll_frame, width=8); ent_max.insert(0, str(max_val))
                ent_guess = ttk.Entry(scroll_frame, width=8); ent_guess.insert(0, str(default_val))
                
                ent_min.grid(row=row, column=3, padx=2, pady=(2, 0)); ent_max.grid(row=row, column=4, padx=2, pady=(2, 0)); ent_guess.grid(row=row, column=5, padx=2, pady=(2, 0))
                self.tabs_state[tab_name]["param_vars"][key] = {'enabled': var_enabled, 'min': ent_min, 'max': ent_max, 'guess': ent_guess}
                
                lbl_desc = ttk.Label(scroll_frame, text=description, font=("Arial", 9), foreground="gray", wraplength=450)
                lbl_desc.grid(row=row+1, column=1, columnspan=5, sticky="w", pady=(0, 2))
                row += 2

        # Render Fixed Group
        if fixed_params:
            ttk.Label(scroll_frame, text="--- Fixed (Physical) Parameters ---", font=("Arial", 9, "bold"), foreground="brown").grid(row=row, column=0, columnspan=6, pady=(10, 5), sticky="w")
            row += 1
            for key, info in fixed_params:
                label_name, default_val, var_type, limit_range, description, is_tunable = info
                var_enabled = tk.BooleanVar(value=False)
                
                ttk.Label(scroll_frame, text="Fixed", font=("Arial", 8, "italic"), foreground="gray").grid(row=row, column=0, pady=(2, 0))
                
                lbl_param = ttk.Label(scroll_frame, text=label_name)
                lbl_param.grid(row=row, column=1, sticky="w", pady=(2, 0))
                ttk.Label(scroll_frame, text=str(default_val)).grid(row=row, column=2, pady=(2, 0))
                
                ent_min = ttk.Entry(scroll_frame, width=8, state="disabled"); ent_min.insert(0, "N/A")
                ent_max = ttk.Entry(scroll_frame, width=8, state="disabled"); ent_max.insert(0, "N/A")
                ent_guess = ttk.Entry(scroll_frame, width=8); ent_guess.insert(0, str(default_val))
                
                ent_min.grid(row=row, column=3, padx=2, pady=(2, 0)); ent_max.grid(row=row, column=4, padx=2, pady=(2, 0)); ent_guess.grid(row=row, column=5, padx=2, pady=(2, 0))
                self.tabs_state[tab_name]["param_vars"][key] = {'enabled': var_enabled, 'min': ent_min, 'max': ent_max, 'guess': ent_guess}
                
                lbl_desc = ttk.Label(scroll_frame, text=description, font=("Arial", 9), foreground="gray", wraplength=450)
                lbl_desc.grid(row=row+1, column=1, columnspan=5, sticky="w", pady=(0, 2))
                row += 2

        # 3. Execution Control
        frame_run = ttk.Frame(left_frame, padding=10)
        frame_run.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(frame_run, text="Tuning Trials:").pack(side="left", padx=5)
        ent_trials = ttk.Entry(frame_run, width=8); ent_trials.insert(0, "100"); ent_trials.pack(side="left", padx=5)
        self.tabs_state[tab_name]["ent_trials"] = ent_trials
        
        btn_run = ttk.Button(frame_run, text="🚀 Start Auto Tuning", command=lambda: self.toggle_tuning(tab_name))
        btn_run.pack(side="right", padx=5)
        self.tabs_state[tab_name]["btn_run"] = btn_run
        
        progress_frame = ttk.Frame(left_frame, padding=5)
        progress_frame.pack(fill="x", padx=10)
        
        lbl_progress = ttk.Label(progress_frame, text="Progress: 0 / 0", font=("Arial", 10))
        lbl_progress.pack(side="left", padx=5)
        self.tabs_state[tab_name]["lbl_progress"] = lbl_progress
        
        lbl_status = ttk.Label(progress_frame, text="Ready", font=("Arial", 10), foreground="blue")
        lbl_status.pack(side="right", padx=5)
        self.tabs_state[tab_name]["lbl_status"] = lbl_status

        btn_export = ttk.Button(left_frame, text="📥 Export Simulation Result", command=lambda: self.export_simulation_result(tab_name))
        btn_export.pack(fill="x", padx=15, pady=10)

        # Right Plot Area
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)
        
        fig = Figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_title(self.tabs_state[tab_name]["plot_title"])
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        wafer = patches.Circle((0, 0), 150, color='#007bff', fill=False, lw=2, alpha=0.5)
        ax.add_artist(wafer)
        ax.set_xlim(-160, 160)
        ax.set_ylim(-160, 160)
        ax.grid(True, linestyle=':', alpha=0.3)
        self.tabs_state[tab_name]["ax"] = ax
        self.tabs_state[tab_name]["fig"] = fig
        
        canvas_plot = FigureCanvasTkAgg(fig, master=right_frame)
        canvas_plot.get_tk_widget().pack(fill="both", expand=True)
        self.tabs_state[tab_name]["canvas"] = canvas_plot

    def load_csv(self, tab_name, label_widget):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path: return
        try:
            # Using encoding='utf-8-sig' to handle files saved with BOM and avoid cp950 decode errors
            data = np.loadtxt(path, delimiter=",", skiprows=1, encoding='utf-8-sig')
            data = data[data[:, 0].argsort()]
            self.tabs_state[tab_name]["exp_radius"] = data[:, 0]
            self.tabs_state[tab_name]["exp_values"] = data[:, 1]
            self.tabs_state[tab_name]["spline_target"] = UnivariateSpline(data[:, 0], data[:, 1], k=1.0, s=0.0)
            label_widget.config(text=f"Exp CSV: {os.path.basename(path)}", foreground="black")
            
            # Enable the Start Auto Tuning button once experimental data is loaded successfully
            if "btn_run" in self.tabs_state[tab_name]:
                self.tabs_state[tab_name]["btn_run"].config(state="normal")
        except Exception as e: messagebox.showerror("Error", f"Failed to load CSV: {e}")

    def fetch_latest_data(self):
        if self.main_app:
            self.current_recipe = self.main_app.parse_and_prepare_recipe()
            self.base_config = self.main_app.get_current_config()
            return True if self.current_recipe else False
        return False

    def toggle_tuning(self, tab_name):
        if not self.is_tuning[tab_name]:
            # Start Tuning
            if not self.fetch_latest_data():
                messagebox.showwarning("Warning", "Failed to fetch recipe.")
                return

            if tab_name != "PRE" and self.tabs_state[tab_name]["exp_radius"] is None:
                messagebox.showwarning("Warning", f"Please load experimental CSV for {tab_name} first!")
                return
                
            active = {k: v for k, v in self.tabs_state[tab_name]["param_vars"].items() if v['enabled'].get()}
            if not active:
                messagebox.showwarning("Warning", "Please select at least one parameter to tune.")
                return
            
            try:
                trials = int(self.tabs_state[tab_name]["ent_trials"].get())
            except ValueError:
                messagebox.showwarning("Warning", "Please enter a valid integer for trials.")
                return
                
            self.is_tuning[tab_name] = True
            self.stop_tuning_flags[tab_name] = False
            
            self.tabs_state[tab_name]["lbl_progress"].config(text=f"Progress: 0 / {trials}")
            self.tabs_state[tab_name]["btn_run"].config(text="🛑 Stop Auto Tuning")
            self.tabs_state[tab_name]["lbl_status"].config(text="Optimization in progress...", foreground="red")
            
            threading.Thread(target=self.run_optimization, args=(tab_name, active, trials), daemon=True).start()
        else:
            # Stop Tuning
            self.stop_tuning_flags[tab_name] = True
            self.tabs_state[tab_name]["btn_run"].config(text="Stopping...", state="disabled")
            self.tabs_state[tab_name]["lbl_status"].config(text="Waiting for current trial to finish...", foreground="orange")

    def run_optimization(self, tab_name, active_params, trials):
        search_space = {k: (float(v['min'].get()), float(v['max'].get())) for k, v in active_params.items()}
        initial_guess = {k: float(v['guess'].get()) for k, v in active_params.items()}
        
        all_ui_guesses = {}
        for k, v in self.tabs_state[tab_name]["param_vars"].items():
            try: all_ui_guesses[k] = float(v['guess'].get())
            except ValueError: pass

        if tab_name != "PRE":
            v_target_smooth = self.tabs_state[tab_name]["spline_target"](self.tabs_state[tab_name]["target_radius_range"])

        if tab_name == "Etching":
            generator = EtchingAmountGenerator(self.main_app)
            precomputed_indices = generator.get_radial_indices_mask((300, 300))
        elif tab_name == "PRE":
            generator = PREGenerator(self.main_app)
            mode = self.tabs_state["PRE"]["target_mode"].get()
            if mode == "count":
                try:
                    target_val = float(self.tabs_state["PRE"]["target_count"].get())
                except:
                    target_val = 50.0
            else:
                try:
                    target_val = float(self.tabs_state["PRE"]["target_pre"].get())
                except:
                    target_val = 99.5
        elif tab_name == "Charging":
            generator = ChargingGenerator(self.main_app)

        def objective(trial):
            if not self.root.winfo_exists() or self.stop_tuning_flags[tab_name]:
                raise optuna.exceptions.TrialPruned()

            config = self.base_config.copy()
            config.update(all_ui_guesses)
            for key, (low, high) in search_space.items():
                config[key] = trial.suggest_float(key, low, high)
            
            try:
                if tab_name == "Etching":
                    etch_matrix, _ = generator.run_fast_simulation(self.current_recipe, config)
                    sim_radial = generator.calculate_radial_average(etch_matrix, precomputed_indices=precomputed_indices)
                    sim_matched = np.interp(self.tabs_state[tab_name]["target_radius_range"], np.arange(len(sim_radial)), sim_radial)
                    mse = np.mean((sim_matched - v_target_smooth)**2)
                    if (trial.number + 1) % 2 == 0 or (trial.number + 1) == trials:
                        self.root.after(0, self.update_plot_radial, tab_name, sim_radial, v_target_smooth, trial.number)
                
                elif tab_name == "PRE":
                    _, final_defects, _ = generator.run_fast_simulation(self.current_recipe, config)
                    sim_count = len(final_defects)
                    
                    if mode == "count":
                        mse = (sim_count - target_val)**2
                        display_target = target_val
                    else:
                        total_in = config.get("PRE_DEFECT_COUNT", 10000.0)
                        sim_pre = ((total_in - sim_count) / total_in) * 100
                        mse = (sim_pre - target_val)**2
                        display_target = target_val
                        
                    if (trial.number + 1) % 2 == 0 or (trial.number + 1) == trials:
                        total_in_for_plot = config.get("PRE_DEFECT_COUNT", 10000.0)
                        self.root.after(0, self.update_plot_scatter, tab_name, final_defects, trial.number, sim_count, display_target, mode, total_in_for_plot)
                
                elif tab_name == "Charging":
                    sim_radial, _ = generator.run_fast_simulation(self.current_recipe, config)
                    sim_matched = np.interp(self.tabs_state[tab_name]["target_radius_range"], np.arange(len(sim_radial)), sim_radial)
                    mse = np.mean((sim_matched - v_target_smooth)**2)
                    if (trial.number + 1) % 2 == 0 or (trial.number + 1) == trials:
                        self.root.after(0, self.update_plot_radial, tab_name, sim_radial, v_target_smooth, trial.number)

                if self.root.winfo_exists():
                    try:
                        self.root.after(0, lambda: self.tabs_state[tab_name]["lbl_progress"].config(text=f"Progress: {trial.number + 1} / {trials}"))
                    except tk.TclError:
                        pass
                return mse
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e: 
                print(f"Trial Error: {e}")
                return float('inf')

        def stop_check_callback(study, trial):
            if not self.root.winfo_exists() or self.stop_tuning_flags[tab_name]:
                study.stop()

        study = optuna.create_study(direction='minimize')
        study.enqueue_trial(initial_guess)
        
        try:
            study.optimize(objective, n_trials=trials, callbacks=[stop_check_callback])
        except Exception as e:
            print(f"Optimization stopped or encountered error: {e}")

        if not self.root.winfo_exists():
            return

        best_config = self.base_config.copy()
        best_config.update(all_ui_guesses)
        if len(study.trials) > 0 and study.best_trial is not None:
            best_config.update(study.best_params)
            best_value = study.best_value
            best_params = study.best_params
        else:
            best_value = float('inf')
            best_params = {}
        
        if tab_name == "Etching":
            best_matrix, _ = generator.run_fast_simulation(self.current_recipe, best_config)
            best_result = generator.calculate_radial_average(best_matrix, precomputed_indices=precomputed_indices)
            target = v_target_smooth
        elif tab_name == "PRE":
            _, best_result, _ = generator.run_fast_simulation(self.current_recipe, best_config)
            target = target_val
        elif tab_name == "Charging":
            best_result, _ = generator.run_fast_simulation(self.current_recipe, best_config)
            target = v_target_smooth

        was_stopped = self.stop_tuning_flags[tab_name]
        if self.root.winfo_exists():
            self.root.after(0, self.finish_tuning, tab_name, best_params, best_value, best_result, target, was_stopped)

    def update_plot_radial(self, tab_name, sim_radial, v_target_smooth, trial_num=None):
        if not self.root.winfo_exists():
            return
        try:
            canvas = self.tabs_state[tab_name]["canvas"]
            if not canvas.get_tk_widget().winfo_exists():
                return
                
            ax = self.tabs_state[tab_name]["ax"]
            ax.clear()
            
            ax.scatter(self.tabs_state[tab_name]["exp_radius"], self.tabs_state[tab_name]["exp_values"], color='red', label='Exp Data', s=30, alpha=0.6, zorder=5)
            ax.plot(self.tabs_state[tab_name]["target_radius_range"], v_target_smooth, color='orange', linestyle='--', label='Spline Target', zorder=2)
            ax.plot(np.arange(len(sim_radial)), sim_radial, color='blue', linewidth=2, label='Current Sim', zorder=3)
            
            title = self.tabs_state[tab_name]["plot_title"] + (f" (Trial: {trial_num + 1})" if trial_num is not None else "")
            ax.set_title(title)
            ax.set_xlabel("Radius (mm)")
            ax.set_ylabel(self.tabs_state[tab_name]["ylabel"])
            ax.set_xlim(-5, 155)
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # 使用 draw_idle 替代 draw 以提升執行緒安全性，避免視窗關閉時發生例外
            canvas.draw_idle()
        except tk.TclError:
            pass

    def update_plot_scatter(self, tab_name, defects, trial_num=None, sim_count=0, target_val=0, mode="count", total_in=10000.0):
        if not self.root.winfo_exists():
            return
        try:
            canvas = self.tabs_state[tab_name]["canvas"]
            if not canvas.get_tk_widget().winfo_exists():
                return
                
            ax = self.tabs_state[tab_name]["ax"]
            ax.clear()
            
            ax.set_aspect('equal')
            wafer = patches.Circle((0, 0), 150, color='#007bff', fill=False, lw=2, alpha=0.5)
            ax.add_artist(wafer)
            
            if len(defects) > 0:
                sizes = np.clip(defects[:, 2] * 0.3, 1, 50)
                ax.scatter(defects[:, 0], defects[:, 1], s=sizes, c='red', alpha=0.6, edgecolors='none', label='Remaining Defects')
                
            if mode == "count":
                title = f"Defect Map (Sim: {sim_count} | Target: {int(target_val)})"
            else:
                sim_pre = ((total_in - sim_count) / total_in) * 100
                title = f"Defect Map (Sim PRE: {sim_pre:.2f}% | Target: {target_val}%)"
                
            if trial_num is not None: title += f" [Trial: {trial_num + 1}]"
            
            ax.set_title(title)
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_xlim(-160, 160)
            ax.set_ylim(-160, 160)
            ax.grid(True, linestyle=':', alpha=0.3)
            if len(defects) > 0: ax.legend(loc='upper right')
            
            # 使用 draw_idle 替代 draw 以提升執行緒安全性
            canvas.draw_idle()
        except tk.TclError:
            pass

    def export_simulation_result(self, tab_name):
        if not self.main_app:
            messagebox.showerror("Error", "Main application reference not found.")
            return

        # 收集當前 AutoTuner 中的參數覆蓋值
        custom_config = self.main_app.get_current_config()
        for key, vars in self.tabs_state[tab_name]["param_vars"].items():
            try:
                val = float(vars['guess'].get())
                custom_config[key] = val
            except ValueError:
                continue

        # 根據不同的 tab 呼叫對應的導出函式
        if tab_name == "Etching":
            self.main_app.export_etching_amount(custom_config=custom_config)
        elif tab_name == "PRE":
            self.main_app.export_pre_efficiency(custom_config=custom_config)
        elif tab_name == "Charging":
            self.main_app.export_charging_simulation(custom_config=custom_config)

    def finish_tuning(self, tab_name, best_params, best_mse, best_result, target, was_stopped=False):
        if not self.root.winfo_exists():
            return
        self.is_tuning[tab_name] = False
        self.tabs_state[tab_name]["btn_run"].config(text="🚀 Start Auto Tuning", state="normal")
        
        trials = int(self.tabs_state[tab_name]["ent_trials"].get())
        
        if tab_name == "PRE":
            final_count = len(best_result)
            try:
                total_in = float(self.tabs_state[tab_name]["param_vars"]["PRE_DEFECT_COUNT"]['guess'].get())
            except:
                total_in = 10000.0
            pre_percent = ((total_in - final_count) / total_in) * 100
            
        if was_stopped:
            self.tabs_state[tab_name]["lbl_status"].config(text=f"Stopped. Best MSE: {best_mse:.6f}" if tab_name != "PRE" else f"Stopped. PRE: {pre_percent:.2f}%", foreground="orange")
        elif tab_name == "PRE":
            self.tabs_state[tab_name]["lbl_status"].config(text=f"Finished! PRE: {pre_percent:.2f}% (Rem: {final_count})", foreground="green")
        else:
            self.tabs_state[tab_name]["lbl_status"].config(text=f"Finished! MSE: {best_mse:.6f}", foreground="green")
            
        self.tabs_state[tab_name]["btn_run"].config(state="normal")
        
        for k, v in best_params.items():
            if k in self.tabs_state[tab_name]["param_vars"]:
                guess_entry = self.tabs_state[tab_name]["param_vars"][k]['guess']
                guess_entry.delete(0, tk.END)
                formatted_v = f"{v:.8e}" if abs(v) < 0.001 else f"{v:.4f}"
                guess_entry.insert(0, formatted_v)

        if tab_name == "PRE":
            mode = self.tabs_state["PRE"]["target_mode"].get()
            try:
                total_in_val = float(self.tabs_state[tab_name]["param_vars"]["PRE_DEFECT_COUNT"]['guess'].get())
            except:
                total_in_val = 10000.0
            self.update_plot_scatter(tab_name, best_result, None, len(best_result), target, mode, total_in_val)
            msg = f"Optimization {'stopped' if was_stopped else 'finished'}!\nAchieved PRE: {pre_percent:.2f}% (Rem: {final_count})\n\nOptimal parameters:\n"
        else:
            self.update_plot_radial(tab_name, best_result, target)
            msg = f"Optimization {'stopped' if was_stopped else 'finished'}! MSE: {best_mse:.6f}\n\nOptimal parameters for {tab_name}:\n"

        for k, v in best_params.items():
            msg += f"{k}: {v:.6e}\n"
            
        title = "Tuning Stopped" if was_stopped else "Tuning Success"
        # Print results to terminal instead of showing message box
        print("\n" + "="*40)
        print(f"[{title}] - {tab_name}")
        print("-"*40)
        print(msg.strip())
        print("="*40 + "\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoTunerGUI(root)
    root.mainloop()
