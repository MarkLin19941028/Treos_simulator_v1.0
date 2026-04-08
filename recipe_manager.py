import tkinter as tk
from tkinter import messagebox, filedialog
import os
from simulation_config_def import get_default_config

class RecipeManager:
    """
    專門處理 SimulationApp 的 Recipe 匯入、匯出和檔案解析邏輯。
    """
    def __init__(self, app_instance):
        # 儲存 SimulationApp 的實例，以便存取 UI 變數和方法
        self.app = app_instance

    def export_recipe(self):
        """
        將當前的全域參數和 Process 參數匯出為 .csv Recipe 檔案。
        """
        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Recipe Files", "*.csv"), ("All Files", "*.*")],
                title="Export Recipe As..."
            )
            if not filepath:
                return

            import csv

            with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                
                writer.writerow(["[GLOBAL]"])
                writer.writerow(["spin_direction", self.app.spin_dir.get()])
                writer.writerow(["water_setting_mode", self.app.water_setting_mode_var.get()])
                writer.writerow(["viscosity", self.app.viscosity_var.get()])
                writer.writerow(["surface_tension", self.app.surface_tension_var.get()])
                writer.writerow(["evaporation_rate", self.app.evaporation_rate_var.get()])

                writer.writerow([])
                writer.writerow(["[PHYSICS_SYSTEM]"])
                for key, var in self.app.config_vars.items():
                    writer.writerow([key, var.get()])

                writer.writerow([])

                for i, proc_data in enumerate(self.app.processes_data):
                    writer.writerow([f"[PROCESS_{i+1}]"])
                    arm_str = proc_data['arm_var'].get()
                    writer.writerow(["dispense_arm", arm_str])
                    writer.writerow(["flow_rate", proc_data['flow_rate_var'].get()])
                    if arm_str == 'Arm 2':
                        writer.writerow(["flow_rate_2", proc_data['flow_rate_var_2'].get()])
                    writer.writerow(["total_duration", proc_data['duration_var'].get()])
                    writer.writerow(["start_from_center", proc_data['start_from_center_var'].get()])
                    
                    spin_mode = proc_data['spin_mode_var'].get()
                    writer.writerow(["spin_mode", spin_mode])
                    
                    if spin_mode == 'Simple':
                        writer.writerow(["simple_rpm", proc_data['simple_rpm_var'].get()])
                    else:
                        writer.writerow(["start_rpm", proc_data['start_rpm_var'].get()])
                        writer.writerow(["end_rpm", proc_data['end_rpm_var'].get()])
                        
                    if proc_data['arm_var'].get() != 'None':
                        num_steps = proc_data['num_steps_var'].get()
                        writer.writerow(["steps", num_steps])
                        for j, step_entry in enumerate(proc_data['step_entries']):
                            writer.writerow([f"step_{j+1}_pos", step_entry['pos'].get()])
                            writer.writerow([f"step_{j+1}_speed", step_entry['speed'].get()])
                    writer.writerow([])

                # --- 匯出 Advanced Function (AutoTuner) 的 Initial Guess 參數 ---
                writer.writerow(["[TUNING_PARAMETERS]"])
                tuning_params = {}
                # 優先從開啟的視窗中獲取最新值
                if self.app.autotuner_instance and self.app.autotuner_instance.root.winfo_exists():
                    tuning_params = self.app.autotuner_instance.get_all_tuning_guesses()
                    # 同時更新 app 的暫存
                    self.app.imported_tuning_params.update(tuning_params)
                else:
                    # 視窗未開啟，使用 app 暫存的參數 (可能是之前匯入的)
                    tuning_params = self.app.imported_tuning_params
                
                # 如果 tuning_params 為空，則從預設配置中獲取 (確保匯出時總是有數值)
                if not tuning_params:
                    from simulation_config_def import PARAMETER_DEFINITIONS
                    for category, params in PARAMETER_DEFINITIONS.items():
                        if category in ["Etching Amount", "Particle Removal", "Charging Simulation"]:
                            for key, info in params.items():
                                tuning_params[key] = info[1] # info[1] 是預設值

                for key, val in tuning_params.items():
                    writer.writerow([key, val])
            
            # messagebox.showinfo("Success", "Recipe exported successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export recipe: {e}")

    def _read_file_with_fallback(self, filepath):
        """
        嘗試以不同的編碼方式讀取檔案，解決 Excel 匯出造成的編碼問題
        """
        encodings = ['utf-8-sig', 'utf-8', 'cp1252', 'big5']
        for enc in encodings:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    return f.read(), enc
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"Failed to decode file {filepath} with any of {encodings}")

    def import_recipe(self):
        """
        從 .csv (或舊版 .txt) 檔案匯入 Recipe，並更新 SimulationApp 的 UI 變數。
        """
        try:
            filepath = filedialog.askopenfilename(
                filetypes=[("Recipe Files", "*.csv *.txt"), ("CSV Files", "*.csv"), ("Text Files", "*.txt"), ("All Files", "*.*")],
                title="Import Recipe"
            )
            if not filepath:
                return

            global_params, imported_processes, current_process_dict = {}, [], None
            
            imported_tuning_params = {}
            if filepath.endswith(".csv"):
                import csv
                content, enc = self._read_file_with_fallback(filepath)
                
                import io
                f = io.StringIO(content)
                reader = csv.reader(f)
                current_section = None
                for raw_row in reader:
                    if not raw_row: continue
                    # 過濾掉空白欄位 (Excel 可能會塞入很多空白)
                    row = [col for col in raw_row if col.strip() != ""]
                    if not row: continue
                    
                    if row[0].startswith('[') and row[0].endswith(']'):
                        current_section = row[0][1:-1]
                        if current_section.startswith('PROCESS_'):
                            current_process_dict = {'steps_data': {}}
                            imported_processes.append(current_process_dict)
                        else:
                            current_process_dict = None
                        continue
                    
                    if len(row) >= 2:
                            key, value = row[0].strip(), row[1].strip()
                            if current_section == "TUNING_PARAMETERS":
                                imported_tuning_params[key] = value
                            elif current_process_dict is None:
                                global_params[key] = value
                                if key.startswith('flow_rate_arm_'):
                                    arm_id = int(key.split('_')[-1])
                                    if arm_id in self.app.arm_flow_rate_vars:
                                        self.app.arm_flow_rate_vars[arm_id].set(value)
                            else:
                                if key.startswith('step_'):
                                    parts = key.split('_')
                                    step_num, step_key = int(parts[1]), parts[2]
                                    if step_num not in current_process_dict['steps_data']: 
                                        current_process_dict['steps_data'][step_num] = {}
                                    current_process_dict['steps_data'][step_num][step_key] = value
                                else:
                                    current_process_dict[key] = value
            else:
                content, enc = self._read_file_with_fallback(filepath)
                lines = content.splitlines()
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    if line.startswith('[') and line.endswith(']'):
                        current_section = line[1:-1]
                        if current_section.startswith('PROCESS_'):
                            current_process_dict = {'steps_data': {}}
                            imported_processes.append(current_process_dict)
                        else:
                            current_process_dict = None
                        continue
                    
                    if '=' not in line: continue
                    key, value = [x.strip() for x in line.split('=', 1)]
                    
                    if current_section == "TUNING_PARAMETERS":
                        imported_tuning_params[key] = value
                    elif current_process_dict is None:
                        global_params[key] = value
                        if key.startswith('flow_rate_arm_'):
                            arm_id = int(key.split('_')[-1])
                            if arm_id in self.app.arm_flow_rate_vars:
                                self.app.arm_flow_rate_vars[arm_id].set(value)
                    else:
                        if key.startswith('step_'):
                            parts = key.split('_')
                            step_num, step_key = int(parts[1]), parts[2]
                            if step_num not in current_process_dict['steps_data']: 
                                current_process_dict['steps_data'][step_num] = {}
                            current_process_dict['steps_data'][step_num][step_key] = value
                        else:
                            current_process_dict[key] = value

            # 更新 Global UI 變數
            if 'spin_direction' in global_params: self.app.spin_dir.set(global_params['spin_direction'])
            if 'water_setting_mode' in global_params: self.app.water_setting_mode_var.set(global_params['water_setting_mode'])
            if 'viscosity' in global_params: self.app.viscosity_var.set(global_params['viscosity'])
            if 'surface_tension' in global_params: self.app.surface_tension_var.set(global_params['surface_tension'])
            if 'evaporation_rate' in global_params: self.app.evaporation_rate_var.set(global_params['evaporation_rate'])

            # 更新 Physics & System 變數 (處理舊版本相容性：若 Recipe 缺項則套用預設值)
            default_config = get_default_config()
            for key, var in self.app.config_vars.items():
                if key in global_params:
                    var.set(global_params[key])
                else:
                    var.set(str(default_config.get(key, '')))

            num_proc = len(imported_processes)
            if num_proc == 0: raise ValueError("No processes found.")
            
            # 準備重新繪製 Process Widgets
            for proc_data in imported_processes:
                proc_data['steps'] = int(proc_data.get('steps', 3))
            
            self.app.is_importing = True # 設置標記防止重複初始化
            self.app.num_processes.set(num_proc)
            self.app.recreate_process_widgets(imported_data=imported_processes) # 重新創建 UI
            self.app.is_importing = False
            
            # 填充 Process UI 變數
            for i, proc_data in enumerate(imported_processes):
                gui_proc = self.app.processes_data[i]
                arm_str = proc_data.get('dispense_arm', 'Arm 1')
                gui_proc['arm_var'].set(arm_str)
                
                # 處理 Flow Rate (新版本優先，舊版本回退)
                if 'flow_rate' in proc_data:
                    gui_proc['flow_rate_var'].set(proc_data['flow_rate'])
                else:
                    # 舊版本：嘗試從全域變數獲取對應手臂的流量
                    try:
                        arm_id = int(arm_str.split(" ")[1]) if arm_str != "None" else 0
                        if arm_id in self.app.arm_flow_rate_vars:
                            gui_proc['flow_rate_var'].set(self.app.arm_flow_rate_vars[arm_id].get())
                        else:
                            gui_proc['flow_rate_var'].set('500')
                    except:
                        gui_proc['flow_rate_var'].set('500')

                # 處理 Arm 2 的第二個噴嘴 (Nozzle 3)
                if 'flow_rate_2' in proc_data:
                    gui_proc['flow_rate_var_2'].set(proc_data['flow_rate_2'])
                else:
                    gui_proc['flow_rate_var_2'].set('1500') # 預設值

                gui_proc['duration_var'].set(proc_data.get('total_duration', '10'))
                gui_proc['start_from_center_var'].set(proc_data.get('start_from_center', 'False').lower() == 'true')
                spin_mode = proc_data.get('spin_mode', 'Simple')
                gui_proc['spin_mode_var'].set(spin_mode)
                
                if spin_mode == 'Simple':
                    gui_proc['simple_rpm_var'].set(proc_data.get('simple_rpm', '200'))
                else:
                    gui_proc['start_rpm_var'].set(proc_data.get('start_rpm', '0'))
                    gui_proc['end_rpm_var'].set(proc_data.get('end_rpm', '200'))
                    
                if gui_proc['arm_var'].get() != 'None':
                    num_steps = proc_data['steps']
                    for j in range(num_steps):
                        if j < len(gui_proc['step_entries']):
                            step_num = j + 1
                            pos = proc_data['steps_data'].get(step_num, {}).get('pos', '0')
                            speed = proc_data['steps_data'].get(step_num, {}).get('speed', '0')
                            gui_proc['step_entries'][j]['pos'].set(pos)
                            gui_proc['step_entries'][j]['speed'].set(speed)
            
            # 觸發更新 Arm 狀態 (啟用/禁用 Steps 輸入欄位)
            for i in range(num_proc):
                self.app._on_arm_change(i)
                
            self.app._on_water_setting_mode_change() # 更新 Water Settings UI 顯示

            # --- 更新 Advanced Function (AutoTuner) 參數 ---
            if imported_tuning_params:
                self.app.imported_tuning_params.update(imported_tuning_params)
                # 若 AutoTuner 視窗已開啟，則立即更新 UI
                if self.app.autotuner_instance and self.app.autotuner_instance.root.winfo_exists():
                    self.app.autotuner_instance.set_tuning_guesses(imported_tuning_params)

            # 更新 UI 顯示的檔案名稱
            import os
            filename = os.path.basename(filepath)
            if hasattr(self.app, 'current_recipe_file_var'):
                self.app.current_recipe_file_var.set(f"Current Recipe: {filename}")

            # messagebox.showinfo("Success", "Recipe imported successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import recipe: {e}")
