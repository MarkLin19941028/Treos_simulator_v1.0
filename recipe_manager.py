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
        將當前的全域參數和 Process 參數匯出為 .txt Recipe 檔案。
        """
        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Recipe Files", "*.txt"), ("All Files", "*.*")],
                title="Export Recipe As..."
            )
            if not filepath:
                return

            # 訪問 app 實例中的變數
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("[GLOBAL]\n")
                f.write(f"spin_direction = {self.app.spin_dir.get()}\n")

                f.write(f"water_setting_mode = {self.app.water_setting_mode_var.get()}\n")
                f.write(f"viscosity = {self.app.viscosity_var.get()}\n")
                f.write(f"surface_tension = {self.app.surface_tension_var.get()}\n")
                f.write(f"evaporation_rate = {self.app.evaporation_rate_var.get()}\n")

                # 匯出 Physics & System 參數
                f.write("\n[PHYSICS_SYSTEM]\n")
                for key, var in self.app.config_vars.items():
                    f.write(f"{key} = {var.get()}\n")

                f.write("\n")

                for i, proc_data in enumerate(self.app.processes_data):
                    f.write(f"[PROCESS_{i+1}]\n")
                    arm_str = proc_data['arm_var'].get()
                    f.write(f"dispense_arm = {arm_str}\n")
                    f.write(f"flow_rate = {proc_data['flow_rate_var'].get()}\n")
                    if arm_str == 'Arm 2':
                        f.write(f"flow_rate_2 = {proc_data['flow_rate_var_2'].get()}\n")
                    f.write(f"total_duration = {proc_data['duration_var'].get()}\n")
                    f.write(f"start_from_center = {proc_data['start_from_center_var'].get()}\n")
                    
                    spin_mode = proc_data['spin_mode_var'].get()
                    f.write(f"spin_mode = {spin_mode}\n")
                    
                    if spin_mode == 'Simple':
                        f.write(f"simple_rpm = {proc_data['simple_rpm_var'].get()}\n")
                    else:
                        f.write(f"start_rpm = {proc_data['start_rpm_var'].get()}\n")
                        f.write(f"end_rpm = {proc_data['end_rpm_var'].get()}\n")
                        
                    if proc_data['arm_var'].get() != 'None':
                        num_steps = proc_data['num_steps_var'].get()
                        f.write(f"steps = {num_steps}\n")
                        for j, step_entry in enumerate(proc_data['step_entries']):
                            f.write(f"step_{j+1}_pos = {step_entry['pos'].get()}\n")
                            f.write(f"step_{j+1}_speed = {step_entry['speed'].get()}\n")
                    f.write("\n")
            
            messagebox.showinfo("Success", "Recipe exported successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export recipe: {e}")

    def import_recipe(self):
        """
        從 .txt 檔案匯入 Recipe，並更新 SimulationApp 的 UI 變數。
        """
        try:
            filepath = filedialog.askopenfilename(
                filetypes=[("Recipe Files", "*.txt"), ("All Files", "*.*")],
                title="Import Recipe"
            )
            if not filepath:
                return

            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            global_params, imported_processes, current_process_dict = {}, [], None
            
            # 檔案內容解析邏輯
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'): continue
                if line.startswith('[') and line.endswith(']'):
                    section = line[1:-1]
                    if section.startswith('PROCESS_'):
                        current_process_dict = {'steps_data': {}}
                        imported_processes.append(current_process_dict)
                    else:
                        # 非 Process 區段（如 GLOBAL, PHYSICS_SYSTEM）皆視為全域
                        current_process_dict = None
                    continue
                
                if '=' not in line: continue
                key, value = [x.strip() for x in line.split('=', 1)]
                
                if current_process_dict is None:
                    global_params[key] = value
                    # 更新 Flow Rate 變數
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

            messagebox.showinfo("Success", "Recipe imported successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import recipe: {e}")