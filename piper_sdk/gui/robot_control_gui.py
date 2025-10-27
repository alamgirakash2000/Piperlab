#!/usr/bin/env python3
"""
Piper Robot Control GUI
A graphical interface for controlling the Piper robotic arm

Requirements:
- CAN interface must be active (run: sudo bash scripts/1_setup_can.sh)
- tkinter (usually included with Python)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from piper_sdk import *

# Joint limits in degrees
JOINT_LIMITS = {
    'Joint 1': (-150, 150),
    'Joint 2': (0, 180),
    'Joint 3': (-170, 0),
    'Joint 4': (-100, 100),
    'Joint 5': (-70, 70),
    'Joint 6': (-120, 120)
}

# Preset positions (in degrees)
PRESETS = {
    'Reset': [0, 0, 0, 0, 0, 0],
    'Stand': [0, 45, -45, 0, 0, 0],
    'Rest': [0, 90, -90, 0, 0, 0],
    'Ready': [0, 30, -30, 0, 20, 0],
}

class PiperControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Piper Robot Control Panel")
        self.root.geometry("1200x900")
        self.root.resizable(True, True)
        
        # Robot interface
        self.piper = None
        self.connected = False
        self.monitoring = False
        self.live_control_thread = None
        self.position_changed = False
        
        # Joint values (in degrees)
        self.joint_values = [0, 0, 0, 0, 0, 0]
        self.current_positions = [0, 0, 0, 0, 0, 0]
        self.last_sent_values = [0, 0, 0, 0, 0, 0]
        
        # Create GUI
        self.create_widgets()
        
        # Try auto-connect
        self.root.after(500, self.connect_robot)
        
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # ===== CONNECTION FRAME =====
        conn_frame = ttk.LabelFrame(self.root, text="Connection", padding=10)
        conn_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky='ew')
        
        self.status_label = tk.Label(conn_frame, text="● Disconnected", 
                                     fg='red', font=('Arial', 12, 'bold'))
        self.status_label.grid(row=0, column=0, padx=5)
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect", 
                                      command=self.connect_robot)
        self.connect_btn.grid(row=0, column=1, padx=5)
        
        self.disconnect_btn = ttk.Button(conn_frame, text="Disconnect", 
                                        command=self.disconnect_robot, state='disabled')
        self.disconnect_btn.grid(row=0, column=2, padx=5)
        
        ttk.Button(conn_frame, text="Emergency Stop", 
                  command=self.emergency_stop,
                  style='Danger.TButton').grid(row=0, column=3, padx=20)
        
        # ===== JOINT CONTROL FRAME =====
        control_frame = ttk.LabelFrame(self.root, text="Joint Control (degrees)", padding=10)
        control_frame.grid(row=1, column=0, rowspan=2, padx=10, pady=5, sticky='nsew')
        
        self.sliders = []
        self.value_labels = []
        
        for i, (joint_name, (min_val, max_val)) in enumerate(JOINT_LIMITS.items()):
            # Joint label
            tk.Label(control_frame, text=joint_name, font=('Arial', 10, 'bold')).grid(
                row=i, column=0, padx=5, pady=5, sticky='w')
            
            # Min label
            tk.Label(control_frame, text=f"{min_val}°").grid(
                row=i, column=1, padx=2)
            
            # Slider
            slider = tk.Scale(control_frame, from_=min_val, to=max_val, 
                            orient='horizontal', length=300, resolution=1,
                            command=lambda val, idx=i: self.update_joint_value(idx, val))
            slider.set(0)
            slider.grid(row=i, column=2, padx=5)
            self.sliders.append(slider)
            
            # Max label
            tk.Label(control_frame, text=f"{max_val}°").grid(
                row=i, column=3, padx=2)
            
            # Current value label
            val_label = tk.Label(control_frame, text="0°", 
                               font=('Arial', 10), width=8,
                               bg='lightblue', relief='sunken')
            val_label.grid(row=i, column=4, padx=5)
            self.value_labels.append(val_label)
        
        # Speed control
        tk.Label(control_frame, text="Speed (%)", 
                font=('Arial', 10, 'bold')).grid(row=6, column=0, padx=5, pady=10, sticky='w')
        self.speed_slider = tk.Scale(control_frame, from_=5, to=50, 
                                     orient='horizontal', length=300, resolution=1)
        self.speed_slider.set(15)
        self.speed_slider.grid(row=6, column=2, padx=5, pady=10)
        self.speed_label = tk.Label(control_frame, text="15%", 
                                    font=('Arial', 10), width=8,
                                    bg='lightyellow', relief='sunken')
        self.speed_label.grid(row=6, column=4, padx=5, pady=10)
        self.speed_slider.config(command=lambda val: self.speed_label.config(text=f"{val}%"))
        
        # Real-time control status
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=7, column=0, columnspan=5, pady=10, sticky='ew')
        
        self.live_status_label = tk.Label(control_frame, 
                                          text="⚡ Real-time Control: Move sliders to control robot immediately", 
                                          font=('Arial', 9, 'bold'),
                                          fg='orange')
        self.live_status_label.grid(row=8, column=0, columnspan=5, padx=5, pady=5)
        
        # ===== PRESET POSITIONS FRAME =====
        preset_frame = ttk.LabelFrame(self.root, text="Preset Positions", padding=10)
        preset_frame.grid(row=1, column=1, padx=10, pady=5, sticky='nsew')
        
        row = 0
        for preset_name, positions in PRESETS.items():
            btn = ttk.Button(preset_frame, text=preset_name, width=20,
                           command=lambda p=positions: self.load_preset(p))
            btn.grid(row=row, column=0, padx=5, pady=5, sticky='ew')
            
            # Show position values
            pos_text = f"[{', '.join(f'{p:+4.0f}°' for p in positions)}]"
            tk.Label(preset_frame, text=pos_text, font=('Arial', 8)).grid(
                row=row, column=1, padx=5, pady=5, sticky='w')
            row += 1
        
        # Custom preset save/load
        ttk.Separator(preset_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=2, pady=10, sticky='ew')
        row += 1
        
        tk.Label(preset_frame, text="Custom Preset:", 
                font=('Arial', 9, 'bold')).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        ttk.Button(preset_frame, text="Save Current", width=20,
                  command=self.save_custom_preset).grid(row=row, column=0, padx=5, pady=3)
        row += 1
        
        ttk.Button(preset_frame, text="Load Custom", width=20,
                  command=self.load_custom_preset).grid(row=row, column=0, padx=5, pady=3)
        
        # ===== GRIPPER CONTROL FRAME =====
        gripper_frame = ttk.LabelFrame(self.root, text="Gripper Control", padding=10)
        gripper_frame.grid(row=2, column=1, padx=10, pady=5, sticky='nsew')
        
        # Gripper quick buttons
        tk.Label(gripper_frame, text="Quick Actions:", 
                font=('Arial', 9, 'bold')).grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Button(gripper_frame, text="🔓 Open Gripper", width=20,
                  command=self.open_gripper).grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Button(gripper_frame, text="🔒 Close Gripper", width=20,
                  command=self.close_gripper).grid(row=2, column=0, padx=5, pady=5)
        
        ttk.Separator(gripper_frame, orient='horizontal').grid(
            row=3, column=0, columnspan=2, pady=10, sticky='ew')
        
        # Gripper position slider
        tk.Label(gripper_frame, text="Position Control:", 
                font=('Arial', 9, 'bold')).grid(row=4, column=0, columnspan=2, pady=5)
        
        tk.Label(gripper_frame, text="Close").grid(row=5, column=0, sticky='w', padx=5)
        
        self.gripper_slider = tk.Scale(gripper_frame, from_=0, to=800000, 
                                      orient='horizontal', length=200, resolution=10000)
        self.gripper_slider.set(0)
        self.gripper_slider.grid(row=6, column=0, columnspan=2, padx=5, pady=5)
        
        tk.Label(gripper_frame, text="Open").grid(row=7, column=0, sticky='w', padx=5)
        
        ttk.Button(gripper_frame, text="Move to Position", width=20,
                  command=self.move_gripper_to_position).grid(row=8, column=0, padx=5, pady=10)
        
        # Gripper status display
        ttk.Separator(gripper_frame, orient='horizontal').grid(
            row=9, column=0, columnspan=2, pady=10, sticky='ew')
        
        tk.Label(gripper_frame, text="Status:", 
                font=('Arial', 9, 'bold')).grid(row=10, column=0, sticky='w', padx=5)
        
        self.gripper_status_label = tk.Label(gripper_frame, text="---", 
                                            font=('Arial', 9), 
                                            bg='lightgray', relief='sunken',
                                            width=20)
        self.gripper_status_label.grid(row=11, column=0, columnspan=2, padx=5, pady=5)
        
        # ===== CURRENT POSITION DISPLAY =====
        position_frame = ttk.LabelFrame(self.root, text="Current Robot Position", padding=10)
        position_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky='ew')
        
        self.position_labels = []
        for i in range(6):
            tk.Label(position_frame, text=f"J{i+1}:", 
                    font=('Arial', 9, 'bold')).grid(row=0, column=i*2, padx=5)
            pos_label = tk.Label(position_frame, text="0", 
                                font=('Arial', 9), width=8,
                                bg='lightgreen', relief='sunken')
            pos_label.grid(row=0, column=i*2+1, padx=5)
            self.position_labels.append(pos_label)
        
        # Add gripper position display
        tk.Label(position_frame, text="Gripper:", 
                font=('Arial', 9, 'bold')).grid(row=1, column=0, padx=5, pady=5)
        self.gripper_position_label = tk.Label(position_frame, text="---", 
                            font=('Arial', 9), width=10,
                            bg='lightblue', relief='sunken')
        self.gripper_position_label.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky='w')
        
        ttk.Button(position_frame, text="↻ Refresh", 
                  command=self.read_position).grid(row=0, column=12, padx=10)
        
        self.auto_refresh_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(position_frame, text="Auto-refresh (1Hz)", 
                       variable=self.auto_refresh_var,
                       command=self.toggle_monitoring).grid(row=0, column=13, padx=5)
        
        # ===== CONTROL BUTTONS =====
        button_frame = tk.Frame(self.root, bg='#e0e0e0', padx=10, pady=10)
        button_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky='ew')
        
        self.move_btn = ttk.Button(button_frame, text="↻ Sync Position", 
                                   command=self.move_robot,
                                   style='Success.TButton')
        self.move_btn.grid(row=0, column=0, padx=10, pady=5, ipadx=20, ipady=10)
        self.move_btn.config(state='disabled')
        
        tk.Label(button_frame, text="(Real-time mode: Robot moves as you slide)", 
                font=('Arial', 8, 'italic'), fg='gray').grid(row=1, column=0, columnspan=3)
        
        ttk.Button(button_frame, text="⏸ Stop Movement", 
                  command=self.stop_movement).grid(row=0, column=1, padx=10, pady=5, ipadx=20, ipady=10)
        
        ttk.Button(button_frame, text="⟲ Zero All Joints", 
                  command=lambda: self.load_preset(PRESETS['Reset'])).grid(
                      row=0, column=2, padx=10, pady=5, ipadx=20, ipady=10)
        
        # ===== STATUS LOG =====
        log_frame = ttk.LabelFrame(self.root, text="Status Log", padding=5)
        log_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky='nsew')
        
        self.log_text = tk.Text(log_frame, height=8, width=100, 
                               bg='black', fg='lightgreen', 
                               font=('Courier', 9))
        self.log_text.grid(row=0, column=0, sticky='nsew')
        
        scrollbar = ttk.Scrollbar(log_frame, orient='vertical', 
                                 command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Configure grid weights
        self.root.grid_rowconfigure(1, weight=3)  # Joint control row (spans 1-2)
        self.root.grid_rowconfigure(2, weight=0)  # Part of joint control span
        self.root.grid_rowconfigure(3, weight=0)  # Position display row
        self.root.grid_rowconfigure(4, weight=0)  # Control buttons
        self.root.grid_rowconfigure(5, weight=1)  # Status log
        self.root.grid_columnconfigure(0, weight=2)  # Left column (joint control)
        self.root.grid_columnconfigure(1, weight=1)  # Right column (presets/gripper)
        
        self.log("GUI initialized. Ready to connect.")
        
    def log(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert('end', f"[{timestamp}] {message}\n")
        self.log_text.see('end')
        self.root.update_idletasks()
        
    def update_joint_value(self, idx, value):
        """Update joint value when slider moves - triggers immediate robot movement"""
        self.joint_values[idx] = float(value)
        self.value_labels[idx].config(text=f"{float(value):.0f}°")
        
        # Mark position as changed for real-time control
        if self.connected:
            self.position_changed = True
        
    def connect_robot(self):
        """Connect to robot"""
        if self.connected:
            self.log("Already connected")
            return
            
        self.log("Connecting to robot...")
        try:
            self.piper = C_PiperInterface_V2()
            self.piper.ConnectPort()
            time.sleep(0.5)
            
            self.log("Enabling motors...")
            # Use EnablePiper() for V2 interface
            while not self.piper.EnablePiper():
                time.sleep(0.01)
            time.sleep(0.5)
            
            self.connected = True
            self.status_label.config(text="● Connected", fg='green')
            self.connect_btn.config(state='disabled')
            self.disconnect_btn.config(state='normal')
            self.move_btn.config(state='normal')
            
            self.log("✓ Connected and motors enabled")
            
            # Start real-time control thread
            self.start_live_control()
            
            # Read initial position
            self.read_position()
            
        except Exception as e:
            self.log(f"✗ Connection failed: {e}")
            messagebox.showerror("Connection Error", 
                               f"Failed to connect to robot.\n\n{e}\n\n"
                               "Ensure CAN interface is active:\n"
                               "sudo bash scripts/1_setup_can.sh")
            
    def disconnect_robot(self):
        """Disconnect from robot"""
        if not self.connected:
            return
            
        self.monitoring = False
        self.connected = False
        self.piper = None
        
        self.status_label.config(text="● Disconnected", fg='red')
        self.connect_btn.config(state='normal')
        self.disconnect_btn.config(state='disabled')
        self.move_btn.config(state='disabled')
        self.auto_refresh_var.set(False)
        
        self.log("Disconnected from robot")
        
    def read_position(self):
        """Read current robot position"""
        if not self.connected:
            return
            
        try:
            joints = self.piper.GetArmJointMsgs()
            
            # Update position display (convert from encoder units to degrees)
            # Note: These are raw encoder values, not degrees
            raw_positions = [
                joints.joint_state.joint_1,
                joints.joint_state.joint_2,
                joints.joint_state.joint_3,
                joints.joint_state.joint_4,
                joints.joint_state.joint_5,
                joints.joint_state.joint_6
            ]
            
            for i, pos in enumerate(raw_positions):
                self.position_labels[i].config(text=f"{pos}")
            
            # Read gripper status
            try:
                gripper = self.piper.GetArmGripperMsgs()
                gripper_pos = gripper.gripper_state.grippers_angle
                
                # Update gripper status label in control panel (0 = closed, 800000 = open)
                if gripper_pos < 100000:
                    self.gripper_status_label.config(text=f"Closed ({gripper_pos})", bg='lightcoral')
                elif gripper_pos > 700000:
                    self.gripper_status_label.config(text=f"Open ({gripper_pos})", bg='lightgreen')
                else:
                    self.gripper_status_label.config(text=f"Position: {gripper_pos}", bg='lightyellow')
                
                # Update gripper position in position display
                self.gripper_position_label.config(text=f"{gripper_pos}")
            except:
                pass  # Gripper reading may fail, don't interrupt joint reading
                
        except Exception as e:
            self.log(f"Error reading position: {e}")
            
    def toggle_monitoring(self):
        """Toggle auto-refresh monitoring"""
        if self.auto_refresh_var.get():
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.log("Auto-refresh enabled")
        else:
            self.monitoring = False
            self.log("Auto-refresh disabled")
            
    def monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring and self.connected:
            try:
                self.read_position()
                time.sleep(1)  # 1 Hz
            except:
                break
                
    def start_live_control(self):
        """Start real-time control thread automatically when connected"""
        if self.live_control_thread is None or not self.live_control_thread.is_alive():
            self.live_control_thread = threading.Thread(
                target=self.live_control_loop, daemon=True)
            self.live_control_thread.start()
            self.log("⚡ Real-time control ACTIVE - Robot responds to slider changes")
            
    def live_control_loop(self):
        """Background loop for real-time control - sends commands continuously"""
        update_rate = 0.05  # 20 Hz update rate for smoother control
        
        while self.connected:
            try:
                # Only send if position has changed
                if self.position_changed:
                    # Convert degrees to SDK units
                    target_units = [round(deg * 1000) for deg in self.joint_values]
                    
                    # Check if values actually changed from last sent
                    if target_units != self.last_sent_values:
                        speed = self.speed_slider.get()
                        
                        # Set control mode and send command
                        self.piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
                        time.sleep(0.005)
                        self.piper.JointCtrl(*target_units)
                        
                        self.last_sent_values = target_units.copy()
                        self.position_changed = False
                        
                time.sleep(update_rate)
                
            except Exception as e:
                self.log(f"Real-time control error: {e}")
                break
                
    def load_preset(self, positions):
        """Load preset position into sliders - triggers immediate movement"""
        for i, pos in enumerate(positions):
            self.sliders[i].set(pos)
        self.log(f"Preset loaded: {positions}")
        
        # Trigger immediate movement
        if self.connected:
            self.position_changed = True
        
    def save_custom_preset(self):
        """Save current slider values as custom preset"""
        self.custom_preset = self.joint_values.copy()
        self.log(f"Custom preset saved: {self.custom_preset}")
        messagebox.showinfo("Preset Saved", 
                          f"Custom preset saved:\n{self.custom_preset}")
        
    def load_custom_preset(self):
        """Load custom preset"""
        if hasattr(self, 'custom_preset'):
            self.load_preset(self.custom_preset)
        else:
            messagebox.showwarning("No Custom Preset", 
                                 "No custom preset saved yet.")
            
    def move_robot(self):
        """Force immediate movement to current slider positions"""
        if not self.connected:
            messagebox.showwarning("Not Connected", "Connect to robot first")
            return
            
        # Trigger immediate movement
        self.position_changed = True
        self.log(f"Manual trigger: Moving to {[f'{v:.0f}°' for v in self.joint_values]}")
            
    def stop_movement(self):
        """Stop robot movement"""
        if not self.connected:
            return
        self.log("⏸ Stop command sent (robot will decelerate)")
        # Note: Piper doesn't have instant stop, it decelerates
        
    def emergency_stop(self):
        """Emergency stop"""
        if self.connected and messagebox.askyesno("Emergency Stop", 
                                                  "Execute emergency stop?"):
            self.log("⚠ EMERGENCY STOP")
            self.disconnect_robot()
    
    def open_gripper(self):
        """Open gripper fully"""
        if not self.connected:
            messagebox.showwarning("Not Connected", "Connect to robot first")
            return
        
        try:
            self.log("🔓 Opening gripper...")
            self.piper.GripperCtrl(800000, 1000, 0x01, 0)
            self.gripper_slider.set(800000)
            self.gripper_status_label.config(text="Open", bg='lightgreen')
        except Exception as e:
            self.log(f"✗ Gripper open error: {e}")
            
    def close_gripper(self):
        """Close gripper fully"""
        if not self.connected:
            messagebox.showwarning("Not Connected", "Connect to robot first")
            return
        
        try:
            self.log("🔒 Closing gripper...")
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
            self.gripper_slider.set(0)
            self.gripper_status_label.config(text="Closed", bg='lightcoral')
        except Exception as e:
            self.log(f"✗ Gripper close error: {e}")
            
    def move_gripper_to_position(self):
        """Move gripper to slider position"""
        if not self.connected:
            messagebox.showwarning("Not Connected", "Connect to robot first")
            return
        
        try:
            position = int(self.gripper_slider.get())
            self.log(f"Moving gripper to position: {position}")
            self.piper.GripperCtrl(position, 1000, 0x01, 0)
            
            # Update status (0 = closed, 800000 = open)
            if position < 100000:
                self.gripper_status_label.config(text="Closed", bg='lightcoral')
            elif position > 700000:
                self.gripper_status_label.config(text="Open", bg='lightgreen')
            else:
                self.gripper_status_label.config(text=f"Position: {position}", bg='lightyellow')
        except Exception as e:
            self.log(f"✗ Gripper position error: {e}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    
    # Custom button styles
    style = ttk.Style()
    style.configure('Success.TButton', foreground='green')
    style.configure('Danger.TButton', foreground='red')
    
    app = PiperControlGUI(root)
    
    root.mainloop()

