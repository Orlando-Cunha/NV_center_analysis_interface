import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import os
from odmr_analy import odmr_analyze_data
from rabi_analy import rabi_analyze_data
import scipy.optimize
import matplotlib.pyplot as plt
import threading
from analysis_utils import preprocess_odmr_data, extract_roi_trace, split_signal_reference

# Helper for modular section creation

def create_roi_section(parent):
    frame = ttk.LabelFrame(parent, text="3. ROI Input")
    labels = ["X start", "X end", "Y start", "Y end"]
    entries = {}
    for i, label in enumerate(labels):
        row = i // 2
        col = (i % 2) * 2
        ttk.Label(frame, text=label).grid(row=row, column=col, padx=5, pady=2, sticky="w")
        entry = ttk.Entry(frame, width=8)
        entry.grid(row=row, column=col+1, padx=5, pady=2)
        entries[label] = entry
    return frame, entries

def create_param_section(parent, protocol):
    frame = ttk.LabelFrame(parent, text="4. Experimental Parameters")
    entries = {}
    if protocol == "CW":
        labels = ["Start (MHz)", "End (MHz)", "Step (MHz)"]
        keys = ["Start", "End", "Steps"]
        for i, (label, key) in enumerate(zip(labels, keys)):
            ttk.Label(frame, text=label).grid(row=0, column=i, padx=5, pady=2, sticky="w")
            entry = ttk.Entry(frame, width=10)
            entry.grid(row=1, column=i, padx=5, pady=2)
            entries[label] = entry
            entries[key] = entry  # allow access by both label and key
        # Add S+R checkbox (Signal+Reference)
        sr_var = tk.BooleanVar(value=False)
        sr_cb = ttk.Checkbutton(frame, text="S+R", variable=sr_var)
        sr_cb.grid(row=0, column=len(labels), rowspan=2, padx=5, pady=2, sticky="w")
        entries["S+R"] = sr_var
        return frame, entries
    elif protocol in ("Rabi", "Ramsey"):
        labels = ["Start (ns)", "End (ns)", "Number of Points"]
        keys = ["Start", "End", "Number of Points"]
        for i, (label, key) in enumerate(zip(labels, keys)):
            ttk.Label(frame, text=label).grid(row=0, column=i, padx=5, pady=2, sticky="w")
            entry = ttk.Entry(frame, width=10)
            entry.grid(row=1, column=i, padx=5, pady=2)
            entries[label] = entry
            entries[key] = entry
        return frame, entries
    else:
        labels = ["Start", "End", "Steps"]
    for i, label in enumerate(labels):
        ttk.Label(frame, text=label).grid(row=0, column=i, padx=5, pady=2, sticky="w")
        entry = ttk.Entry(frame, width=10)
        entry.grid(row=1, column=i, padx=5, pady=2)
        entries[label] = entry
    return frame, entries

# Add the exact model functions from ramsey.py
import numpy as np
from scipy.interpolate import interp1d

def double_exp_decay_cosine(t, A1, T1, f1, phi1, A2, T2, f2, phi2, y0):
    return A1 * np.exp(-t / T1)**2 * np.cos(2 * np.pi * f1 * t + phi1) + \
           A2 * np.exp(-t / T2)**2 * np.cos(2 * np.pi * f2 * t + phi2) + y0

def exp_decay_single_cos(t, A, B, T, beta, f, phi, y0):
    return B * t + A * np.exp(-t / T - beta * t**2) * np.cos(2 * np.pi * f * t + phi) + y0

# Refactor create_fit_param_section for Ramsey to destroy/recreate widgets on model change
import tkinter as tk
from tkinter import ttk

# --- Equations for each protocol (LaTeX) ---
FIT_EQUATIONS = {
    'CW': r"$\sum_i A_i \, / \, \left[1 + \left(\frac{x - x_{0i}}{\gamma_i}\right)^2\right]$",

    'Rabi': r"$A \cos(\Omega x + \phi)\, e^{-x/\tau} + Bx + C$",

    'Ramsey_Single': r"$A\, e^{-t/T - \beta t^2} \cos(2\pi f t + \phi) + y_0 + B t$",

    'Ramsey_Double': r"$\sum_{i=1}^{2} A_i\, e^{-(t/T_i)^2} \cos(2\pi f_i t + \phi_i) + y_0$",

    'T1': r"$a\, e^{-(t/T_1)^\beta} + c$"
}

# Greek and subscript label mapping for fit parameters
GREEK_LABELS = {
    "phi": "ϕ",
    "phi1": "ϕ₁",
    "phi2": "ϕ₂",
    "beta": "β",
    "Omega": "Ω",
    "omega": "ω",
    "tau": "τ",
    "mu": "μ",
    "T1": "T₁",
    "T2": "T₂",
    "A1": "A₁",
    "A2": "A₂",
    "f1": "f₁",
    "f2": "f₂",
    "y0": "y₀",
    "Phase":"ϕ",
    "Decay Rate": "τ"
    # Add more as needed
}

def create_fit_param_section(parent, protocol, fit_param_config):
    frame = ttk.LabelFrame(parent, text="6. Fit Parameters")
    entries = {}
    # --- Equation display ---
    eq_frame = ttk.Frame(frame)
    eq_frame.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(2, 2))
    # Remove any previous equation canvas if present
    def show_equation(equation_latex, protocol=None):
        for widget in eq_frame.winfo_children():
            widget.destroy()
        fig = plt.Figure(figsize=(2.0, 0.18), dpi=300)  # Much shorter height
        ax = fig.add_subplot(111)
        ax.axis('off')
        if protocol in ('Rabi', 'Ramsey'):
            fontsize = 3.3
        elif protocol == 'CW':
            fontsize = 4
        else:
            fontsize = 5.3
        ax.text(0.5, 0.5, equation_latex, fontsize=fontsize, ha='center', va='center')
        fig.subplots_adjust(top=1, bottom=0, left=0, right=1)  # Remove padding
        canvas = FigureCanvasTkAgg(fig, master=eq_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    # --- Protocol-specific logic ---
    if protocol == "Ramsey":
        # Model selection dropdown
        model_var = tk.StringVar(value="Double Cosine")
        ttk.Label(frame, text="Model:").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        model_cb = ttk.Combobox(frame, textvariable=model_var, values=["Single Cosine", "Double Cosine"], state="readonly", width=15)
        model_cb.grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        entries["model"] = model_var
        param_frame = ttk.Frame(frame)
        param_frame.grid(row=2, column=0, columnspan=4, sticky="ew")
        for i in range(4):
            param_frame.columnconfigure(i, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        frame.columnconfigure(3, weight=1)
        param_entries = {}
        def build_param_fields(model):
            for widget in param_frame.winfo_children():
                widget.destroy()
            param_entries.clear()
            if model == "Single Cosine":
                param_list = [
                    ("A", 0.5), ("B", 0.01), ("T", 80e-9), ("beta", 1e9), ("f", 37e6), ("phi", 0), ("y0", 0.3)
                ]
                show_equation(FIT_EQUATIONS['Ramsey_Single'], protocol)
            else:
                param_list = [
                    ("A1", 0.5), ("T1", 80e-9), ("f1", 37.0e6), ("phi1", 0),
                    ("A2", 0.3), ("T2", 60e-9), ("f2", 35.0e6), ("phi2", 0), ("y0", 0.3)
                ]
                show_equation(FIT_EQUATIONS['Ramsey_Double'], protocol)
            for i, (label, default) in enumerate(param_list):
                label_text = GREEK_LABELS.get(label, label)
                ttk.Label(param_frame, text=label_text).grid(row=i//2, column=(i%2)*2, sticky="ew", padx=2, pady=2)
                var = tk.StringVar(value=str(default))
                entry = ttk.Entry(param_frame, textvariable=var, width=8)
                entry.grid(row=i//2, column=(i%2)*2+1, sticky="ew", padx=2, pady=2)
                param_entries[label] = var
            entries.update(param_entries)
        build_param_fields(model_var.get())
        def on_model_change(*args):
            build_param_fields(model_var.get())
        model_var.trace_add('write', on_model_change)
        # Fit and Save Fit Results buttons side by side
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=3, column=0, columnspan=4, sticky="ew", padx=2, pady=(5, 0))
        entries["fit_btn"] = ttk.Button(btn_frame, text="Run Fit")
        entries["fit_btn"].pack(side="left", expand=True, fill=tk.X, padx=(0, 2))
        entries["save_fit_btn"] = ttk.Button(btn_frame, text="Save Fit Results")
        entries["save_fit_btn"].pack(side="left", expand=True, fill=tk.X, padx=(2, 0))
        return frame, entries
    # Default for other protocols
    row = 1
    col = 0
    max_cols = 4
    # Show equation for protocol
    if protocol == 'CW':
        show_equation(FIT_EQUATIONS['CW'], protocol)
    elif protocol == 'Rabi':
        show_equation(FIT_EQUATIONS['Rabi'], protocol)
    elif protocol == 'T1':
        show_equation(FIT_EQUATIONS['T1'], protocol)
    for i in range(max_cols):
        frame.columnconfigure(i, weight=1)
    for param in fit_param_config:
        # Use Greek label if available
        label_text = GREEK_LABELS.get(param["label"], param["label"])
        ttk.Label(frame, text=label_text).grid(row=row, column=col*2, sticky="ew", padx=2, pady=2)
        if param.get("type") == "bool":
            var = tk.BooleanVar(value=param.get("default", False))
            cb = ttk.Checkbutton(frame, variable=var)
            cb.grid(row=row, column=col*2+1, sticky="ew", padx=2, pady=2)
            entries[param["name"]] = var
        else:
            var = tk.StringVar(value=str(param.get("default", "")))
            entry_width = 5 if param["name"] in ("num_peaks", "threshold") else 8
            entry = ttk.Entry(frame, textvariable=var, width=entry_width)
            entry.grid(row=row, column=col*2+1, sticky="ew", padx=2, pady=2)
            entries[param["name"]] = var
        if col == 1:
            row += 1
            col = 0
        else:
            col = 1
    # Fit and Save Fit Results buttons side by side
    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=row+1, column=0, columnspan=4, sticky="ew", padx=2, pady=(5, 0))
    entries["fit_btn"] = ttk.Button(btn_frame, text="Run Fit")
    entries["fit_btn"].pack(side="left", expand=True, fill=tk.X, padx=(0, 2))
    entries["save_fit_btn"] = ttk.Button(btn_frame, text="Save Fit Results")
    entries["save_fit_btn"].pack(side="left", expand=True, fill=tk.X, padx=(2, 0))
    return frame, entries

class NVAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NV Protocol Analysis")
        self.root.geometry("1200x700")
        self.root.state('zoomed')
        self.protocols = ["CW", "Rabi", "Ramsey", "T1"]
        self.fit_param_configs = {
            "CW": [
                {"name": "num_peaks", "label": "Number of Peaks", "type": "int", "default": 4},
                {"name": "threshold", "label": "Threshold", "type": "float", "default": 0.974},
                {"name": "show_plot", "label": "Intermediate plot", "type": "bool", "default": False},
                {"name": "use_filtered", "label": "Use filtered data", "type": "bool", "default": False},
            ],
            "Rabi": [
                {"name": "Amplitude", "label": "Amplitude", "default": 1.0},
                {"name": "Omega", "label": "Omega", "default": 0.01},
                {"name": "Phase", "label": "Phase", "default": 0.0},
                {"name": "Decay Rate", "label": "Decay Rate", "default": 1000.0},
                {"name": "B", "label": "B", "default": 0.0},
                {"name": "C", "label": "C", "default": 1.0},
            ],
            "Ramsey": [
                {"name": "Amplitude", "label": "Amplitude"},
                {"name": "Decay Rate", "label": "Decay Rate"},
                {"name": "Phase Shift", "label": "Phase Shift"},
            ],
            "T1": [
                {"name": "Amplitude", "label": "Amplitude", "default": 0.5},
                {"name": "T1", "label": "T1", "default": 2.0},
                {"name": "beta", "label": "beta", "default": 0.8},
                {"name": "C", "label": "C", "default": 0.1},
            ]
        }
        self.tab_results = {}  # Map tab widget to result object
        self.tab_fit_results = {}  # Map tab widget to latest fit result (dict with x, y, y_fit, params, equation)
        self.build_ui()

    def build_ui(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        main_pane = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)
        # === Left: User Controls ===
        controls = ttk.Frame(main_pane, width=120, padding=(10, 0, 0, 0))
        controls.pack_propagate(False)
        main_pane.add(controls, weight=1)
        # Data Loading Section
        data_load_box = ttk.LabelFrame(controls, text="1. Load Data")
        data_load_box.pack(pady=(0, 10))
        ttk.Button(data_load_box, text="Browse...", command=self.load_data, width=120).pack(pady=(0, 5))
        self.loaded_label = ttk.Label(data_load_box, text="Loaded: <filename>")
        self.loaded_label.pack(anchor="w", pady=(5, 0))
        self.shape_label = ttk.Label(data_load_box, text="Shape: -")
        self.shape_label.pack(anchor="w", pady=(0, 0))
        # Protocol Selection
        proto_box = ttk.LabelFrame(controls, text="2. Select Protocol")
        proto_box.pack(pady=(0, 10))
        self.proto_cb = ttk.Combobox(proto_box, values=self.protocols, state="readonly", width=120)
        self.proto_cb.pack(pady=5)
        self.proto_cb.bind("<<ComboboxSelected>>", self.on_protocol_change)
        # ROI Section
        self.roi_frame, self.roi_entries = create_roi_section(controls)
        self.roi_frame.pack(fill=tk.X, pady=(0, 10))
        # Experimental Parameters Section (or T1 custom range) - do NOT pack here
        self.param_frame, self.param_entries = create_param_section(controls, None)
        self.custom_range_frame = ttk.LabelFrame(controls, text="Custom Range (T1 Only)")
        self.custom_range_label = ttk.Label(self.custom_range_frame, text="Enter comma-separated values (ms):")
        self.custom_range_label.pack(anchor="w", padx=5, pady=2)
        self.custom_range_entry = ttk.Entry(self.custom_range_frame)
        self.custom_range_entry.pack(fill=tk.X, padx=5, pady=2)
        # Run Analysis and Save Results Buttons (side by side)
        analysis_btn_frame = ttk.Frame(controls)
        analysis_btn_frame.pack(pady=(0, 10), fill=tk.X)
        self.run_button = ttk.Button(analysis_btn_frame, text="5. Run Analysis", command=self.run_analysis)
        self.run_button.pack(side="left", expand=True, fill=tk.X, padx=(0, 2))
        self.save_button = ttk.Button(analysis_btn_frame, text="Save Results", command=self.save_results)
        self.save_button.pack(side="left", expand=True, fill=tk.X, padx=(2, 0))
        # Fit Parameters Section (dynamic)
        self.fit_param_frame = None
        self.fit_param_entries = None
        self.create_fit_param_section("CW")
        # === Right: Output Area ===
        output_frame = ttk.Frame(main_pane, padding=10)
        main_pane.add(output_frame, weight=5)
        output_frame.columnconfigure(0, weight=1)
        output_frame.columnconfigure(1, weight=5)
        output_frame.columnconfigure(2, weight=1)
        output_frame.rowconfigure(0, weight=1)
        output_frame.rowconfigure(1, weight=10)
        # Results Notebook (now for each analysis run, not per protocol)
        self.results_notebook = ttk.Notebook(output_frame)
        self.results_notebook.grid(row=1, column=0, columnspan=3, rowspan=2, sticky="nsew", padx=0, pady=0)
        self.result_tabs = []  # List of (tab, label)
        # Add close tab support
        self.results_notebook.enable_traversal()
        # Add context menu for closing tabs
        self.tab_context_menu = tk.Menu(self.root, tearoff=0)
        self.tab_context_menu.add_command(label="Close Tab", command=self._close_tab_from_menu)
        self.results_notebook.bind('<Button-3>', self._on_tab_right_click)
        self._context_menu_tab_index = None
        # Set initial protocol
        self.proto_cb.current(0)
        self.on_protocol_change()

    def _on_tab_right_click(self, event):
        x, y = event.x, event.y
        try:
            index = self.results_notebook.index(f"@{x},{y}")
        except Exception:
            return
        self._context_menu_tab_index = index
        self.tab_context_menu.tk_popup(event.x_root, event.y_root)

    def _close_tab_from_menu(self):
        if self._context_menu_tab_index is not None:
            self._close_tab(self._context_menu_tab_index)
            self._context_menu_tab_index = None

    def _add_tab_with_close(self, tab, label):
        display_label = f"{label}   x"
        self.results_notebook.add(tab, text=display_label)
        self.result_tabs.append((tab, label))

    def create_fit_param_section(self, protocol):
        if self.fit_param_frame:
            self.fit_param_frame.destroy()
        fit_param_config = self.fit_param_configs.get(protocol, [])
        if fit_param_config:
            self.fit_param_frame, self.fit_param_entries = create_fit_param_section(self.root.nametowidget(self.param_frame.master), protocol, fit_param_config)
            self.fit_param_frame.pack(fill=tk.X, pady=10)
            # Wire up Save Fit Results button
            if "save_fit_btn" in self.fit_param_entries:
                self.fit_param_entries["save_fit_btn"].config(command=self.save_fit_results)
        else:
            self.fit_param_frame = None
            self.fit_param_entries = None

    def on_protocol_change(self, event=None):
        proto = self.proto_cb.get()
        # Always remove both from the layout
        self.param_frame.pack_forget()
        self.custom_range_frame.pack_forget()
        # Re-create param_frame for correct protocol
        self.param_frame, self.param_entries = create_param_section(self.root.nametowidget(self.param_frame.master), proto)
        # Pack the correct one just before the Run Analysis button
        if proto == "T1":
            self.custom_range_frame.pack(fill=tk.X, pady=10, before=self.run_button)
        else:
            self.param_frame.pack(fill=tk.X, pady=10, before=self.run_button)
        self.create_fit_param_section(proto)

    def run_analysis(self):
        proto = self.proto_cb.get()
        filename = self.loaded_label.cget("text").replace("Loaded: ", "")
        tab_label = f"{proto} - {filename}" if filename and filename != "<filename>" else proto
        tab = ttk.Frame(self.results_notebook)
        self._add_tab_with_close(tab, tab_label)
        self.results_notebook.select(tab)
        # CW protocol wiring
        if proto == "CW":
            # Validate data
            if not hasattr(self, "data") or self.data is None:
                messagebox.showerror("Error", "No data loaded.")
                return
            # Extract ROI
            try:
                x_min = int(self.roi_entries["X start"].get())
                x_max = int(self.roi_entries["X end"].get())
                y_min = int(self.roi_entries["Y start"].get())
                y_max = int(self.roi_entries["Y end"].get())
            except Exception:
                messagebox.showerror("Input Error", "Invalid ROI values.")
                return
            # Validate ROI bounds for CW (512x512)
            IMAGE_SIZE = 512
            if not (0 <= x_min < x_max <= IMAGE_SIZE and 0 <= y_min < y_max <= IMAGE_SIZE):
                messagebox.showerror("Input Error", f"ROI values must be within 0 and {IMAGE_SIZE}, and min < max for both axes.")
                return
            # Extract experimental parameters
            try:
                start = float(self.param_entries["Start"].get())
                end = float(self.param_entries["End"].get())
                step = float(self.param_entries["Steps"].get())
            except Exception:
                messagebox.showerror("Input Error", "Invalid experimental parameters.")
                return
            # Compute x_range and num_points
            x_range = np.arange(start, end + step, step)
            num_points = len(x_range)
            # Compute num_averages from data shape
            try:
                num_averages = int(self.data.shape[0] // num_points)
            except Exception:
                messagebox.showerror("Data Error", "Data shape does not match computed number of points.")
                return
            if num_averages * num_points != self.data.shape[0]:
                messagebox.showerror("Data Error", f"Data shape ({self.data.shape[0]}) is not a multiple of computed number of points ({num_points}).")
                return
            # --- Use preprocessed data if available ---
            if hasattr(self, "preprocessed_cw_data") and self.preprocessed_cw_data is not None and self.preprocessed_cw_data.shape[0] == num_points:
                print("[CW] Using fast ROI extraction pipeline.")
                try:
                    odmr_trace = extract_roi_trace(self.preprocessed_cw_data, x_min, x_max, y_min, y_max)
                    from analysis_results import ODMRResult
                    result = ODMRResult(
                        x=x_range,
                        y=odmr_trace,
                        y2=odmr_trace,
                        image=self.preprocessed_cw_data[-1],
                        x_min=x_min,
                        x_max=x_max,
                        y_min=y_min,
                        y_max=y_max,
                        filtered_y=None
                    )
                    self.tab_results[tab] = result
                    self.populate_cw_result_tab(tab, result)
                    return
                except Exception as e:
                    print("CW fast ROI extraction failed, falling back:", e)
            # --- First run: do full analysis and preprocess for future fast ROI ---
            print("[CW] Running full analysis and preprocessing for future fast ROI extraction.")
            image = self.data[-1] if self.data.ndim == 3 else self.data
            try:
                result = odmr_analyze_data(
                    image=image,
                    data=self.data,
                    x_range=x_range,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    num_averages=num_averages,
                    num_points=num_points
                )
                # After successful run, preprocess and store for future fast ROI
                usable = self.data[3*num_points:]
                num_usable_frames = usable.shape[0]
                if num_usable_frames % num_points != 0:
                    self.preprocessed_cw_data = None
                    print("[CW] Preprocessing skipped: shape mismatch (usable frames not divisible by num_points).")
                else:
                    num_averages_post = num_usable_frames // num_points
                    self.preprocessed_cw_data = preprocess_odmr_data(usable, num_averages_post, num_points)
                    self.cw_num_points = num_points
                    print("[CW] Preprocessing complete. Fast ROI extraction enabled for future runs.")
            except Exception as e:
                messagebox.showerror("Analysis Error", f"CW analysis failed: {e}")
                return
            self.tab_results[tab] = result  # Attach result to tab
            self.populate_cw_result_tab(tab, result)
        elif proto == "Rabi":
            # Validate data
            if not hasattr(self, "data") or self.data is None:
                messagebox.showerror("Error", "No data loaded.")
                return
            # Extract ROI
            try:
                x_min = int(self.roi_entries["X start"].get())
                x_max = int(self.roi_entries["X end"].get())
                y_min = int(self.roi_entries["Y start"].get())
                y_max = int(self.roi_entries["Y end"].get())
                
                # Validate ROI bounds
                if not hasattr(self, "data") or self.data is None:
                    messagebox.showerror("Error", "No data loaded.")
                    return
                
                data_shape = self.data.shape
                if len(data_shape) >= 3:
                    max_y, max_x = data_shape[-2], data_shape[-1]
                else:
                    max_y, max_x = data_shape[0], data_shape[1]
                
                if (x_min < 0 or x_max > max_x or y_min < 0 or y_max > max_y or
                    x_min >= x_max or y_min >= y_max):
                    messagebox.showerror("Input Error", 
                                       f"Invalid ROI bounds. Data shape: ({max_y}, {max_x}). "
                                       f"ROI must be within bounds and have positive size.")
                    return
                    
            except Exception:
                messagebox.showerror("Input Error", "Invalid ROI values.")
                return
            # Extract experimental parameters
            try:
                start = float(self.param_entries["Start"].get())
                end = float(self.param_entries["End"].get())
                num_points = int(self.param_entries["Number of Points"].get())
                if num_points < 2:
                    raise ValueError("Number of Points must be at least 2.")
                step = (end - start) / (num_points - 1)
            except Exception:
                messagebox.showerror("Input Error", "Invalid experimental parameters.")
                return
            # Compute x_range
            x_range = np.linspace(start, end, num_points)
            # Compute num_averages from data shape
            try:
                num_averages = int(self.data.shape[0] // (2 * num_points))
            except Exception:
                messagebox.showerror("Data Error", "Data shape does not match computed number of points for Rabi.")
                return
            if num_averages * 2 * num_points != self.data.shape[0]:
                messagebox.showerror("Data Error", f"Data shape ({self.data.shape[0]}) is not a multiple of 2 x computed number of points ({num_points}).")
                return
            image = self.data[-1] if self.data.ndim == 3 else self.data
            
            # Check if we can use preprocessed data for fast ROI extraction
            use_preprocessed = (hasattr(self, 'rabi_pixel_traces') and 
                              hasattr(self, 'rabi_num_points') and 
                              self.rabi_num_points == num_points and
                              hasattr(self, 'rabi_x_range') and 
                              np.array_equal(self.rabi_x_range, x_range))
            
            if use_preprocessed:
                print("[Rabi] Using preprocessed data for fast ROI extraction")
                try:
                    # Extract ROI trace from preprocessed data
                    roi_trace = self.extract_rabi_roi_trace(x_min, x_max, y_min, y_max)
                    
                    # Create a minimal result object for compatibility
                    class FastRabiResult:
                        def __init__(self, x, y, image, roi_bounds):
                            self.x = x
                            self.y = y
                            self.error = np.zeros_like(y)  # No error info in fast mode
                            self.mean_signal = y  # Use normalized trace as signal
                            self.mean_reference = np.ones_like(y)  # Reference is 1 after normalization
                            self.image = image
                            self.roi_bounds = roi_bounds
                    
                    result = FastRabiResult(x_range, roi_trace, image, (x_min, x_max, y_min, y_max))
                    print("[Rabi] Fast ROI extraction complete")
                except Exception as e:
                    print(f"[Rabi] Fast extraction failed, falling back to full analysis: {e}")
                    use_preprocessed = False
            
            if not use_preprocessed:
                print("[Rabi] Running full analysis and preprocessing data")
                try:
                    result = rabi_analyze_data(
                        image=image,
                        data=self.data,
                        x_range=x_range,
                        x_min=x_min,
                        x_max=x_max,
                        y_min=y_min,
                        y_max=y_max,
                        num_averages=num_averages,
                        num_points=num_points
                    )
                    
                    # Preprocess data for future fast ROI extraction
                    try:
                        self.rabi_pixel_traces = self.preprocess_rabi_data(self.data, num_averages, num_points)
                        self.rabi_num_points = num_points
                        self.rabi_x_range = x_range.copy()
                        print("[Rabi] Preprocessing complete. Fast ROI extraction enabled for future runs.")
                    except Exception as e:
                        print(f"[Rabi] Preprocessing failed: {e}")
                        
                except Exception as e:
                    messagebox.showerror("Analysis Error", f"Rabi analysis failed: {e}")
                    return
            
            self.tab_results[tab] = result  # Attach result to tab
            self.populate_rabi_result_tab(tab, result)
        elif proto == "Ramsey":
            if not hasattr(self, "data") or self.data is None:
                messagebox.showerror("Error", "No data loaded.")
                return
            try:
                x_min = int(self.roi_entries["X start"].get())
                x_max = int(self.roi_entries["X end"].get())
                y_min = int(self.roi_entries["Y start"].get())
                y_max = int(self.roi_entries["Y end"].get())
                # Validate ROI bounds
                if not hasattr(self, "data") or self.data is None:
                    messagebox.showerror("Error", "No data loaded.")
                    return
                data_shape = self.data.shape
                if len(data_shape) >= 3:
                    max_y, max_x = data_shape[-2], data_shape[-1]
                else:
                    max_y, max_x = data_shape[0], data_shape[1]
                if (x_min < 0 or x_max > max_x or y_min < 0 or y_max > max_y or
                    x_min >= x_max or y_min >= y_max):
                    messagebox.showerror("Input Error", 
                                       f"Invalid ROI bounds. Data shape: ({max_y}, {max_x}). "
                                       f"ROI must be within bounds and have positive size.")
                    return
            except Exception:
                messagebox.showerror("Input Error", "Invalid ROI values.")
                return
            try:
                start = float(self.param_entries["Start"].get())
                end = float(self.param_entries["End"].get())
                num_points = int(self.param_entries["Number of Points"].get())
                if num_points < 2:
                    raise ValueError("Number of Points must be at least 2.")
                step = (end - start) / (num_points - 1)
            except Exception:
                messagebox.showerror("Input Error", "Invalid experimental parameters.")
                return
            x_range = np.linspace(start, end, num_points)
            try:
                num_averages = int(self.data.shape[0] // (2 * num_points))
            except Exception:
                messagebox.showerror("Data Error", "Data shape does not match computed number of points for Ramsey.")
                return
            if num_averages * 2 * num_points != self.data.shape[0]:
                messagebox.showerror("Data Error", f"Data shape ({self.data.shape[0]}) is not a multiple of 2 x computed number of points ({num_points}).")
                return
            image = self.data[-1] if self.data.ndim == 3 else self.data
            # Check if we can use preprocessed data for fast ROI extraction
            use_preprocessed = (hasattr(self, 'ramsey_pixel_traces') and 
                              hasattr(self, 'ramsey_num_points') and 
                              self.ramsey_num_points == num_points and
                              hasattr(self, 'ramsey_x_range') and 
                              np.array_equal(self.ramsey_x_range, x_range))
            if use_preprocessed:
                print("[Ramsey] Using preprocessed data for fast ROI extraction")
                try:
                    roi_trace = self.extract_ramsey_roi_trace(x_min, x_max, y_min, y_max)
                    class FastRamseyResult:
                        def __init__(self, x, y, image, roi_bounds):
                            self.x = x
                            self.y = y
                            self.error = np.zeros_like(y)
                            self.mean_signal = y
                            self.mean_reference = np.ones_like(y)
                            self.image = image
                            self.roi_bounds = roi_bounds
                    result = FastRamseyResult(x_range, roi_trace, image, (x_min, x_max, y_min, y_max))
                    print("[Ramsey] Fast ROI extraction complete")
                except Exception as e:
                    print(f"[Ramsey] Fast extraction failed, falling back to full analysis: {e}")
                    use_preprocessed = False
            if not use_preprocessed:
                print("[Ramsey] Running full analysis and preprocessing data")
                try:
                    # Use the Rabi analysis logic for Ramsey
                    result = rabi_analyze_data(
                        image=image,
                        data=self.data,
                        x_range=x_range,
                        x_min=x_min,
                        x_max=x_max,
                        y_min=y_min,
                        y_max=y_max,
                        num_averages=num_averages,
                        num_points=num_points
                    )
                    try:
                        self.ramsey_pixel_traces = self.preprocess_ramsey_data(self.data, num_averages, num_points)
                        self.ramsey_num_points = num_points
                        self.ramsey_x_range = x_range.copy()
                        print("[Ramsey] Preprocessing complete. Fast ROI extraction enabled for future runs.")
                    except Exception as e:
                        print(f"[Ramsey] Preprocessing failed: {e}")
                except Exception as e:
                    messagebox.showerror("Analysis Error", f"Ramsey analysis failed: {e}")
                    return
            self.tab_results[tab] = result  # Attach result to tab
            self.populate_ramsey_result_tab(tab, result)
        elif proto == "T1":
            if not hasattr(self, "data") or self.data is None:
                messagebox.showerror("Error", "No data loaded.")
                return
            try:
                x_min = int(self.roi_entries["X start"].get())
                x_max = int(self.roi_entries["X end"].get())
                y_min = int(self.roi_entries["Y start"].get())
                y_max = int(self.roi_entries["Y end"].get())
            except Exception:
                messagebox.showerror("Input Error", "Invalid ROI values.")
                return
            # Use custom x_range if all fields are empty
            default_x_range = np.array([15000, 14000, 13000, 12000, 11000, 10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1500, 1200, 1000, 800, 500, 300, 100]) * 1e-3
            start_str = self.param_entries["Start"].get()
            end_str = self.param_entries["End"].get()
            step_str = self.param_entries["Steps"].get()
            if start_str.strip() == "" and end_str.strip() == "" and step_str.strip() == "":
                x_range = default_x_range
            else:
                try:
                    start = float(start_str) if start_str.strip() != "" else default_x_range[-1]
                    end = float(end_str) if end_str.strip() != "" else default_x_range[0]
                    step = float(step_str) if step_str.strip() != "" else -(abs(end-start)/max(1,len(default_x_range)-1))
                    if step > 0:
                        step = -step  # Ensure descending order
                    x_range = np.arange(end, start+step, step)[::-1] if step < 0 else np.arange(start, end+step, step)
                except Exception:
                    messagebox.showerror("Input Error", "Invalid experimental parameters.")
                    return
            num_points = len(x_range)
            try:
                num_averages = int(self.data.shape[0] // num_points)
            except Exception:
                messagebox.showerror("Data Error", "Data shape does not match computed number of points.")
                return
            if num_averages * num_points != self.data.shape[0]:
                messagebox.showerror("Data Error", f"Data shape ({self.data.shape[0]}) is not a multiple of computed number of points ({num_points}).")
                return
            image = self.data[-1] if self.data.ndim == 3 else self.data
            try:
                from t1_analy import t1_analyze_data
                result = t1_analyze_data(
                    image=image,
                    data=self.data,
                    x_range=x_range,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max
                )
            except Exception as e:
                messagebox.showerror("Analysis Error", f"T1 analysis failed: {e}")
                return
            self.tab_results[tab] = result  # Attach result to tab
            self.populate_t1_result_tab(tab, result)
        else:
            self.populate_result_tab(tab, proto)

    def save_results(self):
        # Save only the x and y data shown on the main data graph for the selected tab, with x header including units
        idx = self.results_notebook.index(self.results_notebook.select())
        tab, label = self.result_tabs[idx]
        result = self.tab_results.get(tab, None)
        # Determine protocol from label (format: 'Protocol - filename' or just 'Protocol')
        proto = label.split(' - ')[0].strip()
        x_header = 'x'
        if proto == 'CW':
            x_header = 'x (GHz)'
        elif proto == 'Rabi':
            x_header = 'x (ns)'
        elif proto == 'Ramsey':
            x_header = 'x (ns)'
        elif proto == 'T1':
            x_header = 'x (ms)'
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, "w") as f:
                f.write(f"Results for {label}\n")
                if result is not None and hasattr(result, 'x') and hasattr(result, 'y'):
                    f.write(f"{x_header},y\n")
                    x = getattr(result, 'x')
                    y = getattr(result, 'y')
                    n = min(len(x), len(y))
                    for i in range(n):
                        f.write(f"{x[i]},{y[i]}\n")
                else:
                    messagebox.showerror("Save Error", "No x/y data found for this tab.")
                    return
            messagebox.showinfo("Save Results", f"Results saved to {file_path}")

    def load_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.loaded_label.config(text="Loading...")
            self.shape_label.config(text="Shape: -")
            threading.Thread(target=self._load_data_thread, args=(file_path,), daemon=True).start()

    def _load_data_thread(self, file_path):
        ext = os.path.splitext(file_path)[-1].lower()
        data = None
        try:
            if ext == ".npy":
                data = np.load(file_path)
            elif ext == ".npz":
                data = np.load(file_path)["arr_0"]
            elif ext == ".csv":
                data = np.loadtxt(file_path, delimiter=",")
            else:
                try:
                    from nd2reader import ND2Reader
                    if ext == ".nd2":
                        with ND2Reader(file_path) as images:
                            data = np.array([np.array(img) for img in images])
                except Exception:
                    data = None
        except Exception:
            data = None
        def update_ui():
            if data is None:
                self.data = None
                self.loaded_label.config(text=f"Loaded: <filename>")
                messagebox.showerror("Error", "Failed to load data or unsupported file type.")
                self.shape_label.config(text="Shape: -")
            else:
                self.data = data
                self.loaded_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                self.shape_label.config(text=f"Shape: {self.data.shape}")
                
                # Clear any existing preprocessed data
                if hasattr(self, 'preprocessed_cw_data'):
                    delattr(self, 'preprocessed_cw_data')
                if hasattr(self, 'cw_num_points'):
                    delattr(self, 'cw_num_points')
                if hasattr(self, 'rabi_pixel_traces'):
                    delattr(self, 'rabi_pixel_traces')
                if hasattr(self, 'rabi_num_points'):
                    delattr(self, 'rabi_num_points')
                if hasattr(self, 'rabi_x_range'):
                    delattr(self, 'rabi_x_range')
                
                # --- Optimized CW preprocessing ---
                if self.proto_cb.get() == "CW":
                    try:
                        # Guess num_points from UI or data shape
                        try:
                            start = float(self.param_entries["Start"].get())
                            end = float(self.param_entries["End"].get())
                            step = float(self.param_entries["Steps"].get())
                            x_range = np.arange(start, end + step, step)
                            num_points = len(x_range)
                        except Exception:
                            num_points = 0
                        if num_points > 0:
                            usable = self.data[3*num_points:]
                            num_averages = int(usable.shape[0] // num_points)
                            if num_averages * num_points == usable.shape[0]:
                                self.preprocessed_cw_data = preprocess_odmr_data(usable, num_averages, num_points)
                                self.cw_num_points = num_points
                            else:
                                self.preprocessed_cw_data = None
                        else:
                            self.preprocessed_cw_data = None
                    except Exception as e:
                        print("CW preprocessing failed:", e)
                        self.preprocessed_cw_data = None
        self.root.after(0, update_ui)

    def populate_result_tab(self, tab, proto):
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=5)
        tab.columnconfigure(2, weight=1)
        tab.rowconfigure(0, weight=1)
        tab.rowconfigure(1, weight=10)
        plot1 = ttk.LabelFrame(tab, text="ROI Intensity")
        plot1.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.create_sample_plot(plot1, "ROI Intensity", figsize=(3, 1.5))
        plot2 = ttk.LabelFrame(tab, text="Signal & Reference")
        plot2.grid(row=0, column=1, columnspan=2, sticky="nsew", padx=5, pady=5)
        self.create_sample_plot(plot2, "Raw / Normalized Signal", figsize=(6, 2.5))
        plot3 = ttk.LabelFrame(tab, text="Fitted Curve")
        plot3.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        self.create_sample_plot(plot3, "Fit Results", figsize=(6, 4))
        summary = ttk.LabelFrame(tab, text="Summary")
        summary.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)
        ttk.Label(summary, text="T1 = 3.4 ms\nContrast = 18.2%").pack(anchor="w")

    def populate_cw_result_tab(self, tab, result):
        # Clear any existing widgets
        for widget in tab.winfo_children():
            widget.destroy()
        # --- Layout ---
        # Top row: ROI (left, square) + Signal & Reference (right, fills)
        top_frame = ttk.Frame(tab)
        top_frame.pack(side="top", fill="x", padx=5, pady=5)
        # ROI Intensity (square)
        square_size = 180
        plot1 = ttk.LabelFrame(top_frame, text="ROI Intensity", width=square_size, height=square_size)
        plot1.pack(side="left", padx=0, pady=0)
        plot1.pack_propagate(False)
        self.create_result_plot(
            plot1,
            result.x,
            result.y,
            "ROI Intensity",
            figsize=(1.2, 1.2),
            image=getattr(result, 'image', None),
            roi_bounds=(getattr(result, 'x_min', None), getattr(result, 'x_max', None), getattr(result, 'y_min', None), getattr(result, 'y_max', None)),
            force_square=True
        )
        # Signal & Reference (fills remaining top row)
        plot2 = ttk.LabelFrame(top_frame, text="Signal & Reference")
        plot2.pack(side="left", fill="both", expand=True, padx=5, pady=0)
        fig2 = Figure(figsize=(6, 1.6), dpi=100)
        ax2 = fig2.add_subplot(111)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        ax2.tick_params(axis='both', which='minor', labelsize=7)
        ax2.yaxis.label.set_size(9)
        ax2.xaxis.label.set_size(9)
        ax2.yaxis.labelpad = 2
        ax2.xaxis.labelpad = 2
        fig2.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.28)
        has_reference = False
        if self.param_entries and "S+R" in self.param_entries:
            has_reference = self.param_entries["S+R"].get()
        if has_reference:
            # Only create and pack the Signal & Reference plot if S+R is checked
            if hasattr(result, 'mean_signal') and hasattr(result, 'mean_reference') and result.mean_signal is not None and result.mean_reference is not None:
                ax2.plot(result.x, result.mean_signal, color='r', marker='o', markersize=3, linestyle='-', label='Signal')
                ax2.plot(result.x, result.mean_reference, color='b', marker='o', markersize=3, linestyle='-', label='Reference')
                if hasattr(result, 'filtered_y') and result.filtered_y is not None:
                    ax2.plot(result.x, result.filtered_y, '--', color='gray', label='Filtered', linewidth=1)
                ax2.legend(fontsize=9, loc='best', frameon=False)
            else:
                ax2.plot(result.x, result.y, color='r', marker='o', markersize=3, linestyle='-', label='Signal')
                if hasattr(result, 'filtered_y') and result.filtered_y is not None:
                    ax2.plot(result.x, result.filtered_y, '--', color='gray', label='Filtered', linewidth=1)
                ax2.legend(fontsize=9, loc='best', frameon=False)
            canvas2 = FigureCanvasTkAgg(fig2, master=plot2)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # If has_reference is False, do not pack or show the plot2 widget at all (blank area)
        # Main row: left (Data), right (Summary, full height)
        main_row = ttk.Frame(tab)
        main_row.pack(side="top", fill="both", expand=True, padx=5, pady=(0,0))
        # Data plot
        data_frame = ttk.LabelFrame(main_row, text="Data")
        data_frame.pack(side="left", fill="both", expand=True, pady=(5,0))
        fig_data = Figure(figsize=(6, 2.6), dpi=100)
        ax_data = fig_data.add_subplot(111)
        ax_data.plot(result.x, result.y, 'ks', label="Raw Data", markersize=5, linestyle='None')
        if hasattr(result, 'filtered_y') and result.filtered_y is not None:
            ax_data.plot(result.x, result.filtered_y, '--', color='gray', label="Filtered", linewidth=1)
        ax_data.set_xlabel("Frequency (GHz)")
        ax_data.set_ylabel("Normalized Intensity")
        ax_data.set_title("")
        ax_data.legend()
        fig_data.tight_layout()
        canvas_data = FigureCanvasTkAgg(fig_data, master=data_frame)
        canvas_data.draw()
        canvas_data.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        add_toolbar(canvas_data, data_frame)
        # Summary (right, full height)
        summary = ttk.LabelFrame(main_row, text="Summary", width=320)
        summary.pack(side="left", fill="y", padx=(8,0))
        summary.pack_propagate(False)
        # --- Fit button logic for CW ---
        def run_cw_fit():
            from cw_pulsed_fit import process_odmr_fit
            x = result.x
            num_peaks = int(self.fit_param_entries["num_peaks"].get())
            threshold = float(self.fit_param_entries["threshold"].get())
            show_plot = self.fit_param_entries["show_plot"].get()
            use_filtered = self.fit_param_entries["use_filtered"].get()
            y = result.filtered_y if use_filtered else result.y
            try:
                popt, yfit = process_odmr_fit(x, y, num_peaks=num_peaks, threshold=threshold, plot=show_plot)
            except Exception as e:
                messagebox.showerror("Fit Error", f"CW fit failed: {e}")
                return
            # Store fit results for this tab
            idx = self.results_notebook.index(self.results_notebook.select())
            tab_widget, _ = self.result_tabs[idx]
            # Build parameter names for multi-peak Lorentzian
            param_names = []
            for i in range(num_peaks):
                param_names.extend([f"A{i+1}", f"x0{i+1}", f"gamma{i+1}"])
            self.tab_fit_results[tab_widget] = {
                "x": x,
                "y": y,
                "y_fit": 1 - yfit,
                "params": popt,
                "param_names": param_names,
                "equation": "y = 1 - sum_i A_i * exp(-((x - x0_i)^2) / (2 * gamma_i^2)) (multi-peak Lorentzian)"
            }
            # Update the Data plot in place (clear previous plot)
            for widget in data_frame.winfo_children():
                widget.destroy()
            fig_fit = Figure(figsize=(6, 2.6), dpi=100)
            ax_fit = fig_fit.add_subplot(111)
            ax_fit.plot(x, y, 'ks', label="Raw Data", markersize=5, linestyle='None')
            if hasattr(result, 'filtered_y') and result.filtered_y is not None:
                ax_fit.plot(x, result.filtered_y, '--', color='gray', label="Filtered", linewidth=1)
            ax_fit.plot(x, 1 - yfit, 'r-', label="Fit", linewidth=1.5)
            ax_fit.set_xlabel("Frequency (GHz)")
            ax_fit.set_ylabel("Normalized Intensity")
            ax_fit.set_title("")
            ax_fit.legend()
            fig_fit.tight_layout()
            canvas_fit = FigureCanvasTkAgg(fig_fit, master=data_frame)
            canvas_fit.draw()
            canvas_fit.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            add_toolbar(canvas_fit, data_frame)
            # Show fit parameters in summary
            fit_text = "Fit parameters:\n"
            for i in range(0, len(popt), 3):
                peak_num = i // 3 + 1
                amplitude = 1 - popt[i]
                center = popt[i+1]
                width = popt[i+2]
                fit_text += f"Peak {peak_num}:\n  amplitude: {amplitude:.4g}\n  center: {center:.4g} MHz\n  width: {width:.4g} MHz\n\n"
            for widget in summary.winfo_children():
                widget.destroy()
            ttk.Label(summary, text=fit_text, justify="left").pack(anchor="nw")
        # Wire up the fit button
        self.fit_param_entries["fit_btn"].config(command=run_cw_fit)
        # Initial summary
        for widget in summary.winfo_children():
            widget.destroy()
        ttk.Label(summary, text="CW analysis complete.", justify="left").pack(anchor="nw")

    def populate_rabi_result_tab(self, tab, result):
        # Clear any existing widgets
        for widget in tab.winfo_children():
            widget.destroy()
        # --- Layout ---
        # Top row: ROI (left, square) + Signal & Reference (right, fills)
        top_frame = ttk.Frame(tab)
        top_frame.pack(side="top", fill="x", padx=5, pady=5)
        # ROI Intensity (square)
        square_size = 180
        plot1 = ttk.LabelFrame(top_frame, text="ROI Intensity", width=square_size, height=square_size)
        plot1.pack(side="left", padx=0, pady=0)
        plot1.pack_propagate(False)
        self.create_result_plot(plot1, result.x, result.y, "ROI Intensity", figsize=(1.2, 1.2), image=getattr(result, 'image', None), roi_bounds=getattr(result, 'roi_bounds', None), force_square=True)
        # Signal & Reference (fills remaining top row)
        plot2 = ttk.LabelFrame(top_frame, text="Signal & Reference")
        plot2.pack(side="left", fill="both", expand=True, padx=5, pady=0)
        fig2 = Figure(figsize=(6, 1.6), dpi=100)
        ax2 = fig2.add_subplot(111)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        ax2.tick_params(axis='both', which='minor', labelsize=7)
        ax2.yaxis.label.set_size(9)
        ax2.xaxis.label.set_size(9)
        ax2.yaxis.labelpad = 2
        ax2.xaxis.labelpad = 2
        fig2.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.28)
        if hasattr(result, 'mean_signal') and hasattr(result, 'mean_reference') and result.mean_signal is not None and result.mean_reference is not None:
            ax2.plot(result.x, result.mean_signal, color='r', marker='o', markersize=3, linestyle='-', label='Signal')
            ax2.plot(result.x, result.mean_reference, color='b', marker='o', markersize=3, linestyle='-', label='Reference')
        else:
            ax2.plot(result.x, result.y, color='r', marker='o', markersize=3, linestyle='-', label='Signal')
        ax2.set_xlabel('Rabi time (ns)')
        ax2.set_ylabel('Intensity')
        ax2.legend(fontsize=9, loc='best', frameon=False)
        canvas2 = FigureCanvasTkAgg(fig2, master=plot2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Main row: left (Data), right (Summary, full height)
        main_row = ttk.Frame(tab)
        main_row.pack(side="top", fill="both", expand=True, padx=5, pady=(0,0))
        # Data plot
        data_frame = ttk.LabelFrame(main_row, text="Data")
        data_frame.pack(side="left", fill="both", expand=True, pady=(5,0))
        fig_data = Figure(figsize=(6, 2.6), dpi=100)
        ax_data = fig_data.add_subplot(111)
        ax_data.plot(result.x, result.y, 'ks', label="Raw Data", markersize=3, linestyle='None')
        ax_data.set_xlabel('time (ns)')
        ax_data.set_ylabel('Intensity')
        ax_data.set_title("")
        ax_data.legend()
        fig_data.tight_layout()
        canvas_data = FigureCanvasTkAgg(fig_data, master=data_frame)
        canvas_data.draw()
        canvas_data.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        add_toolbar(canvas_data, data_frame)
        # Summary (right, full height)
        summary = ttk.LabelFrame(main_row, text="Summary", width=320)
        summary.pack(side="left", fill="y", padx=(8,0))
        summary.pack_propagate(False)
        contrast = (np.max(result.y) - np.min(result.y)) / np.max(result.y) * 100 if np.max(result.y) != 0 else 0
        mean_intensity = np.mean(result.y)
        summary_text = f"Rabi analysis complete.\nContrast: {contrast:.2f}%\nMean Intensity: {mean_intensity:.3g}"
        for widget in summary.winfo_children():
            widget.destroy()
        ttk.Label(summary, text=summary_text, justify="left").pack(anchor="nw")
        # --- Fit button logic for Rabi ---
        def run_rabi_fit():
            from scipy.optimize import curve_fit
            proto = self.proto_cb.get()
            if proto == "Ramsey":
                # Should not happen here, but keep for completeness
                        return
            else:
                # Rabi fit as before
                A = float(self.fit_param_entries["Amplitude"].get())
                omega = float(self.fit_param_entries["Omega"].get())
                phase = float(self.fit_param_entries["Phase"].get())
                decay = float(self.fit_param_entries["Decay Rate"].get())
                B = float(self.fit_param_entries["B"].get())
                C = float(self.fit_param_entries["C"].get())
                p0 = [A, omega, phase, decay, B, C]
                try:
                    popt, pcov = curve_fit(self.rabi_model, result.x, result.y, p0=p0, maxfev=10000)
                    yfit = self.rabi_model(result.x, *popt)
                except Exception as e:
                    messagebox.showerror("Fit Error", f"Rabi fit failed: {e}")
                    return
                # Store fit results for this tab
                idx = self.results_notebook.index(self.results_notebook.select())
                tab_widget, _ = self.result_tabs[idx]
                param_names = ["A", "omega", "phase", "decay", "B", "C"]
                self.tab_fit_results[tab_widget] = {
                    "x": result.x,
                    "y": result.y,
                    "y_fit": yfit,
                    "params": popt,
                    "param_names": param_names,
                    "equation": "y = A * cos(omega * x + phase) * exp(-x / decay) + B * x + C (Rabi)"
                }
                for widget in data_frame.winfo_children():
                    widget.destroy()
                fig_fit = Figure(figsize=(6, 2.6), dpi=100)
                ax_fit = fig_fit.add_subplot(111)
                ax_fit.plot(result.x, result.y, 'ks', label="Raw Data", markersize=3, linestyle='None')
                ax_fit.plot(result.x, yfit, 'r-', linewidth=1.5, label='Fit')
                ax_fit.set_xlabel('time (ns)')
                ax_fit.set_ylabel('Intensity')
                ax_fit.set_title("")
                ax_fit.legend()
                fig_fit.tight_layout()
                canvas_fit = FigureCanvasTkAgg(fig_fit, master=data_frame)
                canvas_fit.draw()
                canvas_fit.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                add_toolbar(canvas_fit, data_frame)
                # Show fit parameters in summary
                fit_text = "Fit parameters:\n"
                param_names = ["A", "omega", "phase", "decay", "B", "C"]
                for name, val in zip(param_names, popt):
                    fit_text += f"{name}: {val:.4g}\n"
                for widget in summary.winfo_children():
                    widget.destroy()
                ttk.Label(summary, text=fit_text, justify="left").pack(anchor="nw")
        # Wire up the fit button
        self.fit_param_entries["fit_btn"].config(command=run_rabi_fit)

    def populate_ramsey_result_tab(self, tab, result):
        # Clear any existing widgets
        for widget in tab.winfo_children():
            widget.destroy()
        # --- Layout ---
        # Top row: ROI (left, square) + Signal & Reference (right, fills)
        top_frame = ttk.Frame(tab)
        top_frame.pack(side="top", fill="x", padx=5, pady=5)
        # ROI Intensity (square)
        square_size = 180
        plot1 = ttk.LabelFrame(top_frame, text="ROI Intensity", width=square_size, height=square_size)
        plot1.pack(side="left", padx=0, pady=0)
        plot1.pack_propagate(False)
        self.create_result_plot(plot1, result.x, result.y, "ROI Intensity", figsize=(1.2, 1.2), image=getattr(result, 'image', None), roi_bounds=getattr(result, 'roi_bounds', None), force_square=True)
        # Signal & Reference (fills remaining top row)
        plot2 = ttk.LabelFrame(top_frame, text="Signal & Reference")
        plot2.pack(side="left", fill="both", expand=True, padx=5, pady=0)
        fig2 = Figure(figsize=(6, 1.6), dpi=100)  # Keep short height
        ax2 = fig2.add_subplot(111)
        # --- Polished axis: smaller ticks, tighter layout, less label padding ---
        ax2.tick_params(axis='both', which='major', labelsize=8)
        ax2.tick_params(axis='both', which='minor', labelsize=7)
        ax2.yaxis.label.set_size(9)
        ax2.xaxis.label.set_size(9)
        ax2.yaxis.labelpad = 2
        ax2.xaxis.labelpad = 2
        fig2.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.28)  # Increased bottom for x-label
        # Use mean_signal and mean_reference if present, else just y
        if hasattr(result, 'mean_signal') and hasattr(result, 'mean_reference') and result.mean_signal is not None and result.mean_reference is not None:
            ax2.plot(result.x, result.mean_signal, color='r', marker='o', markersize=3, linestyle='-', label='Signal')
            ax2.plot(result.x, result.mean_reference, color='b', marker='o', markersize=3, linestyle='-', label='Reference')
        else:
            ax2.plot(result.x, result.y, color='r', marker='o', markersize=3, linestyle='-', label='Signal')
        ax2.set_xlabel('Ramsey time (ns)')
        ax2.set_ylabel('Intensity')
        ax2.legend(fontsize=9, loc='best', frameon=False)
        canvas2 = FigureCanvasTkAgg(fig2, master=plot2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Main row: left (FFT + Time Domain, vertical), right (Summary, full height)
        main_row = ttk.Frame(tab)
        main_row.pack(side="top", fill="both", expand=True, padx=5, pady=(0,0))
        # Left column: FFT + Time Domain (vertical stack)
        left_col = ttk.Frame(main_row)
        left_col.pack(side="left", fill="both", expand=False)
        # FFT
        fft_frame = ttk.LabelFrame(left_col, text="FFT")
        fft_frame.pack(side="top", fill="x", expand=False)
        import numpy as np
        from scipy.interpolate import interp1d
        x_ns = np.array(result.x)
        y = np.array(result.y)
        x_s = x_ns * 1e-9
        uniform_time = np.linspace(x_s.min(), x_s.max(), len(x_s))
        interp_func = interp1d(x_s, y, kind='cubic')
        uniform_signal = interp_func(uniform_time)
        norm_signal = (uniform_signal - np.min(uniform_signal)) / (np.max(uniform_signal) - np.min(uniform_signal))
        dt = uniform_time[1] - uniform_time[0]
        N = len(uniform_time)
        freq = np.fft.fftfreq(N, d=dt)
        fft_vals = np.fft.fft(norm_signal)
        mask = freq > 0
        fig_fft = Figure(figsize=(6, 1.6), dpi=100)  # Increased height
        ax_fft = fig_fft.add_subplot(111)
        # --- Polished axis: smaller ticks, tighter layout, less label padding ---
        ax_fft.tick_params(axis='both', which='major', labelsize=8)
        ax_fft.tick_params(axis='both', which='minor', labelsize=7)
        ax_fft.yaxis.label.set_size(9)
        ax_fft.xaxis.label.set_size(9)
        ax_fft.yaxis.labelpad = 2
        ax_fft.xaxis.labelpad = 2
        fig_fft.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.28)
        ax_fft.plot(freq[mask]*1e-6, np.abs(fft_vals[mask]), color='tab:green')
        ax_fft.set_xlabel('Frequency (MHz)')
        ax_fft.set_ylabel('FFT Amplitude')
        ax_fft.set_title('')
        fig_fft.tight_layout()
        canvas_fft = FigureCanvasTkAgg(fig_fft, master=fft_frame)
        canvas_fft.draw()
        canvas_fft.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Time Domain
        time_frame = ttk.LabelFrame(left_col, text="Time Domain")
        time_frame.pack(side="top", fill="both", expand=True, pady=(5,0))
        # Destroy all children before adding new plot and toolbar
        for child in time_frame.winfo_children():
            child.destroy()
        fig_time = Figure(figsize=(6, 2.6), dpi=100)  # Increased width/height
        ax_time = fig_time.add_subplot(111)
        ax_time.plot(uniform_time*1e9, norm_signal, 'ks', label="Raw Data", markersize=3, linestyle='None')
        ax_time.set_xlabel('time (ns)')
        ax_time.set_ylabel('Normalized Intensity')
        ax_time.set_title("")
        ax_time.legend()
        fig_time.tight_layout()
        canvas_time = FigureCanvasTkAgg(fig_time, master=time_frame)
        canvas_time.draw()
        canvas_time.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        print("Adding toolbar to Ramsey Time Domain plot (initial)")
        try:
            add_toolbar(canvas_time, time_frame)
        except Exception as e:
            print(f"Toolbar error: {e}")
        # Right column: Summary (spans both FFT and Time Domain)
        summary = ttk.LabelFrame(main_row, text="Summary", width=320)
        summary.pack(side="left", fill="y", padx=(8,0))
        summary.pack_propagate(False)
        contrast = (np.max(result.y) - np.min(result.y)) / np.max(result.y) * 100 if np.max(result.y) != 0 else 0
        mean_intensity = np.mean(result.y)
        summary_text = f"Ramsey analysis complete.\nContrast: {contrast:.2f}%\nMean Intensity: {mean_intensity:.3g}"
        ttk.Label(summary, text=summary_text, justify="left").pack(anchor="nw")

        # --- Fit button logic for Ramsey ---
        def run_ramsey_fit():
            import numpy as np
            from scipy.optimize import curve_fit
            from scipy.interpolate import interp1d
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            # Prepare data
            model = self.fit_param_entries["model"].get()
            x_ns = np.array(result.x)
            y = np.array(result.y)
            x_s = x_ns * 1e-9
            uniform_time = np.linspace(x_s.min(), x_s.max(), len(x_s))
            interp_func = interp1d(x_s, y, kind='cubic')
            uniform_signal = interp_func(uniform_time)
            norm_signal = (uniform_signal - np.min(uniform_signal)) / (np.max(uniform_signal) - np.min(uniform_signal))
            # Fit
            fit_text = "Fit parameters:\n"
            try:
                if model == "Single Cosine":
                    param_names = ["A", "B", "T", "beta", "f", "phi", "y0"]
                    p0 = [float(self.fit_param_entries[label].get()) for label in param_names]
                    popt, pcov = curve_fit(exp_decay_single_cos, uniform_time, norm_signal, p0=p0, maxfev=20000)
                    yfit = exp_decay_single_cos(uniform_time, *popt)
                    equation = "y = B*t + A*exp(-t/T-beta*t^2)*cos(2*pi*f*t+phi)+y0 (Single Cosine)"
                else:
                    param_names = ["A1", "T1", "f1", "phi1", "A2", "T2", "f2", "phi2", "y0"]
                    p0 = [float(self.fit_param_entries[label].get()) for label in param_names]
                    popt, pcov = curve_fit(double_exp_decay_cosine, uniform_time, norm_signal, p0=p0, maxfev=20000)
                    yfit = double_exp_decay_cosine(uniform_time, *popt)
                    equation = "y = A1*exp(-(t/T1)^2)*cos(2*pi*f1*t+phi1) + A2*exp(-(t/T2)^2)*cos(2*pi*f2*t+phi2) + y0 (Double Cosine)"
                for name, val in zip(param_names, popt):
                    fit_text += f"{name}: {val:.4g}\n"
                # Store fit results for this tab
                idx = self.results_notebook.index(self.results_notebook.select())
                tab_widget, _ = self.result_tabs[idx]
                self.tab_fit_results[tab_widget] = {
                    "x": uniform_time,
                    "y": norm_signal,
                    "y_fit": yfit,
                    "params": popt,
                    "param_names": param_names,
                    "equation": equation
                }
            except Exception as e:
                from tkinter import messagebox
                messagebox.showerror("Fit Error", f"Ramsey fit failed: {e}")
                return
            # FFT peak calculation
            dt = uniform_time[1] - uniform_time[0]
            N = len(uniform_time)
            freq = np.fft.fftfreq(N, d=dt)
            fft_vals = np.fft.fft(norm_signal)
            mask = freq > 0
            abs_fft = np.abs(fft_vals[mask])
            freq_mhz = freq[mask] * 1e-6
            if len(abs_fft) > 0:
                peak_idx = np.argmax(abs_fft)
                peak_freq = freq_mhz[peak_idx]
                peak_amp = abs_fft[peak_idx]
                fit_text += f"\nFFT peak: {peak_freq:.2f} MHz (ampl: {peak_amp:.2f})\n"
            # Update the Time Domain plot (clear and redraw)
            for widget in time_frame.winfo_children():
                widget.destroy()
            fig_time = Figure(figsize=(6, 2.6), dpi=100)
            ax_time = fig_time.add_subplot(111)
            ax_time.plot(uniform_time*1e9, norm_signal, 'ks', label="Raw Data", markersize=3, linestyle='None')
            ax_time.plot(uniform_time*1e9, yfit, 'r-', linewidth=1.5, label='Fit')
            ax_time.set_xlabel('time (ns)')
            ax_time.set_ylabel('Normalized Intensity')
            ax_time.set_title("")
            ax_time.legend()
            fig_time.tight_layout()
            canvas_time = FigureCanvasTkAgg(fig_time, master=time_frame)
            canvas_time.draw()
            canvas_time.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            add_toolbar(canvas_time, time_frame)
            # Update summary with fit parameters and FFT peak
            for widget in summary.winfo_children():
                widget.destroy()
            ttk.Label(summary, text=fit_text, justify="left").pack(anchor="nw")
        # Wire up the fit button for Ramsey
        if hasattr(self, 'fit_param_entries') and "fit_btn" in self.fit_param_entries:
            try:
                self.fit_param_entries["fit_btn"].config(command=run_ramsey_fit)
            except Exception:
                pass

    def populate_t1_result_tab(self, tab, result):
        # Clear any existing widgets
        for widget in tab.winfo_children():
            widget.destroy()
        # --- Layout ---
        # Top row: ROI (left, square) + Signal & Reference (right, fills)
        top_frame = ttk.Frame(tab)
        top_frame.pack(side="top", fill="x", padx=5, pady=5)
        # ROI Intensity (square)
        square_size = 180
        plot1 = ttk.LabelFrame(top_frame, text="ROI Intensity", width=square_size, height=square_size)
        plot1.pack(side="left", padx=0, pady=0)
        plot1.pack_propagate(False)
        self.create_result_plot(plot1, result.x, result.y, "ROI Intensity", figsize=(1.2, 1.2), image=getattr(result, 'image', None), roi_bounds=getattr(result, 'roi_bounds', None), force_square=True)
        # Signal & Reference (fills remaining top row)
        plot2 = ttk.LabelFrame(top_frame, text="Signal & Reference")
        plot2.pack(side="left", fill="both", expand=True, padx=5, pady=0)
        fig2 = Figure(figsize=(6, 1.6), dpi=100)
        ax2 = fig2.add_subplot(111)
        # --- Polished axis: smaller ticks, tighter layout, less label padding ---
        ax2.tick_params(axis='both', which='major', labelsize=8)
        ax2.tick_params(axis='both', which='minor', labelsize=7)
        ax2.yaxis.label.set_size(9)
        ax2.xaxis.label.set_size(9)
        ax2.yaxis.labelpad = 2
        ax2.xaxis.labelpad = 2
        fig2.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.28)
        mean_signal = getattr(result, 'mean_signal', None)
        mean_reference = getattr(result, 'mean_reference', None)
        if mean_signal is not None and mean_reference is not None:
            ax2.plot(result.x, mean_signal, color='r', marker='o', markersize=3, linestyle='-', label='Signal')
            ax2.plot(result.x, mean_reference, color='b', marker='o', markersize=3, linestyle='-', label='Reference')
        else:
            ax2.plot(result.x, result.y, color='r', marker='o', markersize=3, linestyle='-', label='Signal')
        ax2.set_xlabel('T1 time (ms)')
        ax2.set_ylabel('Intensity')
        ax2.legend(fontsize=9, loc='best', frameon=False)
        canvas2 = FigureCanvasTkAgg(fig2, master=plot2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Main row: left (Time Domain), right (Summary, full height)
        main_row = ttk.Frame(tab)
        main_row.pack(side="top", fill="both", expand=True, padx=5, pady=(0,0))
        # Time Domain
        time_frame = ttk.LabelFrame(main_row, text="Time Domain")
        time_frame.pack(side="left", fill="both", expand=True, pady=(5,0))
        # Destroy all children before adding new plot and toolbar
        for child in time_frame.winfo_children():
            child.destroy()
        fig_time = Figure(figsize=(6, 2.6), dpi=100)  # Increased width/height
        ax_time = fig_time.add_subplot(111)
        ax_time.plot(result.x, result.y, 'ks', label="Raw Data", markersize=3, linestyle='None')
        ax_time.set_xlabel('time (ms)')
        ax_time.set_ylabel('Intensity')
        ax_time.set_title("")
        ax_time.legend()
        fig_time.tight_layout()
        canvas_time = FigureCanvasTkAgg(fig_time, master=time_frame)
        canvas_time.draw()
        canvas_time.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        print("Adding toolbar to T1 Time Domain plot (initial)")
        try:
            add_toolbar(canvas_time, time_frame)
        except Exception as e:
            print(f"Toolbar error: {e}")
        # Right column: Summary (spans full height)
        summary = ttk.LabelFrame(main_row, text="Summary", width=320)
        summary.pack(side="left", fill="y", padx=(8,0))
        summary.pack_propagate(False)
        contrast = (np.max(result.y) - np.min(result.y)) / np.max(result.y) * 100 if np.max(result.y) != 0 else 0
        mean_intensity = np.mean(result.y)
        summary_text = f"T1 analysis complete.\nContrast: {contrast:.2f}%\nMean Intensity: {mean_intensity:.3g}"
        ttk.Label(summary, text=summary_text, justify="left").pack(anchor="nw")

        # --- Fit button logic for T1 ---
        def run_t1_fit():
            import numpy as np
            from scipy.optimize import curve_fit
            # Stretched exponential: y = a * exp(-((t / T1)^beta)) + c
            def stretched_exp_decay(x, a, T1, beta, c):
                return a * np.exp(-((x / T1) ** beta)) + c
            x = np.array(result.x)
            y = np.array(result.y)
            yerr = getattr(result, 'error', None)
            # Initial guess: [a, T1, beta, c] from GUI fields
            try:
                p0 = [float(self.fit_param_entries[name].get()) for name in ["Amplitude", "T1", "Beta", "Offset"]]
            except Exception:
                p0 = [0.5, 2.0, 0.8, 0.1]
            try:
                if yerr is not None:
                    popt, pcov = curve_fit(stretched_exp_decay, x, y, p0=p0, sigma=yerr, absolute_sigma=True, maxfev=10000)
                else:
                    popt, pcov = curve_fit(stretched_exp_decay, x, y, p0=p0, maxfev=10000)
                yfit = stretched_exp_decay(x, *popt)
                # Store fit results for this tab
                idx = self.results_notebook.index(self.results_notebook.select())
                tab_widget, _ = self.result_tabs[idx]
                param_names = ["a", "T1", "beta", "c"]
                self.tab_fit_results[tab_widget] = {
                    "x": x,
                    "y": y,
                    "y_fit": yfit,
                    "params": popt,
                    "param_names": param_names,
                    "equation": "y = a * exp(-((t / T1)^beta)) + c (Stretched Exponential)"
                }
            except Exception as e:
                from tkinter import messagebox
                messagebox.showerror("Fit Error", f"T1 fit failed: {e}")
                return
            # Update the Time Domain plot (clear and redraw)
            for widget in time_frame.winfo_children():
                widget.destroy()
            fig_time = Figure(figsize=(6, 2.6), dpi=100)
            ax_time = fig_time.add_subplot(111)
            ax_time.errorbar(x, y, yerr=yerr, fmt='ks', label="Data", markersize=3, linestyle='None')
            ax_time.plot(x, yfit, 'r-', linewidth=1.5, label='Fit')
            ax_time.set_xlabel('time (ms)')
            ax_time.set_ylabel('Normalized Intensity')
            ax_time.set_title("")
            ax_time.legend()
            fig_time.tight_layout()
            canvas_time = FigureCanvasTkAgg(fig_time, master=time_frame)
            canvas_time.draw()
            canvas_time.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            add_toolbar(canvas_time, time_frame)
            # Update summary with fit parameters
            fit_text = f"Fit parameters:\na: {popt[0]:.4g}\nT1: {popt[1]:.4g} ms\nbeta: {popt[2]:.4g}\nc: {popt[3]:.4g}\n"
            for widget in summary.winfo_children():
                widget.destroy()
            ttk.Label(summary, text=fit_text, justify="left").pack(anchor="nw")
        # Wire up the fit button for T1
        if hasattr(self, 'fit_param_entries') and "fit_btn" in self.fit_param_entries:
            try:
                self.fit_param_entries["fit_btn"].config(command=run_t1_fit)
            except Exception:
                pass

    def create_sample_plot(self, frame, title, figsize=(4, 2.5)):
        fig = Figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        ax.plot([0, 1, 2, 3], [10, 20, 10, 30])
        ax.set_title(title)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_result_plot(self, frame, x, y, title, figsize=(4, 2.5), image=None, roi_bounds=None, filtered=None, fit_curve=None, force_square=False):
        # For ROI Intensity, use a larger square figure and fill the frame
        if title == "ROI Intensity":
            figsize = (1.8, 1.8)  # 1.8in * 100dpi = 180px
        fig = Figure(figsize=figsize, dpi=100, facecolor='none')
        ax = fig.add_subplot(111, facecolor='none')
        if title == "ROI Intensity" and image is not None and roi_bounds is not None:
            from analysis_utils import plot_roi_on_image
            plot_roi_on_image(image, *roi_bounds, ax=ax)
            ax.set_aspect('equal')  # force square aspect
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(image.shape[0], 0)
            ax.axis('off')  # Hide all axes, ticks, and labels
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.tight_layout(pad=0)
        if title == "ROI Intensity" or force_square:
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(fill=tk.BOTH, expand=True)
            widget.configure(width=180, height=180)
            return
        elif title == "Data":
            # Plot raw data as points, filtered as a neutral dashed line
            ax.plot(x, y, 'ks', label="Raw Data", markersize=5, linestyle='None')
            if filtered is not None:
                ax.plot(x, filtered, '--', color='gray', label="Filtered", linewidth=1)
            if fit_curve is not None:
                ax.plot(x, 1 - fit_curve, 'r-', label="Fit", linewidth=1.5)
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Normalized Intensity")
            ax.legend()
            ax.set_title(title)
        else:
            ax.plot(x, y)
            ax.set_title(title)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        add_toolbar(canvas, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

    def rabi_model(self, x, A, omega, phase, decay, B, C):
        return A * np.cos(omega * x + phase) * np.exp(-x / decay) + B * x + C

    def save_fit_results(self):
        # Save x, y, y_fit, fit parameters, and equation for the current tab
        idx = self.results_notebook.index(self.results_notebook.select())
        tab, label = self.result_tabs[idx]
        fit_result = self.tab_fit_results.get(tab, None)
        if fit_result is None:
            messagebox.showerror("Save Error", "No fit results available for this tab. Please run a fit first.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, "w") as f:
                f.write(f"Fit results for {label}\n")
                f.write(f"Equation: {fit_result['equation']}\n")
                # Write parameter names and values as header rows
                f.write(",".join(fit_result["param_names"]) + ",,,\n")
                f.write(",".join([str(p) for p in fit_result["params"]]) + ",,,\n")
                f.write("x,y,y_fit\n")
                n = min(len(fit_result['x']), len(fit_result['y']), len(fit_result['y_fit']))
                for i in range(n):
                    f.write(f"{fit_result['x'][i]},{fit_result['y'][i]},{fit_result['y_fit'][i]}\n")
            messagebox.showinfo("Save Fit Results", f"Fit results saved to {file_path}")

    def _close_tab(self, index):
        print(f"Closing tab at index {index}")  # Debug print
        tab_id = self.results_notebook.tabs()[index]
        self.results_notebook.forget(tab_id)
        if index < len(self.result_tabs):
            tab_widget, _ = self.result_tabs[index]
            if tab_widget in self.tab_results:
                del self.tab_results[tab_widget]
            if tab_widget in self.tab_fit_results:
                del self.tab_fit_results[tab_widget]
            del self.result_tabs[index]

    def close_current_tab(self):
        current = self.results_notebook.select()
        if current:
            index = self.results_notebook.index(current)
            self._close_tab(index)

    def preprocess_rabi_data(self, data: np.ndarray, num_averages: int, num_points: int) -> np.ndarray:
        """
        Preprocess Rabi data to compute normalized traces for all pixels.
        Returns a 3D array of shape (num_points, y, x) containing the normalized Rabi traces.
        """
        print(f"[Rabi] Preprocessing data with shape: {data.shape}")
        # Split data into signal and reference (alternating frames)
        signal, reference = split_signal_reference(data)
        # Reshape to (num_points, num_averages, y, x)
        signal = signal.reshape(num_points, num_averages, signal.shape[1], signal.shape[2])
        reference = reference.reshape(num_points, num_averages, reference.shape[1], reference.shape[2])
        # Average over num_averages for each pixel
        mean_signal = np.mean(signal, axis=1)  # (num_points, y, x)
        mean_reference = np.mean(reference, axis=1)  # (num_points, y, x)
        # Normalize for each pixel
        normalized_traces = mean_signal / mean_reference  # (num_points, y, x)
        print(f"[Rabi] Preprocessing complete. Normalized traces shape: {normalized_traces.shape}")
        return normalized_traces

    def extract_rabi_roi_trace(self, x_min: int, x_max: int, y_min: int, y_max: int) -> np.ndarray:
        """
        Extract and average the Rabi trace from a specific ROI using preprocessed data.
        """
        if not hasattr(self, 'rabi_pixel_traces'):
            raise ValueError("Rabi data not preprocessed. Run analysis first.")
        
        # Extract ROI from preprocessed data
        roi_traces = self.rabi_pixel_traces[:, y_min:y_max, x_min:x_max]  # Shape: (num_points, roi_y, roi_x)
        
        # Average over spatial dimensions
        rabi_trace = np.mean(roi_traces, axis=(1, 2))  # Shape: (num_points,)
        
        return rabi_trace

    def preprocess_ramsey_data(self, data: np.ndarray, num_averages: int, num_points: int) -> np.ndarray:
        """
        Preprocess Ramsey data to compute normalized traces for all pixels.
        Returns a 3D array of shape (num_points, y, x) containing the normalized Ramsey traces.
        """
        print(f"[Ramsey] Preprocessing data with shape: {data.shape}")
        # Split data into signal and reference (alternating frames)
        signal, reference = split_signal_reference(data)
        # Reshape to (num_points, num_averages, y, x)
        signal = signal.reshape(num_points, num_averages, signal.shape[1], signal.shape[2])
        reference = reference.reshape(num_points, num_averages, reference.shape[1], reference.shape[2])
        # Average over num_averages for each pixel
        mean_signal = np.mean(signal, axis=1)  # (num_points, y, x)
        mean_reference = np.mean(reference, axis=1)  # (num_points, y, x)
        # Normalize for each pixel
        normalized_traces = mean_signal / mean_reference  # (num_points, y, x)
        print(f"[Ramsey] Preprocessing complete. Normalized traces shape: {normalized_traces.shape}")
        return normalized_traces

    def extract_ramsey_roi_trace(self, x_min: int, x_max: int, y_min: int, y_max: int) -> np.ndarray:
        """
        Extract and average the Ramsey trace from a specific ROI using preprocessed data.
        """
        if not hasattr(self, 'ramsey_pixel_traces'):
            raise ValueError("Ramsey data not preprocessed. Run analysis first.")
        roi_traces = self.ramsey_pixel_traces[:, y_min:y_max, x_min:x_max]  # (num_points, roi_y, roi_x)
        ramsey_trace = np.mean(roi_traces, axis=(1, 2))  # (num_points,)
        return ramsey_trace

# Helper to add the matplotlib navigation toolbar below a plot
def add_toolbar(canvas, frame):
    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.update()
    toolbar.pack(side=tk.TOP, fill=tk.X)

if __name__ == "__main__":
    root = tk.Tk()
    app = NVAnalysisApp(root)
    root.mainloop()