from tkinter import messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
from odmr_analy import odmr_analyze_data
from rabi_analy import rabi_analyze_data
import matplotlib.pyplot as plt
import threading
from analysis_utils import preprocess_odmr_data, extract_roi_trace, split_signal_reference
import csv
import numpy as np
import tkinter as tk
from tkinter import ttk


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
        entry.grid(row=row, column=col + 1, padx=5, pady=2)
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


def double_exp_decay_cosine(t, A1, T1, f1, phi1, A2, T2, f2, phi2, y0):
    return A1 * np.exp(-t / T1) ** 2 * np.cos(2 * np.pi * f1 * t + phi1) + \
        A2 * np.exp(-t / T2) ** 2 * np.cos(2 * np.pi * f2 * t + phi2) + y0


def exp_decay_single_cos(t, A, B, T, beta, f, phi, y0):
    return B * t + A * np.exp(-t / T - beta * t ** 2) * np.cos(2 * np.pi * f * t + phi) + y0


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
    "Phase": "ϕ",
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
        model_cb = ttk.Combobox(frame, textvariable=model_var, values=["Single Cosine", "Double Cosine"],
                                state="readonly", width=15)
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
                if label_text is None:
                    label_text = str(label)
                ttk.Label(param_frame, text=label_text).grid(row=i // 2, column=(i % 2) * 2, sticky="ew", padx=2,
                                                             pady=2)
                var = tk.StringVar(value=str(default))
                entry = ttk.Entry(param_frame, textvariable=var, width=8)
                entry.grid(row=i // 2, column=(i % 2) * 2 + 1, sticky="ew", padx=2, pady=2)
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
        label_text = GREEK_LABELS.get(param["label"], param["label"])
        if label_text is None:
            label_text = str(param["label"])
        ttk.Label(frame, text=label_text).grid(row=row, column=col * 2, sticky="ew", padx=2, pady=2)
        if param.get("type") == "bool":
            var = tk.BooleanVar(value=param.get("default", False))
            cb = ttk.Checkbutton(frame, variable=var)
            cb.grid(row=row, column=col * 2 + 1, sticky="ew", padx=2, pady=2)
            entries[param["name"]] = var
        else:
            var = tk.StringVar(value=str(param.get("default", "")))
            entry_width = 5 if param["name"] in ("num_peaks", "threshold") else 8
            entry = ttk.Entry(frame, textvariable=var, width=entry_width)
            entry.grid(row=row, column=col * 2 + 1, sticky="ew", padx=2, pady=2)
            entries[param["name"]] = var
        if col == 1:
            row += 1
            col = 0
        else:
            col = 1
    # Fit and Save Fit Results buttons side by side
    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=row + 1, column=0, columnspan=4, sticky="ew", padx=2, pady=(5, 0))
    entries["fit_btn"] = ttk.Button(btn_frame, text="Run Fit")
    entries["fit_btn"].pack(side="left", expand=True, fill=tk.X, padx=(0, 2))
    entries["save_fit_btn"] = ttk.Button(btn_frame, text="Save Fit Results")
    entries["save_fit_btn"].pack(side="left", expand=True, fill=tk.X, padx=(2, 0))
    return frame, entries


class NVAnalysisApp:
    def __init__(self, root):
        print("DEBUG: NVAnalysisApp __init__ called")
        self.root = root
        self.root.title("NV Protocol Analysis (by Orlando Cunha)")
        self.root.geometry("1200x700")
        self.root.state('zoomed')
        self.protocols = ["CW", "Rabi", "Ramsey", "T1"]
        self.fit_param_configs = {
            "CW": [
                {"name": "num_peaks", "label": "Number of Peaks", "type": "int", "default": 4},
                {"name": "threshold", "label": "Threshold", "type": "float", "default": 0.974},
                {"name": "lookahead", "label": "Lookahead", "type": "int", "default": 4},
                {"name": "show_plot", "label": "Intermediate plot", "type": "bool", "default": False},
                {"name": "show_peaks", "label": "Show peaks", "type": "bool", "default": False},
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
        self.data = None  # Ensure self.data always exists
        self.build_ui()
        self.fit_animating = False
        self.fit_anim_progress = 0
        self.fit_anim_done = False

    def _bind_roi_param_tab_order(self):
        # Helper to robustly bind Tab/Shift-Tab between ROI and Experimental Params
        roi_entries = list(self.roi_entries.values())
        param_entries = [e for k, e in self.param_entries.items() if
                         hasattr(e, 'winfo_exists') and e.winfo_exists() and isinstance(e, tk.Entry)]
        for entry in roi_entries + param_entries:
            entry.configure(takefocus=True)
        # Remove previous bindings to avoid stacking
        if roi_entries and param_entries:
            last_roi = roi_entries[-1]
            first_param = param_entries[0]
            try:
                last_roi.unbind("<Tab>")
            except Exception:
                pass
            try:
                first_param.unbind("<Shift-Tab>")
            except Exception:
                pass

            def focus_first_param(event):
                first_param.focus_set()
                return "break"

            def focus_last_roi(event):
                last_roi.focus_set()
                return "break"

            last_roi.bind("<Tab>", focus_first_param)
            first_param.bind("<Shift-Tab>", focus_last_roi)

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
        # Frame for Loaded: label and spinner
        loaded_frame = ttk.Frame(data_load_box)
        loaded_frame.pack(anchor="w", pady=(5, 0), fill=tk.X)
        self.loaded_label = ttk.Label(loaded_frame, text="Loaded: <filename>")
        self.loaded_label.pack(side="left")
        self.spinner_label = ttk.Label(loaded_frame, text="", width=2)
        self.spinner_label.pack(side="left")
        self.shape_label = ttk.Label(data_load_box, text="Shape: -")
        self.shape_label.pack(anchor="w", pady=(0, 0))
        self._spinner_running = False
        self._spinner_cycle = ['|', '/', '-', '\\']
        self._spinner_index = 0
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
        # --- Ensure all Entry widgets have takefocus=True for correct tab order ---
        self._bind_roi_param_tab_order()
        # --- Animated Analysis Progress Label ---
        self.analysis_anim_text = "Analyzing..."
        self.analysis_anim_progress = 0
        self.analysis_animating = False
        self.analysis_thread = None
        self.analysis_done = False
        self.analysis_anim_frame = ttk.Frame(controls)
        self.analysis_anim_frame.pack(pady=(0, 0))
        for c in self.analysis_anim_text:
            lbl = ttk.Label(self.analysis_anim_frame, text=c, foreground="#bbbbbb", font=("Arial", 11, "bold"))
            lbl.pack(side="left")
        self.analysis_anim_frame.pack_forget()  # Hide initially
        # Run Analysis and Save Results Buttons (side by side)
        analysis_btn_frame = ttk.Frame(controls)
        analysis_btn_frame.pack(pady=(0, 10), fill=tk.X)
        self.run_button = ttk.Button(analysis_btn_frame, text="5. Run Analysis", command=self.start_analysis)
        self.run_button.pack(side="left", expand=True, fill=tk.X, padx=(0, 2))
        self.save_button = ttk.Button(analysis_btn_frame, text="Save Results", command=self.save_results)
        self.save_button.pack(side="left", expand=True, fill=tk.X, padx=(2, 0))
        # Fit Parameters Section (dynamic)
        self.fit_param_frame = None
        self.fit_param_entries = None
        self.create_fit_param_section("CW")
        # === Right: Output Area ===
        output_frame = ttk.Frame(main_pane, padding=10)
        output_frame.pack_propagate(False)
        # Add a top spacer inside the right panel for alignment
        ttk.Frame(output_frame, height=20).pack(fill='x')
        main_pane.add(output_frame, weight=5)
        output_frame.columnconfigure(0, weight=1)
        output_frame.columnconfigure(1, weight=5)
        output_frame.columnconfigure(2, weight=1)
        output_frame.rowconfigure(0, weight=1)
        output_frame.rowconfigure(1, weight=10)
        # Results Notebook (now for each analysis run, not per protocol)
        self.results_notebook = ttk.Notebook(output_frame)
        self.results_notebook.grid(row=0, column=0, columnspan=3, rowspan=2, sticky="nsew", padx=0)
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
            self.fit_param_frame, self.fit_param_entries = create_fit_param_section(
                self.root.nametowidget(self.param_frame.master), protocol, fit_param_config)
            self.fit_param_frame.pack(fill=tk.X, pady=10)
            # Wire up Save Fit Results button
            if self.fit_param_entries and "save_fit_btn" in self.fit_param_entries:
                self.fit_param_entries["save_fit_btn"].config(command=self.save_fit_results)
            # Wire up the fit button to always use the latest result for the selected tab
            if self.fit_param_entries and "fit_btn" in self.fit_param_entries:
                def fit_command():
                    idx = self.results_notebook.index(self.results_notebook.select())
                    tab_widget, _ = self.result_tabs[idx]
                    result = self.tab_results.get(tab_widget, None)
                    data_frame = None
                    summary = None
                    # Find the data_frame and summary frames in the tab
                    for child in tab_widget.winfo_children():
                        for subchild in child.winfo_children():
                            if isinstance(subchild, ttk.LabelFrame):
                                if subchild.cget("text") == "Data":
                                    data_frame = subchild
                                elif subchild.cget("text") == "Summary":
                                    summary = subchild
                    if protocol == "CW":
                        self.run_cw_fit(result, data_frame, summary)
                    elif protocol == "Rabi":
                        self.run_rabi_fit(result, data_frame, summary)
                    elif protocol == "Ramsey":
                        self.run_ramsey_fit(result, data_frame, summary)
                    elif protocol == "T1":
                        self.run_t1_fit(result, data_frame, summary)

                self.fit_param_entries["fit_btn"].config(command=fit_command)
        else:
            self.fit_param_frame = None
            self.fit_param_entries = None

    def on_protocol_change(self, event=None):
        proto = self.proto_cb.get()
        # Always remove both from the layout
        self.param_frame.pack_forget()
        self.custom_range_frame.pack_forget()
        # Re-create param_frame for correct protocol
        self.param_frame, self.param_entries = create_param_section(self.root.nametowidget(self.param_frame.master),
                                                                    proto)
        # Pack the correct one just before the Run Analysis button
        if proto == "T1":
            # Only show the custom range for T1
            self.custom_range_frame.pack(fill=tk.X, pady=10, before=self.run_button)
            self.custom_range_frame.lift()  # Ensure it's on top/visible
        else:
            self.param_frame.pack(fill=tk.X, pady=10, before=self.run_button)
        self.create_fit_param_section(proto)
        self._bind_roi_param_tab_order()

    def _run_analysis_with_callback_safe(self):
        print("DEBUG: _run_analysis_with_callback_safe called")
        try:
            result = self._run_analysis_computation_only()
            self.root.after(0, lambda: self._handle_analysis_result(result))
        finally:
            self.root.after(0, self.finish_analysis)

    def start_analysis(self):
        print("DEBUG: dir(self) =", dir(self))
        self.analysis_anim_progress = 0
        self.analysis_animating = True
        self.analysis_done = False
        self.run_button.config(state="disabled", text="")
        self.animate_analysis_button()
        # If using external data, skip threading and analysis
        if hasattr(self, 'data_is_external') and self.data_is_external:
            result = self._run_external_analysis()
            self._handle_analysis_result(result)
            self.finish_analysis()
            return
        # Start the real analysis in a background thread
        self.analysis_thread = threading.Thread(target=self._run_analysis_with_callback_safe, daemon=True)
        self.analysis_thread.start()

    def _run_external_analysis(self):
        # Use self.external_data to create a result object for plotting/fitting
        if self.external_data is None:
            return {
                'proto': 'external',
                'tab_label': 'External Data',
                'result_obj': None,
                'error': 'No external data loaded.'
            }
        x = self.external_data['x']
        y = self.external_data['y']
        yerr = self.external_data.get('yerr', None)
        protocol = self.external_data['protocol']
        filename = self.external_data['filename']
        tab_label = f"{protocol} (external) - {filename}"
        result = {
            'proto': 'external',
            'tab_label': tab_label,
            'result_obj': {'x': x, 'y': y, 'yerr': yerr, 'protocol': protocol, 'filename': filename},
            'error': None
        }
        # --- Always compute and store FFT for Ramsey ---
        if protocol == "Ramsey":
            import numpy as np
            from scipy.interpolate import interp1d
            x_ns = np.array(x)
            y_arr = np.array(y)
            x_s = x_ns * 1e-9
            uniform_time = np.linspace(x_s.min(), x_s.max(), len(x_s))
            interp_func = interp1d(x_s, y_arr, kind='cubic')
            uniform_signal = interp_func(uniform_time)
            norm_signal = (uniform_signal - np.min(uniform_signal)) / (np.max(uniform_signal) - np.min(uniform_signal))
            dt = uniform_time[1] - uniform_time[0]
            N = len(uniform_time)
            freq = np.fft.fftfreq(N, d=dt)
            fft_vals = np.fft.fft(norm_signal)
            mask = freq > 0
            result['result_obj']['fft_freq'] = freq[mask] * 1e-6  # MHz
            result['result_obj']['fft_amp'] = np.abs(fft_vals[mask])
        return result

    def _run_analysis_computation_only(self):
        import numpy as np
        if hasattr(self, 'data_is_external') and self.data_is_external:
            return self._run_external_analysis()
        # Validate ROI for internal data only
        roi_validation = self.validate_roi(return_values=True)
        # Unpack with default for third value
        if len(roi_validation) == 3:
            valid, msg, _ = roi_validation
        else:
            valid, msg = roi_validation
            _ = None
        if not valid:
            return {"proto": self.proto_cb.get().strip(),
                    "tab_label": self.loaded_label.cget("text").replace("Loaded: ", ""), "result_obj": None,
                    "error": f"ROI validation failed: {msg}"}
        proto = self.proto_cb.get().strip()
        filename = self.loaded_label.cget("text").replace("Loaded: ", "")
        tab_label = f"{proto} - {filename}" if filename and filename != "<filename>" else proto
        result = {"proto": proto, "tab_label": tab_label, "result_obj": None, "error": None}
        try:
            if self.data is None:
                result["error"] = "No data loaded."
                return result
            if proto == "CW":
                # Get ROI and parameters
                x_min = int(self.roi_entries["X start"].get())
                x_max = int(self.roi_entries["X end"].get())
                y_min = int(self.roi_entries["Y start"].get())
                y_max = int(self.roi_entries["Y end"].get())
                start = float(self.param_entries["Start"].get())
                end = float(self.param_entries["End"].get())
                step = float(self.param_entries["Steps"].get())
                x_range = np.arange(start, end + step, step)
                num_points = len(x_range)
                num_averages = int(self.data.shape[0] // num_points)
                if num_averages * num_points != self.data.shape[0]:
                    raise ValueError(
                        f"Data shape ({self.data.shape[0]}) is not a multiple of computed number of points ({num_points}).")
                image = self.data[-1] if self.data.ndim == 3 else self.data
                # --- Use preprocessed data if available ---
                if hasattr(self, "preprocessed_cw_data") and self.preprocessed_cw_data is not None and \
                        self.preprocessed_cw_data.shape[0] == num_points:
                    print("[CW] Using fast ROI extraction pipeline.")
                    try:
                        from analysis_utils import extract_roi_trace
                        odmr_trace = extract_roi_trace(self.preprocessed_cw_data, x_min, x_max, y_min, y_max)
                        from analysis_results import ODMRResult
                        result_obj = ODMRResult(
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
                        result["result_obj"] = result_obj
                        return result
                    except Exception as e:
                        print("CW fast ROI extraction failed, falling back:", e)
                # --- First run: do full analysis and preprocess for future fast ROI ---
                print("[CW] Running full analysis and preprocessing for future fast ROI extraction.")
                from odmr_analy import odmr_analyze_data
                result_obj = odmr_analyze_data(
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
                usable = self.data[3 * num_points:]
                num_usable_frames = usable.shape[0]
                if num_usable_frames % num_points != 0:
                    self.preprocessed_cw_data = None
                    print("[CW] Preprocessing skipped: shape mismatch (usable frames not divisible by num_points).")
                else:
                    from analysis_utils import preprocess_odmr_data
                    num_averages_post = num_usable_frames // num_points
                    self.preprocessed_cw_data = preprocess_odmr_data(usable, num_averages_post, num_points)
                    self.cw_num_points = num_points
                    print("[CW] Preprocessing complete. Fast ROI extraction enabled for future runs.")
                result["result_obj"] = result_obj
            elif proto == "Rabi":
                x_min = int(self.roi_entries["X start"].get())
                x_max = int(self.roi_entries["X end"].get())
                y_min = int(self.roi_entries["Y start"].get())
                y_max = int(self.roi_entries["Y end"].get())
                start = float(self.param_entries["Start"].get())
                end = float(self.param_entries["End"].get())
                num_points = int(self.param_entries["Number of Points"].get())
                x_range = np.linspace(start, end, num_points)
                num_averages = int(self.data.shape[0] // (2 * num_points))
                if num_averages * 2 * num_points != self.data.shape[0]:
                    raise ValueError(
                        f"Data shape ({self.data.shape[0]}) is not a multiple of 2 x computed number of points ({num_points}).")
                image = self.data[-1] if self.data.ndim == 3 else self.data
                # --- Use preprocessed data if available ---
                use_preprocessed = (
                        hasattr(self, 'rabi_pixel_traces') and self.rabi_pixel_traces is not None and
                        hasattr(self, 'rabi_num_points') and self.rabi_num_points == num_points and
                        hasattr(self, 'rabi_x_range') and np.array_equal(self.rabi_x_range, x_range)
                )
                if use_preprocessed:
                    print("[Rabi] Using preprocessed data for fast ROI extraction")
                    try:
                        roi_trace = self.extract_rabi_roi_trace(x_min, x_max, y_min, y_max)

                        class FastRabiResult:
                            def __init__(self, x, y, image, roi_bounds):
                                self.x = x
                                self.y = y
                                self.error = np.zeros_like(y)
                                self.mean_signal = y
                                self.mean_reference = np.ones_like(y)
                                self.image = image
                                self.roi_bounds = roi_bounds

                        result_obj = FastRabiResult(x_range, roi_trace, image, (x_min, x_max, y_min, y_max))
                        return {
                            'proto': 'Rabi',
                            'tab_label': tab_label,
                            'result_obj': result_obj,
                            'error': None
                        }
                    except Exception as e:
                        print(f"[Rabi] Fast extraction failed, falling back to full analysis: {e}")
                # --- First run: do full analysis and preprocess for future fast ROI ---
                print("[Rabi] Running full analysis and preprocessing data")
                from rabi_analy import rabi_analyze_data
                result_obj = rabi_analyze_data(
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
                result["result_obj"] = result_obj
            elif proto == "Ramsey":
                x_min = int(self.roi_entries["X start"].get())
                x_max = int(self.roi_entries["X end"].get())
                y_min = int(self.roi_entries["Y start"].get())
                y_max = int(self.roi_entries["Y end"].get())
                start = float(self.param_entries["Start"].get())
                end = float(self.param_entries["End"].get())
                num_points = int(self.param_entries["Number of Points"].get())
                x_range = np.linspace(start, end, num_points)
                num_averages = int(self.data.shape[0] // (2 * num_points))
                if num_averages * 2 * num_points != self.data.shape[0]:
                    raise ValueError(
                        f"Data shape ({self.data.shape[0]}) is not a multiple of 2 x computed number of points ({num_points}).")
                image = self.data[-1] if self.data.ndim == 3 else self.data
                # --- Use preprocessed data if available ---
                use_preprocessed = (
                        hasattr(self, 'ramsey_pixel_traces') and self.ramsey_pixel_traces is not None and
                        hasattr(self, 'ramsey_num_points') and self.ramsey_num_points == num_points and
                        hasattr(self, 'ramsey_x_range') and np.array_equal(self.ramsey_x_range, x_range)
                )
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
                        result_obj = FastRamseyResult(x_range, roi_trace, image, (x_min, x_max, y_min, y_max))
                        # --- Always compute and store FFT for Ramsey ---
                        try:
                            import numpy as np
                            from scipy.interpolate import interp1d
                            x_ns = np.array(result_obj.x)
                            y_arr = np.array(result_obj.y)
                            x_s = x_ns * 1e-9
                            uniform_time = np.linspace(x_s.min(), x_s.max(), len(x_s))
                            interp_func = interp1d(x_s, y_arr, kind='cubic')
                            uniform_signal = interp_func(uniform_time)
                            norm_signal = (uniform_signal - np.min(uniform_signal)) / (np.max(uniform_signal) - np.min(uniform_signal))
                            dt = uniform_time[1] - uniform_time[0]
                            N = len(uniform_time)
                            freq = np.fft.fftfreq(N, d=dt)
                            fft_vals = np.fft.fft(norm_signal)
                            mask = freq > 0
                            result_obj.fft_freq = freq[mask] * 1e-6  # MHz
                            result_obj.fft_amp = np.abs(fft_vals[mask])
                        except Exception:
                            pass
                        return {
                            'proto': 'Ramsey',
                            'tab_label': tab_label,
                            'result_obj': result_obj,
                            'error': None
                        }
                    except Exception as e:
                        print(f"[Ramsey] Fast extraction failed, falling back to full analysis: {e}")
                # --- First run: do full analysis and preprocess for future fast ROI ---
                print("[Ramsey] Running full analysis and preprocessing data")
                from rabi_analy import rabi_analyze_data
                result_obj = rabi_analyze_data(
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
                    self.ramsey_pixel_traces = self.preprocess_ramsey_data(self.data, num_averages, num_points)
                    self.ramsey_num_points = num_points
                    self.ramsey_x_range = x_range.copy()
                    print("[Ramsey] Preprocessing complete. Fast ROI extraction enabled for future runs.")
                except Exception as e:
                    print(f"[Ramsey] Preprocessing failed: {e}")
                # --- Always compute and store FFT for Ramsey ---
                try:
                    import numpy as np
                    from scipy.interpolate import interp1d
                    x_ns = np.array(result_obj.x)
                    y_arr = np.array(result_obj.y)
                    x_s = x_ns * 1e-9
                    uniform_time = np.linspace(x_s.min(), x_s.max(), len(x_s))
                    interp_func = interp1d(x_s, y_arr, kind='cubic')
                    uniform_signal = interp_func(uniform_time)
                    norm_signal = (uniform_signal - np.min(uniform_signal)) / (np.max(uniform_signal) - np.min(uniform_signal))
                    dt = uniform_time[1] - uniform_time[0]
                    N = len(uniform_time)
                    freq = np.fft.fftfreq(N, d=dt)
                    fft_vals = np.fft.fft(norm_signal)
                    mask = freq > 0
                    result_obj.fft_freq = freq[mask] * 1e-6  # MHz
                    result_obj.fft_amp = np.abs(fft_vals[mask])
                except Exception:
                    pass
                result["result_obj"] = result_obj
            elif proto == "T1":
                # T1: Always run full analysis. No fast ROI pipeline is implemented for T1.
                x_min = int(self.roi_entries["X start"].get())
                x_max = int(self.roi_entries["X end"].get())
                y_min = int(self.roi_entries["Y start"].get())
                y_max = int(self.roi_entries["Y end"].get())
                # Use custom x_range if provided in the custom_range_entry
                custom_range_str = self.custom_range_entry.get().strip()
                default_x_range = np.array(
                    [15000, 14000, 13000, 12000, 11000, 10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1500,
                     1200, 1000, 800, 500, 300, 100]) * 1e-3
                start_str = self.param_entries[
                    "Start"].get() if self.param_entries and "Start" in self.param_entries else ""
                end_str = self.param_entries["End"].get() if self.param_entries and "End" in self.param_entries else ""
                step_str = self.param_entries[
                    "Steps"].get() if self.param_entries and "Steps" in self.param_entries else ""
                x_range = None
                if custom_range_str:
                    try:
                        x_range = np.array([float(val) for val in custom_range_str.split(",")]) * 1e-3
                    except Exception:
                        messagebox.showerror("Input Error",
                                             "Invalid custom range values. Please enter comma-separated numbers.")
                        return
                elif start_str.strip() == "" and end_str.strip() == "" and step_str.strip() == "":
                    x_range = default_x_range
                else:
                    try:
                        start = float(start_str) if start_str.strip() != "" else default_x_range[-1]
                        end = float(end_str) if end_str.strip() != "" else default_x_range[0]
                        step = float(step_str) if step_str.strip() != "" else -(
                                abs(end - start) / max(1, len(default_x_range) - 1))
                        if step > 0:
                            step = -step  # Ensure descending order
                        x_range = np.arange(end, start + step, step)[::-1] if step < 0 else np.arange(start, end + step,
                                                                                                      step)
                    except Exception:
                        messagebox.showerror("Input Error", "Invalid experimental parameters.")
                        return
                num_points = len(x_range)
                try:
                    num_averages = int(self.data.shape[0] // num_points)
                except Exception:
                    result["error"] = "Data shape does not match computed number of points."
                    return result
                if num_averages * num_points != self.data.shape[0]:
                    result[
                        "error"] = f"Data shape ({self.data.shape[0]}) is not a multiple of computed number of points ({num_points})."
                    return result
                image = self.data[-1] if self.data.ndim == 3 else self.data
                try:
                    from t1_analy import t1_analyze_data
                    result_obj = t1_analyze_data(
                        image=image,
                        data=self.data,
                        x_range=x_range,
                        x_min=x_min,
                        x_max=x_max,
                        y_min=y_min,
                        y_max=y_max
                    )
                    result["result_obj"] = result_obj
                except Exception as e:
                    result["error"] = f"T1 analysis failed: {e}"
                    return result
            else:
                result["error"] = "Unknown protocol"
        except Exception as e:
            result["error"] = str(e)
        return result

    def _handle_analysis_result(self, result):
        proto = result.get("proto")
        tab_label = result.get("tab_label")
        error = result.get("error")
        if error:
            messagebox.showerror("Analysis Error", error)
            return
        if proto == "external":
            tab = ttk.Frame(self.results_notebook)
            self._add_tab_with_close(tab, tab_label)
            self.results_notebook.select(tab)
            self.tab_results[tab] = result["result_obj"]
            self.populate_external_result_tab(tab, result["result_obj"])
            return
        if proto == "CW":
            tab = ttk.Frame(self.results_notebook)
            self._add_tab_with_close(tab, tab_label)
            self.results_notebook.select(tab)
            self.tab_results[tab] = result["result_obj"]
            self.populate_cw_result_tab(tab, result["result_obj"])
        elif proto == "Rabi":
            tab = ttk.Frame(self.results_notebook)
            self._add_tab_with_close(tab, tab_label)
            self.results_notebook.select(tab)
            self.tab_results[tab] = result["result_obj"]
            self.populate_rabi_result_tab(tab, result["result_obj"])
        elif proto == "Ramsey":
            tab = ttk.Frame(self.results_notebook)
            self._add_tab_with_close(tab, tab_label)
            self.results_notebook.select(tab)
            self.tab_results[tab] = result["result_obj"]
            self.populate_ramsey_result_tab(tab, result["result_obj"])
        elif proto == "T1":
            tab = ttk.Frame(self.results_notebook)
            self._add_tab_with_close(tab, tab_label)
            self.results_notebook.select(tab)
            self.tab_results[tab] = result["result_obj"]
            self.populate_t1_result_tab(tab, result["result_obj"])
        else:
            tab = ttk.Frame(self.results_notebook)
            self._add_tab_with_close(tab, tab_label)
            self.results_notebook.select(tab)
            self.populate_result_tab(tab, proto)

    def animate_analysis_button(self):
        # Typewriter effect: show 1 more letter each time, fade color from gray to black
        if self.analysis_done:
            self.run_button.config(text="5. Run Analysis", state="normal")
            return
        display = self.analysis_anim_text[:self.analysis_anim_progress]
        self.run_button.config(text=display)
        if self.analysis_animating and not self.analysis_done:
            # Loop the animation
            self.analysis_anim_progress = (self.analysis_anim_progress + 1) % (len(self.analysis_anim_text) + 1)
            self.root.after(80, self.animate_analysis_button)

    def finish_analysis(self):
        # Robustly stop animation and reset button label/state
        self.analysis_animating = False
        self.analysis_done = True
        # Force update the button label and state
        self.run_button.config(text="5. Run Analysis", state="normal")

    def run_analysis(self):
        import numpy as np
        print("DEBUG: run_analysis called")
        proto = self.proto_cb.get()
        filename = self.loaded_label.cget("text").replace("Loaded: ", "")
        tab_label = f"{proto} - {filename}" if filename and filename != "<filename>" else proto
        tab = ttk.Frame(self.results_notebook)
        self._add_tab_with_close(tab, tab_label)
        self.results_notebook.select(tab)
        # ROI validation and parsing
        valid, error, roi_vals = self.validate_roi(return_values=True)
        print(f"DEBUG: ROI validation result: valid={valid}, error={error}, roi_vals={roi_vals}")
        if not valid or roi_vals is None:
            messagebox.showerror("Input Error", error)
            return
        x_min, x_max, y_min, y_max = roi_vals
        # Remove all protocol-specific ROI bound checks below
        # ... rest of the method unchanged, just use x_min, x_max, y_min, y_max ...
        # CW protocol wiring
        if proto == "CW":
            if not hasattr(self, "data") or self.data is None:
                messagebox.showerror("Error", "No data loaded.")
                return
            # Use validated ROI values
            IMAGE_SIZE = 512
            if not (0 <= x_min < x_max <= IMAGE_SIZE and 0 <= y_min < y_max <= IMAGE_SIZE):
                messagebox.showerror("Input Error",
                                     f"ROI values must be within 0 and {IMAGE_SIZE}, and min < max for both axes.")
                return
            # Extract experimental parameters
            try:
                start = float(self.param_entries["Start"].get())
                end = float(self.param_entries["End"].get())
                step = float(self.param_entries["Steps"].get())
            except Exception:
                messagebox.showerror("Input Error", "Invalid experimental parameters.")
                return
            x_range = np.arange(start, end + step, step)
            num_points = len(x_range)
            try:
                num_averages = int(self.data.shape[0] // num_points)
            except Exception:
                messagebox.showerror("Data Error", "Data shape does not match computed number of points.")
                return
            if num_averages * num_points != self.data.shape[0]:
                messagebox.showerror("Data Error",
                                     f"Data shape ({self.data.shape[0]}) is not a multiple of computed number of points ({num_points}).")
                return
            if hasattr(self, "preprocessed_cw_data") and self.preprocessed_cw_data is not None and \
                    self.preprocessed_cw_data.shape[0] == num_points:
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
                usable = self.data[3 * num_points:]
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
            self.tab_results[tab] = result
            self.populate_cw_result_tab(tab, result)
        elif proto == "Rabi":
            if not hasattr(self, "data") or self.data is None:
                messagebox.showerror("Error", "No data loaded.")
                return
            # Use validated ROI values
            data_shape = self.data.shape
            if len(data_shape) >= 3:
                max_y, max_x = data_shape[-2], data_shape[-1]
            else:
                max_y, max_x = data_shape[0], data_shape[1]
            if (x_min < 0 or x_max > max_x or y_min < 0 or y_max > max_y or x_min >= x_max or y_min >= y_max):
                messagebox.showerror("Input Error",
                                     f"Invalid ROI bounds. Data shape: ({max_y}, {max_x}). ROI must be within bounds and have positive size.")
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
                messagebox.showerror("Data Error", "Data shape does not match computed number of points for Rabi.")
                return
            if num_averages * 2 * num_points != self.data.shape[0]:
                messagebox.showerror("Data Error",
                                     f"Data shape ({self.data.shape[0]}) is not a multiple of 2 x computed number of points ({num_points}).")
                return
            image = self.data[-1] if self.data.ndim == 3 else self.data
            use_preprocessed = (hasattr(self, 'rabi_pixel_traces') and hasattr(self,
                                                                               'rabi_num_points') and self.rabi_num_points == num_points and hasattr(
                self, 'rabi_x_range') and np.array_equal(self.rabi_x_range, x_range))
            if use_preprocessed:
                print("[Rabi] Using preprocessed data for fast ROI extraction")
                try:
                    roi_trace = self.extract_rabi_roi_trace(x_min, x_max, y_min, y_max)

                    class FastRabiResult:
                        def __init__(self, x, y, image, roi_bounds):
                            self.x = x
                            self.y = y
                            self.error = np.zeros_like(y)
                            self.mean_signal = y
                            self.mean_reference = np.ones_like(y)
                            self.image = image
                            self.roi_bounds = roi_bounds

                    result = FastRabiResult(x_range, roi_trace, image, (x_min, x_max, y_min, y_max))
                    return {
                        'proto': 'Rabi',
                        'tab_label': tab_label,
                        'result_obj': result,
                        'error': None
                    }
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
                try:
                    self.rabi_pixel_traces = self.preprocess_rabi_data(self.data, num_averages, num_points)
                    self.rabi_num_points = num_points
                    self.rabi_x_range = x_range.copy()
                    print("[Rabi] Preprocessing complete. Fast ROI extraction enabled for future runs.")
                except Exception as e:
                    print(f"[Rabi] Preprocessing failed: {e}")
                result_obj = result
            except Exception as e:
                messagebox.showerror("Analysis Error", f"Rabi analysis failed: {e}")
                return
            self.tab_results[tab] = result_obj
            self.populate_rabi_result_tab(tab, result_obj)
        elif proto == "Ramsey":
            if not hasattr(self, "data") or self.data is None:
                messagebox.showerror("Error", "No data loaded.")
                return
            # Use validated ROI values
            data_shape = self.data.shape
            if len(data_shape) >= 3:
                max_y, max_x = data_shape[-2], data_shape[-1]
            else:
                max_y, max_x = data_shape[0], data_shape[1]
            if (x_min < 0 or x_max > max_x or y_min < 0 or y_max > max_y or x_min >= x_max or y_min >= y_max):
                messagebox.showerror("Input Error",
                                     f"Invalid ROI bounds. Data shape: ({max_y}, {max_x}). ROI must be within bounds and have positive size.")
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
                messagebox.showerror("Data Error",
                                     f"Data shape ({self.data.shape[0]}) is not a multiple of 2 x computed number of points ({num_points}).")
                return
            image = self.data[-1] if self.data.ndim == 3 else self.data
            use_preprocessed = (hasattr(self, 'ramsey_pixel_traces') and hasattr(self,
                                                                                 'ramsey_num_points') and self.ramsey_num_points == num_points and hasattr(
                self, 'ramsey_x_range') and np.array_equal(self.ramsey_x_range, x_range))
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
                    return {
                        'proto': 'Ramsey',
                        'tab_label': tab_label,
                        'result_obj': result,
                        'error': None
                    }
                except Exception as e:
                    print(f"[Ramsey] Fast extraction failed, falling back to full analysis: {e}")
                    use_preprocessed = False
            if not use_preprocessed:
                print("[Ramsey] Running full analysis and preprocessing data")
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
                try:
                    self.ramsey_pixel_traces = self.preprocess_ramsey_data(self.data, num_averages, num_points)
                    self.ramsey_num_points = num_points
                    self.ramsey_x_range = x_range.copy()
                    print("[Ramsey] Preprocessing complete. Fast ROI extraction enabled for future runs.")
                except Exception as e:
                    print(f"[Ramsey] Preprocessing failed: {e}")
                # --- Always compute and store FFT for Ramsey ---
                try:
                    import numpy as np
                    from scipy.interpolate import interp1d
                    x_ns = np.array(result.x)
                    y_arr = np.array(result.y)
                    x_s = x_ns * 1e-9
                    uniform_time = np.linspace(x_s.min(), x_s.max(), len(x_s))
                    interp_func = interp1d(x_s, y_arr, kind='cubic')
                    uniform_signal = interp_func(uniform_time)
                    norm_signal = (uniform_signal - np.min(uniform_signal)) / (np.max(uniform_signal) - np.min(uniform_signal))
                    dt = uniform_time[1] - uniform_time[0]
                    N = len(uniform_time)
                    freq = np.fft.fftfreq(N, d=dt)
                    fft_vals = np.fft.fft(norm_signal)
                    mask = freq > 0
                    result.fft_freq = freq[mask] * 1e-6  # MHz
                    result.fft_amp = np.abs(fft_vals[mask])
                except Exception:
                    pass
                result_obj = result
            except Exception as e:
                messagebox.showerror("Analysis Error", f"Ramsey analysis failed: {e}")
                return
            self.tab_results[tab] = result_obj
            self.populate_ramsey_result_tab(tab, result_obj)
        elif proto == "T1":
            if not hasattr(self, "data") or self.data is None:
                messagebox.showerror("Error", "No data loaded.")
                return
            # Use validated ROI values
            try:
                custom_range_str = self.custom_range_entry.get().strip()
                default_x_range = np.array(
                    [15000, 14000, 13000, 12000, 11000, 10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1500,
                     1200, 1000, 800, 500, 300, 100]) * 1e-3
                start_str = self.param_entries[
                    "Start"].get() if self.param_entries and "Start" in self.param_entries else ""
                end_str = self.param_entries["End"].get() if self.param_entries and "End" in self.param_entries else ""
                step_str = self.param_entries[
                    "Steps"].get() if self.param_entries and "Steps" in self.param_entries else ""
                x_range = None
                if custom_range_str:
                    try:
                        x_range = np.array([float(val) for val in custom_range_str.split(",")]) * 1e-3
                    except Exception:
                        messagebox.showerror("Input Error",
                                             "Invalid custom range values. Please enter comma-separated numbers.")
                        return
                elif start_str.strip() == "" and end_str.strip() == "" and step_str.strip() == "":
                    x_range = default_x_range
                else:
                    try:
                        start = float(start_str) if start_str.strip() != "" else default_x_range[-1]
                        end = float(end_str) if end_str.strip() != "" else default_x_range[0]
                        step = float(step_str) if step_str.strip() != "" else -(
                                abs(end - start) / max(1, len(default_x_range) - 1))
                        if step > 0:
                            step = -step  # Ensure descending order
                        x_range = np.arange(end, start + step, step)[::-1] if step < 0 else np.arange(start, end + step,
                                                                                                      step)
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
                    messagebox.showerror("Data Error",
                                         f"Data shape ({self.data.shape[0]}) is not a multiple of computed number of points ({num_points}).")
                    return
                image = self.data[-1] if self.data.ndim == 3 else self.data
                from t1_analy import t1_analyze_data
                result_obj = t1_analyze_data(
                    image=image,
                    data=self.data,
                    x_range=x_range,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max
                )
                self.tab_results[tab] = result_obj
                self.populate_t1_result_tab(tab, result_obj)
            except Exception as e:
                messagebox.showerror("Analysis Error", f"T1 analysis failed: {e}")
                return
        else:
            tab = ttk.Frame(self.results_notebook)
            self._add_tab_with_close(tab, tab_label)
            self.results_notebook.select(tab)
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
                    # --- For Ramsey, if FFT data is present, save both time and FFT columns ---
                    if proto == 'Ramsey' and (hasattr(result, 'fft_freq') or (isinstance(result, dict) and ('fft_freq' in result))):
                        # Get arrays
                        x = getattr(result, 'x') if hasattr(result, 'x') else result['x']
                        y = getattr(result, 'y') if hasattr(result, 'y') else result['y']
                        fft_freq = getattr(result, 'fft_freq', None) if hasattr(result, 'fft_freq') else result.get('fft_freq', None)
                        fft_amp = getattr(result, 'fft_amp', None) if hasattr(result, 'fft_amp') else result.get('fft_amp', None)
                        n = max(len(x), len(fft_freq) if fft_freq is not None else 0)
                        f.write("time (ns),y,fft_freq (MHz),fft_amp\n")
                        for i in range(n):
                            t_val = x[i] if i < len(x) else ""
                            y_val = y[i] if i < len(y) else ""
                            freq_val = fft_freq[i] if (fft_freq is not None and i < len(fft_freq)) else ""
                            amp_val = fft_amp[i] if (fft_amp is not None and i < len(fft_amp)) else ""
                            f.write(f"{t_val},{y_val},{freq_val},{amp_val}\n")
                    else:
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
            ext = os.path.splitext(file_path)[-1].lower()
            if ext == ".csv":
                # Read first 10 lines for preview and header selection
                preview_lines = []
                with open(file_path, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    for i, row in enumerate(reader):
                        preview_lines.append(row)
                        if i >= 9:
                            break
                
                # Combined dialog for header row and column/protocol selection
                dialog = CSVColumnSelectorDialog(self.root, preview_lines, self.protocols)
                if dialog.result is None:
                    return  # User cancelled
                
                header_row_idx = dialog.result['header_row']
                x_col = dialog.result['x']
                y_col = dialog.result['y']
                yerr_col = dialog.result['yerr']
                protocol = dialog.result['protocol']
                
                # Read full CSV as dict, skipping lines before header
                headers = preview_lines[header_row_idx]
                data_dict = {h: [] for h in headers}
                with open(file_path, newline='') as csvfile:
                    for _ in range(header_row_idx):
                        next(csvfile)
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        for h in headers:
                            data_dict[h].append(row[h])
                try:
                    x = np.array([float(val) for val in data_dict[x_col]])
                    y = np.array([float(val) for val in data_dict[y_col]])
                    yerr = None
                    if yerr_col:
                        yerr = np.array([float(val) for val in data_dict[yerr_col]])
                except Exception as e:
                    messagebox.showerror("CSV Error", f"Could not parse selected columns as numbers: {e}")
                    return
                self.external_data = {'x': x, 'y': y, 'yerr': yerr, 'protocol': protocol,
                                      'filename': os.path.basename(file_path)}
                self.data_is_external = True
                # Update protocol dropdown and disable it
                self.proto_cb.set(protocol)
                self.proto_cb.config(state="disabled")
                self.on_protocol_change()
                # Show loaded filename and shape
                self.loaded_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                self.shape_label.config(text=f"Shape: {x.shape}")
                # Prepare for plotting/fitting in later steps
                self.disable_controls_for_external()
                self.create_fit_param_section(protocol)
                # Show the external data in a new results tab immediately
                result = self._run_external_analysis()
                self._handle_analysis_result(result)
                return
            # If loading a non-external file, re-enable all controls
            self.enable_all_controls()
            self.loaded_label.config(text="Loading:")
            self._start_spinner()
            self.shape_label.config(text="Shape: -")
            self.data_is_external = False
            self.external_data = None
            self.proto_cb.config(state="readonly")
            threading.Thread(target=self._load_data_thread, args=(file_path,), daemon=True).start()

    def disable_controls_for_external(self):
        # Disable ROI, experimental params, protocol selection, run analysis, save results
        for entry in self.roi_entries.values():
            entry.config(state="disabled")
        if self.param_entries:
            for entry in self.param_entries.values():
                if hasattr(entry, 'config'):
                    entry.config(state="disabled")
                elif hasattr(entry, 'set'):
                    entry.set(False)
        self.proto_cb.config(state="disabled")
        self.run_button.config(state="disabled")
        self.save_button.config(state="disabled")
        # Only enable load button and fit section (if present)
        if self.fit_param_entries:
            for k, entry in self.fit_param_entries.items():
                if k in ("fit_btn", "save_fit_btn"):
                    entry.config(state="normal")
                else:
                    if hasattr(entry, 'config'):
                        entry.config(state="disabled")
                    elif hasattr(entry, 'set'):
                        entry.set(False)

    def enable_all_controls(self):
        # Enable all controls for normal data
        for entry in self.roi_entries.values():
            entry.config(state="normal")
        if self.param_entries:
            for entry in self.param_entries.values():
                if hasattr(entry, 'config'):
                    entry.config(state="normal")
        self.proto_cb.config(state="readonly")
        self.run_button.config(state="normal")
        self.save_button.config(state="normal")
        if self.fit_param_entries:
            for entry in self.fit_param_entries.values():
                if hasattr(entry, 'config'):
                    entry.config(state="normal")

    def _start_spinner(self):
        self._spinner_running = True
        self._spinner_index = 0
        self._animate_spinner()

    def _animate_spinner(self):
        if self._spinner_running:
            self.spinner_label.config(text=self._spinner_cycle[self._spinner_index])
            self._spinner_index = (self._spinner_index + 1) % len(self._spinner_cycle)
            self.root.after(100, self._animate_spinner)
        else:
            self.spinner_label.config(text="")

    def _stop_spinner(self):
        self._spinner_running = False
        self.spinner_label.config(text="")

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
            self._stop_spinner()
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
                for attr in [
                    'preprocessed_cw_data', 'cw_num_points',
                    'rabi_pixel_traces', 'rabi_num_points', 'rabi_x_range']:
                    if hasattr(self, attr):
                        delattr(self, attr)
                # --- Remove CW preprocessing from data load ---
                # (No preprocessing here; only after successful analysis)

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
            roi_bounds=(getattr(result, 'x_min', None), getattr(result, 'x_max', None), getattr(result, 'y_min', None),
                        getattr(result, 'y_max', None)),
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
            if hasattr(result, 'mean_signal') and hasattr(result,
                                                          'mean_reference') and result.mean_signal is not None and result.mean_reference is not None:
                ax2.plot(result.x, result.mean_signal, color='r', marker='o', markersize=3, linestyle='-',
                         label='Signal')
                ax2.plot(result.x, result.mean_reference, color='b', marker='o', markersize=3, linestyle='-',
                         label='Reference')
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
        main_row.pack(side="top", fill="both", expand=True, padx=5, pady=(0, 0))
        # Data plot
        data_frame = ttk.LabelFrame(main_row, text="Data")
        data_frame.pack(side="left", fill="both", expand=True, pady=(5, 0))
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
        summary.pack(side="left", fill="y", padx=(8, 0))
        summary.pack_propagate(False)
        # --- Fit button logic for CW ---
        if self.fit_param_entries and "fit_btn" in self.fit_param_entries:
            self.fit_param_entries["fit_btn"].config(command=lambda: self.run_cw_fit(result, data_frame, summary))
        # Initial summary
        for widget in summary.winfo_children():
            widget.destroy()
        ttk.Label(summary, text="CW analysis complete.", justify="left").pack(anchor="nw")

    def run_cw_fit(self, result, data_frame, summary):
        self.start_fit_animation()
        threading.Thread(target=self._run_cw_fit_thread, args=(result, data_frame, summary), daemon=True).start()

    def _run_cw_fit_thread(self, result, data_frame, summary):
        if self.fit_param_entries is None:
            self.root.after(0, lambda: messagebox.showerror("Fit Error", "Fit parameter entries are not available."))
            self.root.after(0, self.finish_fit_animation)
            return
        try:
            from cw_pulsed_fit import process_odmr_fit
            if isinstance(result, dict):
                x = result['x']
                y = result['y']
            else:
                x = result.x
                use_filtered = self.fit_param_entries["use_filtered"].get()
                y = result.filtered_y if use_filtered else result.y
            num_peaks = int(self.fit_param_entries["num_peaks"].get())
            threshold = float(self.fit_param_entries["threshold"].get())
            lookahead = int(self.fit_param_entries["lookahead"].get())
            show_plot = self.fit_param_entries["show_plot"].get()
            show_peaks = self.fit_param_entries["show_peaks"].get()
            try:
                popt, yfit, peak_x, peak_y = process_odmr_fit(x, y, num_peaks=num_peaks, threshold=threshold, lookahead=lookahead, plot=show_plot, show_peaks=show_peaks, return_peaks=True)
                fit_result = {
                    "x": x,
                    "y": y,
                    "yfit": yfit,
                    "popt": popt,
                    "num_peaks": num_peaks,
                    "threshold": threshold,
                    "lookahead": lookahead,
                    "show_peaks": show_peaks,
                    "peak_x": peak_x,
                    "peak_y": peak_y
                }
                self.root.after(0, lambda: self._handle_cw_fit_result(fit_result, None, data_frame, summary, result))
            except Exception as e:
                self.root.after(0, lambda e=e: self._handle_cw_fit_result(None, str(e), data_frame, summary, result))
        finally:
            self.root.after(0, self.finish_fit_animation)

    def _handle_cw_fit_result(self, fit_result, error, data_frame, summary, result):
        if error:
            messagebox.showerror("Fit Error", f"CW fit failed: {error}")
            return
        if data_frame is None:
            messagebox.showerror("Fit Error", "Internal error: Data frame not found.")
            return
        idx = self.results_notebook.index(self.results_notebook.select())
        tab_widget, _ = self.result_tabs[idx]
        param_names = []
        for i in range(fit_result["num_peaks"]):
            param_names.extend([f"A{i + 1}", f"x0{i + 1}", f"gamma{i + 1}"])
        self.tab_fit_results[tab_widget] = {
            "x": fit_result["x"],
            "y": fit_result["y"],
            "y_fit": 1 - fit_result["yfit"],
            "params": fit_result["popt"],
            "param_names": param_names,
            "equation": "y = 1 - sum_i A_i * exp(-((x - x0_i)^2) / (2 * gamma_i^2)) (multi-peak Lorentzian)"
        }
        for widget in data_frame.winfo_children():
            widget.destroy()
        from matplotlib.figure import Figure
        fig_fit = Figure(figsize=(6, 2.6), dpi=100)
        ax_fit = fig_fit.add_subplot(111)
        ax_fit.plot(fit_result["x"], fit_result["y"], 'ks', label="Raw Data", markersize=5, linestyle='None')
        if hasattr(result, 'filtered_y') and result.filtered_y is not None:
            ax_fit.plot(fit_result["x"], result.filtered_y, '--', color='gray', label="Filtered", linewidth=1)
        ax_fit.plot(fit_result["x"], 1 - fit_result["yfit"], 'r-', label="Fit", linewidth=1.5)
        # Show peaks if requested (ensure this is after all plot calls so peaks are on top)
        if fit_result.get("show_peaks") and fit_result.get("peak_x") is not None and fit_result.get("peak_y") is not None:
            ax_fit.scatter(fit_result["peak_x"], fit_result["peak_y"],
                           facecolors='blue', edgecolors='white',
                           marker='o', s=60, linewidths=1.5, label='Peaks', zorder=10)
        ax_fit.set_xlabel("Frequency (GHz)")
        ax_fit.set_ylabel("Normalized Intensity")
        ax_fit.set_title("")
        ax_fit.legend()
        fig_fit.tight_layout()
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas_fit = FigureCanvasTkAgg(fig_fit, master=data_frame)
        canvas_fit.draw()
        canvas_fit.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        add_toolbar(canvas_fit, data_frame)
        fit_text = "Fit parameters:\n"
        for i in range(0, len(fit_result["popt"]), 3):
            peak_num = i // 3 + 1
            amplitude = 1 - fit_result["popt"][i]
            center = fit_result["popt"][i + 1]
            width = fit_result["popt"][i + 2]
            fit_text += f"Peak {peak_num}:\n  amplitude: {amplitude:.4g}\n  center: {center:.4g} MHz\n  width: {width:.4g} MHz\n\n"
        for widget in summary.winfo_children():
            widget.destroy()
        ttk.Label(summary, text=fit_text, justify="left").pack(anchor="nw")

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
        self.create_result_plot(plot1, result.x, result.y, "ROI Intensity", figsize=(1.2, 1.2),
                                image=getattr(result, 'image', None), roi_bounds=getattr(result, 'roi_bounds', None),
                                force_square=True)
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
        if hasattr(result, 'mean_signal') and hasattr(result,
                                                      'mean_reference') and result.mean_signal is not None and result.mean_reference is not None:
            ax2.plot(result.x, result.mean_signal, color='r', marker='o', markersize=3, linestyle='-', label='Signal')
            ax2.plot(result.x, result.mean_reference, color='b', marker='o', markersize=3, linestyle='-',
                     label='Reference')
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
        main_row.pack(side="top", fill="both", expand=True, padx=5, pady=(0, 0))
        # Data plot
        data_frame = ttk.LabelFrame(main_row, text="Data")
        data_frame.pack(side="left", fill="both", expand=True, pady=(5, 0))
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
        summary.pack(side="left", fill="y", padx=(8, 0))
        summary.pack_propagate(False)
        contrast = (np.max(result.y) - np.min(result.y)) / np.max(result.y) * 100 if np.max(result.y) != 0 else 0
        mean_intensity = np.mean(result.y)
        summary_text = f"Rabi analysis complete.\nContrast: {contrast:.2f}%\nMean Intensity: {mean_intensity:.3g}"
        for widget in summary.winfo_children():
            widget.destroy()
        ttk.Label(summary, text=summary_text, justify="left").pack(anchor="nw")
        # --- Fit button logic for Rabi ---
        if self.fit_param_entries and "fit_btn" in self.fit_param_entries:
            self.fit_param_entries["fit_btn"].config(command=lambda: self.run_rabi_fit(result, data_frame, summary))

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
        self.create_result_plot(plot1, result.x, result.y, "ROI Intensity", figsize=(1.2, 1.2),
                                image=getattr(result, 'image', None), roi_bounds=getattr(result, 'roi_bounds', None),
                                force_square=True)
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
        if hasattr(result, 'mean_signal') and hasattr(result,
                                                      'mean_reference') and result.mean_signal is not None and result.mean_reference is not None:
            ax2.plot(result.x, result.mean_signal, color='r', marker='o', markersize=3, linestyle='-', label='Signal')
            ax2.plot(result.x, result.mean_reference, color='b', marker='o', markersize=3, linestyle='-',
                     label='Reference')
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
        main_row.pack(side="top", fill="both", expand=True, padx=5, pady=(0, 0))
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
        # --- Store FFT data in result object for saving ---
        try:
            result.fft_freq = freq[mask] * 1e-6  # MHz
            result.fft_amp = np.abs(fft_vals[mask])
        except Exception:
            pass
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
        ax_fft.plot(freq[mask] * 1e-6, np.abs(fft_vals[mask]), color='tab:green')
        ax_fft.set_xlabel('Frequency (MHz)')
        ax_fft.set_ylabel('FFT Amplitude')
        ax_fft.set_title('')
        fig_fft.tight_layout()
        canvas_fft = FigureCanvasTkAgg(fig_fft, master=fft_frame)
        canvas_fft.draw()
        canvas_fft.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Time Domain
        time_frame = ttk.LabelFrame(left_col, text="Time Domain")
        time_frame.pack(side="top", fill="both", expand=True, pady=(5, 0))
        for child in time_frame.winfo_children():
            child.destroy()
        fig_time = Figure(figsize=(6, 2.6), dpi=100)  # Increased width/height
        ax_time = fig_time.add_subplot(111)
        ax_time.plot(uniform_time * 1e9, norm_signal, 'ks', label="Raw Data", markersize=3, linestyle='None')
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
        summary.pack(side="left", fill="y", padx=(8, 0))
        summary.pack_propagate(False)
        contrast = (np.max(result.y) - np.min(result.y)) / np.max(result.y) * 100 if np.max(result.y) != 0 else 0
        mean_intensity = np.mean(result.y)
        summary_text = f"Ramsey analysis complete.\nContrast: {contrast:.2f}%\nMean Intensity: {mean_intensity:.3g}"
        ttk.Label(summary, text=summary_text, justify="left").pack(anchor="nw")
        # --- Fit button logic for Ramsey ---
        if self.fit_param_entries and "fit_btn" in self.fit_param_entries:
            self.fit_param_entries["fit_btn"].config(command=lambda: self.run_ramsey_fit(result, time_frame, summary))

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
        self.create_result_plot(plot1, result.x, result.y, "ROI Intensity", figsize=(1.2, 1.2),
                                image=getattr(result, 'image', None), roi_bounds=getattr(result, 'roi_bounds', None),
                                force_square=True)
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
        main_row.pack(side="top", fill="both", expand=True, padx=5, pady=(0, 0))
        # Time Domain
        time_frame = ttk.LabelFrame(main_row, text="Time Domain")
        time_frame.pack(side="left", fill="both", expand=True, pady=(5, 0))
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
        summary.pack(side="left", fill="y", padx=(8, 0))
        summary.pack_propagate(False)
        contrast = (np.max(result.y) - np.min(result.y)) / np.max(result.y) * 100 if np.max(result.y) != 0 else 0
        mean_intensity = np.mean(result.y)
        summary_text = f"T1 analysis complete.\nContrast: {contrast:.2f}%\nMean Intensity: {mean_intensity:.3g}"
        ttk.Label(summary, text=summary_text, justify="left").pack(anchor="nw")

        # --- Fit button logic for T1 ---
        if self.fit_param_entries and "fit_btn" in self.fit_param_entries:
            self.fit_param_entries["fit_btn"].config(command=lambda: self.run_t1_fit(result, time_frame, summary))

    def create_sample_plot(self, frame, title, figsize=(4, 2.5)):
        fig = Figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        ax.plot([0, 1, 2, 3], [10, 20, 10, 30])
        ax.set_title(title)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_result_plot(self, frame, x, y, title, figsize=(4, 2.5), image=None, roi_bounds=None, filtered=None,
                           fit_curve=None, force_square=False):
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
                # --- If FFT data is present, write as columns side by side ---
                if 'fft_freq' in fit_result and 'fft_amp' in fit_result:
                    f.write("x,y,y_fit,fft_freq (MHz),fft_amp\n")
                    x = fit_result['x']
                    y = fit_result['y']
                    y_fit = fit_result['y_fit']
                    fft_freq = fit_result['fft_freq']
                    fft_amp = fit_result['fft_amp']
                    n = max(len(x), len(fft_freq))
                    for i in range(n):
                        x_val = x[i] if i < len(x) else ""
                        y_val = y[i] if i < len(y) else ""
                        yfit_val = y_fit[i] if i < len(y_fit) else ""
                        freq_val = fft_freq[i] if i < len(fft_freq) else ""
                        amp_val = fft_amp[i] if i < len(fft_amp) else ""
                        f.write(f"{x_val},{y_val},{yfit_val},{freq_val},{amp_val}\n")
                else:
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

    def animate_fit_button(self):
        fitting_text = "Fitting..."
        display = fitting_text[:self.fit_anim_progress]
        if self.fit_anim_done:
            if self.fit_param_entries and "fit_btn" in self.fit_param_entries:
                self.fit_param_entries["fit_btn"].config(text="Run Fit", state="normal")
            return
        if self.fit_param_entries and "fit_btn" in self.fit_param_entries:
            self.fit_param_entries["fit_btn"].config(text=display)
        if self.fit_animating and not self.fit_anim_done:
            self.fit_anim_progress = (self.fit_anim_progress + 1) % (len(fitting_text) + 1)
            self.root.after(80, self.animate_fit_button)

    def start_fit_animation(self):
        self.fit_anim_progress = 0
        self.fit_animating = True
        self.fit_anim_done = False
        if self.fit_param_entries and "fit_btn" in self.fit_param_entries:
            self.fit_param_entries["fit_btn"].config(state="disabled")
        self.animate_fit_button()

    def finish_fit_animation(self):
        self.fit_animating = False
        self.fit_anim_done = True
        if self.fit_param_entries and "fit_btn" in self.fit_param_entries:
            self.fit_param_entries["fit_btn"].config(text="Run Fit", state="normal")

    def run_rabi_fit(self, result, data_frame, summary):
        self.start_fit_animation()
        threading.Thread(target=self._run_rabi_fit_thread, args=(result, data_frame, summary), daemon=True).start()

    def _run_rabi_fit_thread(self, result, data_frame, summary):
        try:
            from scipy.optimize import curve_fit
            if isinstance(result, dict):
                x = result['x']
                y = result['y']
            else:
                x = result.x
                y = result.y
            A = float(self.fit_param_entries["Amplitude"].get())
            omega = float(self.fit_param_entries["Omega"].get())
            phase = float(self.fit_param_entries["Phase"].get())
            decay = float(self.fit_param_entries["Decay Rate"].get())
            B = float(self.fit_param_entries["B"].get())
            C = float(self.fit_param_entries["C"].get())
            p0 = [A, omega, phase, decay, B, C]
            try:
                popt, pcov = curve_fit(self.rabi_model, x, y, p0=p0, maxfev=10000)
                yfit = self.rabi_model(x, *popt)
                fit_result = {
                    "x": x,
                    "y": y,
                    "yfit": yfit,
                    "popt": popt,
                    "param_names": ["A", "omega", "phase", "decay", "B", "C"]
                }
                self.root.after(0, lambda: self._handle_rabi_fit_result(fit_result, None, data_frame, summary, result))
            except Exception as e:
                self.root.after(0, lambda: self._handle_rabi_fit_result(None, str(e), data_frame, summary, result))
        finally:
            self.root.after(0, self.finish_fit_animation)

    def _handle_rabi_fit_result(self, fit_result, error, data_frame, summary, result):
        if error:
            messagebox.showerror("Fit Error", f"Rabi fit failed: {error}")
            return
        idx = self.results_notebook.index(self.results_notebook.select())
        tab_widget, _ = self.result_tabs[idx]
        self.tab_fit_results[tab_widget] = {
            "x": fit_result["x"],
            "y": fit_result["y"],
            "y_fit": fit_result["yfit"],
            "params": fit_result["popt"],
            "param_names": fit_result["param_names"],
            "equation": "y = A * cos(omega * x + phase) * exp(-x / decay) + B * x + C (Rabi)"
        }
        # Update the Data plot in place (clear previous plot)
        for widget in data_frame.winfo_children():
            widget.destroy()
        from matplotlib.figure import Figure
        fig_fit = Figure(figsize=(6, 2.6), dpi=100)
        ax_fit = fig_fit.add_subplot(111)
        ax_fit.plot(fit_result["x"], fit_result["y"], 'ks', label="Raw Data", markersize=3, linestyle='None')
        ax_fit.plot(fit_result["x"], fit_result["yfit"], 'r-', linewidth=1.5, label='Fit')
        ax_fit.set_xlabel('time (ns)')
        ax_fit.set_ylabel('Intensity')
        ax_fit.set_title("")
        ax_fit.legend()
        fig_fit.tight_layout()
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas_fit = FigureCanvasTkAgg(fig_fit, master=data_frame)
        canvas_fit.draw()
        canvas_fit.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        add_toolbar(canvas_fit, data_frame)
        # Show fit parameters in summary
        fit_text = "Fit parameters:\n"
        for name, val in zip(fit_result["param_names"], fit_result["popt"]):
            fit_text += f"{name}: {val:.4g}\n"
        for widget in summary.winfo_children():
            widget.destroy()
        ttk.Label(summary, text=fit_text, justify="left").pack(anchor="nw")

    def populate_external_result_tab(self, tab, result):
        # Clear any existing widgets
        for widget in tab.winfo_children():
            widget.destroy()
        protocol = result.get('protocol', None)
        if protocol == "Ramsey":
            # --- Layout: FFT + Time Domain (vertical), Summary (right) ---
            main_row = ttk.Frame(tab)
            main_row.pack(side="top", fill="both", expand=True, padx=5, pady=(0, 0))
            # Left column: FFT + Time Domain (vertical stack)
            left_col = ttk.Frame(main_row)
            left_col.pack(side="left", fill="both", expand=False)
            # FFT
            fft_frame = ttk.LabelFrame(left_col, text="FFT")
            fft_frame.pack(side="top", fill="x", expand=False)
            import numpy as np
            from scipy.interpolate import interp1d
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            x_ns = np.array(result['x'])
            y = np.array(result['y'])
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
            # --- Store FFT data in result dict for saving ---
            try:
                result['fft_freq'] = freq[mask] * 1e-6  # MHz
                result['fft_amp'] = np.abs(fft_vals[mask])
            except Exception:
                pass
            fig_fft = Figure(figsize=(6, 1.6), dpi=100)
            ax_fft = fig_fft.add_subplot(111)
            ax_fft.tick_params(axis='both', which='major', labelsize=8)
            ax_fft.tick_params(axis='both', which='minor', labelsize=7)
            ax_fft.yaxis.label.set_size(9)
            ax_fft.xaxis.label.set_size(9)
            ax_fft.yaxis.labelpad = 2
            ax_fft.xaxis.labelpad = 2
            fig_fft.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.28)
            ax_fft.plot(freq[mask] * 1e-6, np.abs(fft_vals[mask]), color='tab:green')
            ax_fft.set_xlabel('Frequency (MHz)')
            ax_fft.set_ylabel('FFT Amplitude')
            ax_fft.set_title('')
            fig_fft.tight_layout()
            canvas_fft = FigureCanvasTkAgg(fig_fft, master=fft_frame)
            canvas_fft.draw()
            canvas_fft.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            # Time Domain
            time_frame = ttk.LabelFrame(left_col, text="Time Domain")
            time_frame.pack(side="top", fill="both", expand=True, pady=(5, 0))
            for child in time_frame.winfo_children():
                child.destroy()
            fig_time = Figure(figsize=(6, 2.6), dpi=100)
            ax_time = fig_time.add_subplot(111)
            ax_time.plot(uniform_time * 1e9, norm_signal, 'ks', label="Raw Data", markersize=3, linestyle='None')
            ax_time.set_xlabel('time (ns)')
            ax_time.set_ylabel('Normalized Intensity')
            ax_time.set_title("")
            ax_time.legend()
            fig_time.tight_layout()
            canvas_time = FigureCanvasTkAgg(fig_time, master=time_frame)
            canvas_time.draw()
            canvas_time.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            try:
                add_toolbar(canvas_time, time_frame)
            except Exception as e:
                print(f"Toolbar error: {e}")
            # Right column: Summary (spans both FFT and Time Domain)
            summary = ttk.LabelFrame(main_row, text="Summary", width=320)
            summary.pack(side="left", fill="y", padx=(8, 0))
            summary.pack_propagate(False)
            contrast = (np.max(y) - np.min(y)) / np.max(y) * 100 if np.max(y) != 0 else 0
            mean_intensity = np.mean(y)
            summary_text = f"External Ramsey data loaded.\nContrast: {contrast:.2f}%\nMean Intensity: {mean_intensity:.3g}"
            ttk.Label(summary, text=summary_text, justify="left").pack(anchor="nw")
            # --- Fit button logic for external Ramsey ---
            self._external_ramsey_time_frame = time_frame
            self._external_ramsey_summary_frame = summary
            if self.fit_param_entries and "fit_btn" in self.fit_param_entries:
                self.fit_param_entries["fit_btn"].config(
                    command=lambda: self.run_ramsey_fit(result, self._external_ramsey_time_frame,
                                                        self._external_ramsey_summary_frame))
        else:
            # Default: simple data plot and summary
            main_row = ttk.Frame(tab)
            main_row.pack(side="top", fill="both", expand=True, padx=5, pady=(0, 0))
            data_frame = ttk.LabelFrame(main_row, text="Data")
            data_frame.pack(side="left", fill="both", expand=True, pady=(5, 0))
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import numpy as np
            fig_data = Figure(figsize=(6, 2.6), dpi=100)
            ax_data = fig_data.add_subplot(111)
            x = np.array(result['x'])
            y = np.array(result['y'])
            yerr = result.get('yerr', None)
            if yerr is not None:
                ax_data.errorbar(x, y, yerr=yerr, fmt='ks', label="Data", markersize=5, linestyle='None')
            else:
                ax_data.plot(x, y, 'ks', label="Data", markersize=5, linestyle='None')
            ax_data.set_xlabel("x")
            ax_data.set_ylabel("y")
            ax_data.set_title("")
            ax_data.legend()
            fig_data.tight_layout()
            canvas_data = FigureCanvasTkAgg(fig_data, master=data_frame)
            canvas_data.draw()
            canvas_data.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            add_toolbar(canvas_data, data_frame)
            summary = ttk.LabelFrame(main_row, text="Summary", width=320)
            summary.pack(side="left", fill="y", padx=(8, 0))
            summary.pack_propagate(False)
            summary_text = f"External data loaded.\nProtocol: {result['protocol']}\nFile: {result['filename']}\nPoints: {len(result['x'])}"
            ttk.Label(summary, text=summary_text, justify="left").pack(anchor="nw")
        # --- Add fit parameter section for external data ---
        # (REMOVED: fit section should only be in the left panel, not here)

    def run_ramsey_fit(self, result, data_frame, summary):
        self.start_fit_animation()
        threading.Thread(target=self._run_ramsey_fit_thread, args=(result, data_frame, summary), daemon=True).start()

    def run_t1_fit(self, result, data_frame, summary):
        self.start_fit_animation()
        threading.Thread(target=self._run_t1_fit_thread, args=(result, data_frame, summary), daemon=True).start()

    def _run_ramsey_fit_thread(self, result, data_frame, summary):
        try:
            import numpy as np
            from scipy.optimize import curve_fit
            from scipy.interpolate import interp1d
            import matplotlib.pyplot as plt
            plt.close('all')
            model = self.fit_param_entries["model"].get()
            # Handle both dict and object result types
            if isinstance(result, dict):
                x_ns = np.array(result['x'])
                y = np.array(result['y'])
            else:
                x_ns = np.array(result.x)
                y = np.array(result.y)
            x_s = x_ns * 1e-9
            uniform_time = np.linspace(x_s.min(), x_s.max(), len(x_s))
            interp_func = interp1d(x_s, y, kind='cubic')
            uniform_signal = interp_func(uniform_time)
            norm_signal = (uniform_signal - np.min(uniform_signal)) / (np.max(uniform_signal) - np.min(uniform_signal))
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
                idx = self.results_notebook.index(self.results_notebook.select())
                tab_widget, _ = self.result_tabs[idx]
                fit_result = {
                    "x": uniform_time,
                    "y": norm_signal,
                    "y_fit": yfit,
                    "params": popt,
                    "param_names": param_names,
                    "equation": equation
                }
                # --- Copy FFT data from input result if present ---
                fft_freq = None
                fft_amp = None
                if isinstance(result, dict):
                    fft_freq = result.get('fft_freq', None)
                    fft_amp = result.get('fft_amp', None)
                    # For external data, FFT may be in result['result_obj']
                    if fft_freq is None and 'result_obj' in result:
                        fft_freq = result['result_obj'].get('fft_freq', None)
                    if fft_amp is None and 'result_obj' in result:
                        fft_amp = result['result_obj'].get('fft_amp', None)
                else:
                    fft_freq = getattr(result, 'fft_freq', None)
                    fft_amp = getattr(result, 'fft_amp', None)
                if fft_freq is not None and fft_amp is not None:
                    fit_result['fft_freq'] = fft_freq
                    fit_result['fft_amp'] = fft_amp
                self.tab_fit_results[tab_widget] = fit_result
            except Exception as e:
                from tkinter import messagebox
                self.root.after(0, lambda: messagebox.showerror("Fit Error", f"Ramsey fit failed: {e}"))
                return
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

            def update_plot():
                for widget in data_frame.winfo_children():
                    widget.destroy()
                from matplotlib.figure import Figure
                fig_time = Figure(figsize=(6, 2.6), dpi=100)
                ax_time = fig_time.add_subplot(111)
                ax_time.plot(uniform_time * 1e9, norm_signal, 'ks', label="Raw Data", markersize=3, linestyle='None')
                ax_time.plot(uniform_time * 1e9, yfit, 'r-', linewidth=1.5, label='Fit')
                ax_time.set_xlabel('time (ns)')
                ax_time.set_ylabel('Normalized Intensity')
                ax_time.set_title("")
                ax_time.legend()
                fig_time.tight_layout()
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                canvas_time = FigureCanvasTkAgg(fig_time, master=data_frame)
                canvas_time.draw()
                canvas_time.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                add_toolbar(canvas_time, data_frame)
                for widget in summary.winfo_children():
                    widget.destroy()
                ttk.Label(summary, text=fit_text, justify="left").pack(anchor="nw")

            self.root.after(0, update_plot)
        finally:
            self.root.after(0, self.finish_fit_animation)

    def _run_t1_fit_thread(self, result, data_frame, summary):
        try:
            import numpy as np
            from scipy.optimize import curve_fit
            def stretched_exp_decay(x, a, T1, beta, c):
                return a * np.exp(-((x / T1) ** beta)) + c

            # Handle both dict and object result types
            if isinstance(result, dict):
                x = np.array(result['x'])
                y = np.array(result['y'])
                yerr = result.get('yerr', None)
            else:
                x = np.array(result.x)
                y = np.array(result.y)
                yerr = getattr(result, 'error', None)
            try:
                p0 = [float(self.fit_param_entries[name].get()) for name in ["Amplitude", "T1", "Beta", "Offset"]]
            except Exception:
                p0 = [0.5, 2.0, 0.8, 0.1]
            try:
                if yerr is not None:
                    popt, pcov = curve_fit(stretched_exp_decay, x, y, p0=p0, sigma=yerr, absolute_sigma=True,
                                           maxfev=10000)
                else:
                    popt, pcov = curve_fit(stretched_exp_decay, x, y, p0=p0, maxfev=10000)
                yfit = stretched_exp_decay(x, *popt)
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
                self.root.after(0, lambda: messagebox.showerror("Fit Error", f"T1 fit failed: {e}"))
                return

            def update_plot():
                for widget in data_frame.winfo_children():
                    widget.destroy()
                from matplotlib.figure import Figure
                fig_time = Figure(figsize=(6, 2.6), dpi=100)
                ax_time = fig_time.add_subplot(111)
                ax_time.errorbar(x, y, yerr=yerr, fmt='ks', label="Data", markersize=3, linestyle='None')
                ax_time.plot(x, yfit, 'r-', linewidth=1.5, label='Fit')
                ax_time.set_xlabel('time (ms)')
                ax_time.set_ylabel('Normalized Intensity')
                ax_time.set_title("")
                ax_time.legend()
                fig_time.tight_layout()
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                canvas_time = FigureCanvasTkAgg(fig_time, master=data_frame)
                canvas_time.draw()
                canvas_time.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                add_toolbar(canvas_time, data_frame)
                fit_text = f"Fit parameters:\na: {popt[0]:.4g}\nT1: {popt[1]:.4g} ms\nbeta: {popt[2]:.4g}\nc: {popt[3]:.4g}\n"
                for widget in summary.winfo_children():
                    widget.destroy()
                ttk.Label(summary, text=fit_text, justify="left").pack(anchor="nw")

            self.root.after(0, update_plot)
        finally:
            self.root.after(0, self.finish_fit_animation)

    def validate_roi(self, return_values=False):
        try:
            x_min = int(self.roi_entries["X start"].get())
            x_max = int(self.roi_entries["X end"].get())
            y_min = int(self.roi_entries["Y start"].get())
            y_max = int(self.roi_entries["Y end"].get())
            print(f"DEBUG: ROI values: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
        except Exception:
            print("DEBUG: ROI values not integers")
            return False, "ROI values must be integers.", None
        if x_min >= x_max or y_min >= y_max:
            print("DEBUG: ROI min >= max")
            return False, "ROI: X start must be less than X end and Y start less than Y end.", None
        if not (0 <= x_min < 512 and 0 < x_max <= 512 and 0 <= y_min < 512 and 0 < y_max <= 512):
            print("DEBUG: ROI out of bounds")
            return False, "ROI must be within data bounds: x [0, 512], y [0, 512]", None
        return True, None, (x_min, x_max, y_min, y_max)


# Helper to add the matplotlib navigation toolbar below a plot
def add_toolbar(canvas, frame):
    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.update()
    toolbar.pack(side=tk.TOP, fill=tk.X)


class CSVColumnSelectorDialog(tk.Toplevel):
    def __init__(self, parent, preview_lines, protocols, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.title("Select Header Row, Columns and Protocol")
        self.resizable(False, False)
        self.result = None
        self.preview_lines = preview_lines
        self.protocols = protocols
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        # Header row selection
        ttk.Label(self, text="Select Header Row:").grid(row=0, column=0, columnspan=4, sticky="w", padx=8, pady=(8, 2))
        self.header_row_var = tk.IntVar(value=0)
        header_frame = ttk.Frame(self)
        header_frame.grid(row=1, column=0, columnspan=4, sticky="ew", padx=8, pady=(0, 8))
        
        # Show preview as a table with radio buttons for header selection
        for i, row in enumerate(preview_lines):
            rb = ttk.Radiobutton(header_frame, variable=self.header_row_var, value=i)
            rb.grid(row=i, column=0, sticky="w")
            for j, val in enumerate(row):
                ttk.Label(header_frame, text=val, borderwidth=1, relief="solid", width=12, anchor="w").grid(row=i, column=j + 1, sticky="w")
        
        # Column and protocol selection
        ttk.Label(self, text="Select Columns and Protocol:").grid(row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(8, 2))
        
        # Dropdowns for x, y, yerr
        ttk.Label(self, text="x column:").grid(row=3, column=0, sticky="e", padx=4, pady=2)
        self.x_var = tk.StringVar()
        self.x_cb = ttk.Combobox(self, textvariable=self.x_var, values=[], state="readonly")
        self.x_cb.grid(row=3, column=1, sticky="w", padx=4, pady=2)
        
        ttk.Label(self, text="y column:").grid(row=3, column=2, sticky="e", padx=4, pady=2)
        self.y_var = tk.StringVar()
        self.y_cb = ttk.Combobox(self, textvariable=self.y_var, values=[], state="readonly")
        self.y_cb.grid(row=3, column=3, sticky="w", padx=4, pady=2)
        
        ttk.Label(self, text="y-error (optional):").grid(row=4, column=0, sticky="e", padx=4, pady=2)
        self.yerr_var = tk.StringVar()
        self.yerr_cb = ttk.Combobox(self, textvariable=self.yerr_var, values=["<none>"], state="readonly")
        self.yerr_cb.current(0)
        self.yerr_cb.grid(row=4, column=1, sticky="w", padx=4, pady=2)
        
        # Protocol dropdown
        ttk.Label(self, text="Protocol:").grid(row=4, column=2, sticky="e", padx=4, pady=2)
        self.protocol_var = tk.StringVar()
        proto_cb = ttk.Combobox(self, textvariable=self.protocol_var, values=protocols, state="readonly")
        proto_cb.grid(row=4, column=3, sticky="w", padx=4, pady=2)
        
        # OK/Cancel buttons
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=5, column=0, columnspan=4, pady=(8, 8))
        self.ok_btn = ttk.Button(btn_frame, text="OK", command=self._on_ok, state="disabled")
        self.ok_btn.pack(side="left", padx=8)
        ttk.Button(btn_frame, text="Cancel", command=self._on_cancel).pack(side="left", padx=8)

        # Enable OK only if header row, x, y, and protocol are selected
        def validate_inputs(*args):
            if (self.header_row_var.get() is not None and 
                self.x_var.get() and 
                self.y_var.get() and 
                self.protocol_var.get()):
                self.ok_btn.config(state="normal")
            else:
                self.ok_btn.config(state="disabled")

        # Update column dropdowns when header row changes
        def update_columns(*args):
            header_row = self.header_row_var.get()
            if header_row < len(preview_lines):
                headers = preview_lines[header_row]
                self.x_cb['values'] = headers
                self.y_cb['values'] = headers
                self.yerr_cb['values'] = ["<none>"] + headers
                # Auto-select first column as x and second column as y if available
                if len(headers) >= 1:
                    self.x_var.set(headers[0])
                if len(headers) >= 2:
                    self.y_var.set(headers[1])
                # Reset y-error to none
                self.yerr_var.set('<none>')

        self.header_row_var.trace_add('write', update_columns)
        
        self.x_var.trace_add('write', validate_inputs)
        self.y_var.trace_add('write', validate_inputs)
        self.protocol_var.trace_add('write', validate_inputs)
        
        # Initialize column dropdowns
        update_columns()
        
        # Center the dialog on the parent
        self.update_idletasks()
        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        parent_w = parent.winfo_width()
        parent_h = parent.winfo_height()
        win_w = self.winfo_width()
        win_h = self.winfo_height()
        x = parent_x + (parent_w // 2) - (win_w // 2)
        y = parent_y + (parent_h // 2) - (win_h // 2)
        self.geometry(f"+{x}+{y}")
        self.transient(parent)
        self.grab_set()
        self.wait_window(self)

    def _on_ok(self):
        self.result = {
            'header_row': self.header_row_var.get(),
            'x': self.x_var.get(),
            'y': self.y_var.get(),
            'yerr': self.yerr_var.get() if self.yerr_var.get() != '<none>' else None,
            'protocol': self.protocol_var.get()
        }
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()





if __name__ == "__main__":
    print("DEBUG: Running as main script")
    root = tk.Tk()
    app = NVAnalysisApp(root)
    root.mainloop()
