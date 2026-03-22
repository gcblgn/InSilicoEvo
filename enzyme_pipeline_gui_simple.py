"""
ISEF 2026 - Simplified Pipeline GUI (Minimal Dependencies)
==========================================================
Falls back to tkinter if PyQt6/PySide6 not available
"""

import sys
import os
import subprocess
import webbrowser
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from threading import Thread
from pathlib import Path


class PipelineGUI:
    """Simplified GUI using tkinter (included with Python)"""

    def __init__(self, root):
        self.root = root
        self.root.title("ISEF 2026 - Enzyme Engineering Pipeline")
        self.root.geometry("1200x820")

        self.current_step = 0
        self.worker_thread = None
        self.process = None

        # Dictionary to store user-entered parameters for each step
        self.step_params = {}

        # File name chain based on adaptation type
        self._file_names = {
            "cold": {
                "s0_output":  "cold_dataset.csv",
                "s1_input":   "cold_dataset.csv",
                "s1_output":  "cold_dataset_features.csv",
                "s2_input":   "cold_dataset_features.csv",
            },
            "heat": {
                "s0_output":  "heat_dataset.csv",
                "s1_input":   "heat_dataset.csv",
                "s1_output":  "heat_dataset_features.csv",
                "s2_input":   "heat_dataset_features.csv",
            },
        }

        # Pipeline steps
        self.pipeline_steps = [
            {
                "number": 0,
                "title": "Data Extraction",
                "description": "Extract enzyme data\nfrom BRENDA database",
                "script_cold": "0_extract_data_cold.py",
                "script_heat": "0_extract_data_heat.py"
            },
            {
                "number": 1,
                "title": "Feature Calculation",
                "description": "Calculate biochemical\nand structural features",
                "script": "1_calculate_enzyme_features.py"
            },
            {
                "number": 2,
                "title": "AutoML Training",
                "description": "Train & compare ML models\nwith PyCaret AutoML",
                "script": "2_train_model_automlV3.py"
            },
            {
                "number": 3,
                "title": "Directed Evolution",
                "description": "Optimize enzyme sequence\nwith AI-guided evolution",
                "script": "3_directed_evolutionV5.py"
            },
            {
                "number": 4,
                "title": "ESM-2 Validation",
                "description": "Validate variant sequences\non Google Colab (GPU)",
                "url": "https://colab.research.google.com/drive/1gpMmO41AB-xkJo-bBlyYAlw6ShsMEfaa?usp=sharing"
            },
            {
                "number": 5,
                "title": "ESMFold Prediction",
                "description": "Predict 3D protein structure\nvia ESMFold API (Meta)",
                "script": "5_esmfold_structure_prediction.py"
            }
        ]

        self.step_status = ["pending"] * len(self.pipeline_steps)
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=70)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title = tk.Label(
            header_frame,
            text="Computational Enzyme Engineering Pipeline",
            font=("Arial", 16, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title.pack(pady=10)
        
        subtitle = tk.Label(
            header_frame,
            text="ISEF 2026 - AI-Assisted Enzyme Optimization",
            font=("Arial", 9),
            bg="#2c3e50",
            fg="#ecf0f1"
        )
        subtitle.pack()
        
        # Main content
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel: Pipeline steps
        left_frame = tk.LabelFrame(main_frame, text="Pipeline Steps", font=("Arial", 10, "bold"))
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        
        self.step_labels = []
        for step in self.pipeline_steps:
            step_frame = tk.Frame(left_frame, relief=tk.RAISED, borderwidth=2, cursor="hand2")
            step_frame.pack(fill=tk.X, padx=5, pady=3)

            # Step number (large)
            num_label = tk.Label(
                step_frame,
                text=str(step["number"]),
                font=("Arial", 16, "bold"),
                width=2,
                height=2,
                bg="#95a5a6",
                fg="white"
            )
            num_label.pack(side=tk.LEFT, padx=(6, 8), pady=6)

            # Step info
            info_frame = tk.Frame(step_frame)
            info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=4)

            title_label = tk.Label(
                info_frame,
                text=step["title"],
                font=("Arial", 10, "bold"),
                anchor=tk.W
            )
            title_label.pack(fill=tk.X)

            desc_label = tk.Label(
                info_frame,
                text=step["description"],
                font=("Arial", 8),
                fg="#7f8c8d",
                anchor=tk.W,
                justify=tk.LEFT
            )
            desc_label.pack(fill=tk.X)

            # Status
            status_label = tk.Label(
                step_frame,
                text=" Pending",
                font=("Arial", 9),
                fg="#95a5a6"
            )
            status_label.pack(side=tk.RIGHT, padx=5)

            self.step_labels.append({
                "frame": step_frame,
                "number": num_label,
                "status": status_label
            })

            # --- Click event: bind to step frame and all child widgets ---
            step_num = step["number"]
            for widget in [step_frame, num_label, info_frame, title_label, desc_label, status_label]:
                widget.bind("<Button-1>", lambda e, s=step_num: self.go_to_step(s))

        # DNA emoji below steps
        tk.Label(left_frame, text="\U0001F9EC", font=("Arial", 65)).pack(expand=True)

        # Visually highlight the first step as active
        self.highlight_active_step()

        # Right panel: Controls and output
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Configuration
        config_frame = tk.LabelFrame(right_frame, text="Configuration", font=("Arial", 10, "bold"))
        config_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.current_step_label = tk.Label(
            config_frame,
            text="Current Step: 0 - Data Extraction",
            font=("Arial", 11, "bold")
        )
        self.current_step_label.pack(pady=5)
        
        # Parameters (Step 0 example)
        self.params_frame = tk.Frame(config_frame)
        self.params_frame.pack(fill=tk.X, padx=5, pady=5)
        self.create_step0_params()
        
        # Navigation
        nav_frame = tk.Frame(config_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.prev_btn = tk.Button(
            nav_frame,
            text="◀️ Previous",
            command=self.previous_step,
            state=tk.DISABLED
        )
        self.prev_btn.pack(side=tk.LEFT, padx=2)
        
        self.next_btn = tk.Button(
            nav_frame,
            text="Next     ▶️",
            command=self.next_step
        )
        self.next_btn.pack(side=tk.LEFT, padx=2)
        
        # Control buttons
        control_frame = tk.Frame(right_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        self.run_step_btn = tk.Button(
            control_frame,
            text=" Run Current Step",
            command=self.run_current_step,
            bg="#27ae60",
            fg="white",
            font=("Arial", 10, "bold")
        )
        self.run_step_btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        self.run_all_btn = tk.Button(
            control_frame,
            text=" Run All Steps",
            command=self.run_all_steps,
            bg="#e67e22",
            fg="white",
            font=("Arial", 10, "bold")
        )
        self.run_all_btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        self.stop_btn = tk.Button(
            control_frame,
            text=" Stop",
            command=self.stop_execution,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 10, "bold"),
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # Progress bar
        self.progress = ttk.Progressbar(right_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # Console output
        output_frame = tk.LabelFrame(right_frame, text="Console Output", font=("Arial", 10, "bold"))
        output_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Console buttons (pack first so they keep fixed height when window shrinks)
        console_btn_frame = tk.Frame(output_frame)
        console_btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0, 5))

        tk.Button(
            console_btn_frame,
            text="Clear Console",
            command=lambda: self.console.delete(1.0, tk.END)
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            console_btn_frame,
            text="Export Log",
            command=self.export_log
        ).pack(side=tk.LEFT, padx=2)

        self.console = scrolledtext.ScrolledText(
            output_frame,
            height=10,
            bg="#2c3e50",
            fg="#ecf0f1",
            font=("Courier New", 9)
        )
        self.console.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    # ==========================================================================
    # STEP SELECTION AND ACTIVE STEP HIGHLIGHTING
    # ==========================================================================

    def go_to_step(self, step_num):
        """Navigate to the step clicked in the left panel"""
        # Save parameters of the previous step
        self.save_current_step_params()
        self.current_step = step_num
        self.update_current_step()

    def highlight_active_step(self):
        """Visually highlight the active step in the left panel"""
        for i, lbl in enumerate(self.step_labels):
            if i == self.current_step:
                lbl["frame"].config(relief=tk.GROOVE, borderwidth=3, bg="#d6eaf8")
                for child in lbl["frame"].winfo_children():
                    if isinstance(child, tk.Frame):
                        child.config(bg="#d6eaf8")
                        for sub in child.winfo_children():
                            if isinstance(sub, tk.Label):
                                sub.config(bg="#d6eaf8")
            else:
                lbl["frame"].config(relief=tk.RAISED, borderwidth=2, bg="SystemButtonFace")
                for child in lbl["frame"].winfo_children():
                    if isinstance(child, tk.Frame):
                        child.config(bg="SystemButtonFace")
                        for sub in child.winfo_children():
                            if isinstance(sub, tk.Label):
                                sub.config(bg="SystemButtonFace")

    def show_step_config(self, step_num):
        """Show the configuration panel for the selected step"""
        config_creators = {
            0: self.create_step0_params,
            1: self.create_step1_params,
            2: self.create_step2_params,
            3: self.create_step3_params,
            4: self.create_step4_params,
            5: self.create_step5_params,
        }
        creator = config_creators.get(step_num)
        if creator:
            creator()

    # ==========================================================================
    # STEP 0: DATA EXTRACTION CONFIGURATION
    # ==========================================================================

    def _get_default_filename(self, key):
        """Return the default filename based on adaptation type"""
        adapt = self.step_params.get("s0_adaptation", "cold")
        return self._file_names[adapt][key]

    def _on_adaptation_changed(self, *args):
        """Update file names when radio button changes"""
        adapt = self.adaptation_type.get()
        self.step_params["s0_adaptation"] = adapt
        names = self._file_names[adapt]

        # Update Step 0 output field (if visible on screen)
        if hasattr(self, 'step0_output') and self.step0_output.winfo_exists():
            self.step0_output.delete(0, tk.END)
            self.step0_output.insert(0, names["s0_output"])

        # Update other step parameters as well
        self.step_params["s0_output"] = names["s0_output"]
        self.step_params["s1_input"]  = names["s1_input"]
        self.step_params["s1_output"] = names["s1_output"]
        self.step_params["s2_input"]  = names["s2_input"]

    def create_step0_params(self):
        """Create parameter inputs for Step 0"""
        # Clear existing
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        # Adaptation type selection
        type_frame = tk.Frame(self.params_frame)
        type_frame.grid(row=0, column=0, columnspan=3, pady=10, sticky=tk.W)

        tk.Label(
            type_frame,
            text="Adaptation Type:",
            font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.adaptation_type = tk.StringVar(value=self.step_params.get("s0_adaptation", "cold"))
        # Automatically update file names when changed
        self.adaptation_type.trace_add("write", self._on_adaptation_changed)

        cold_radio = tk.Radiobutton(
            type_frame,
            text="Cold Adaptation (Lower Topt)",
            variable=self.adaptation_type,
            value="cold",
            font=("Arial", 9)
        )
        cold_radio.pack(side=tk.LEFT, padx=5)

        heat_radio = tk.Radiobutton(
            type_frame,
            text="Heat Adaptation (Higher Topt)",
            variable=self.adaptation_type,
            value="heat",
            font=("Arial", 9)
        )
        heat_radio.pack(side=tk.LEFT, padx=5)

        # Separator
        separator = tk.Frame(self.params_frame, height=2, bg="#bdc3c7")
        separator.grid(row=1, column=0, columnspan=3, sticky="ew", pady=10)

        # Database file
        tk.Label(self.params_frame, text="Database File:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.db_file = tk.Entry(self.params_frame, width=30)
        self.db_file.insert(0, self.step_params.get("s0_db_file", "brenda.sql"))
        self.db_file.grid(row=2, column=1, pady=2, padx=5)
        browse_btn = tk.Button(
            self.params_frame,
            text="Browse",
            command=lambda: self.browse_file(self.db_file)
        )
        browse_btn.grid(row=2, column=2, pady=2)

        # Output file - default based on adaptation type
        adapt = self.adaptation_type.get()
        default_output = self.step_params.get("s0_output", self._file_names[adapt]["s0_output"])

        tk.Label(self.params_frame, text="Output File:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.step0_output = tk.Entry(self.params_frame, width=30)
        self.step0_output.insert(0, default_output)
        self.step0_output.grid(row=3, column=1, pady=2, padx=5)


    # ==========================================================================
    # STEP 1: FEATURE CALCULATION CONFIGURATION
    # ==========================================================================

    def create_step1_params(self):
        """Create parameter inputs for Step 1 - Feature Calculation"""
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        adapt = self.step_params.get("s0_adaptation", "cold")

        # Input file
        tk.Label(self.params_frame, text="Input CSV File:", font=("Arial", 9, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=2)
        self.step1_input = tk.Entry(self.params_frame, width=30)
        self.step1_input.insert(0, self.step_params.get("s1_input", self._file_names[adapt]["s1_input"]))
        self.step1_input.grid(row=0, column=1, pady=2, padx=5)
        tk.Button(self.params_frame, text="Browse",
                  command=lambda: self.browse_file(self.step1_input)).grid(row=0, column=2, pady=2)

        # Output file
        tk.Label(self.params_frame, text="Output File:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.step1_output = tk.Entry(self.params_frame, width=30)
        self.step1_output.insert(0, self.step_params.get("s1_output", self._file_names[adapt]["s1_output"]))
        self.step1_output.grid(row=1, column=1, pady=2, padx=5)

        # Info
        info_text = (
            "Calculates biochemical features using enzyme_feature_lib:\n"
            "  - Physicochemical properties (MW, GRAVY, pI, ...)\n"
            "  - Thermostability features (aliphatic index, Boman, ...)\n"
            "  - Amino acid & dipeptide frequencies\n"
            "  - Flexibility properties\n\n"
            "Requires: enzyme_feature_lib.py in the same directory"
        )
        tk.Label(self.params_frame, text=info_text, font=("Arial", 8), fg="#7f8c8d",
                 justify=tk.LEFT).grid(row=2, column=0, columnspan=3, pady=(10, 0), sticky=tk.W)

    # ==========================================================================
    # STEP 2: AUTOML TRAINING CONFIGURATION
    # ==========================================================================

    def create_step2_params(self):
        """Create parameter inputs for Step 2 - AutoML Training"""
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        adapt = self.step_params.get("s0_adaptation", "cold")

        # Input file
        tk.Label(self.params_frame, text="Input Features CSV:", font=("Arial", 9, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=2)
        self.step2_input = tk.Entry(self.params_frame, width=30)
        self.step2_input.insert(0, self.step_params.get("s2_input", self._file_names[adapt]["s2_input"]))
        self.step2_input.grid(row=0, column=1, pady=2, padx=5)
        tk.Button(self.params_frame, text="Browse",
                  command=lambda: self.browse_file(self.step2_input)).grid(row=0, column=2, pady=2)

        # --- Target Variable ---
        tk.Label(self.params_frame, text="Target Variable:", font=("Arial", 9, "bold")).grid(
            row=1, column=0, sticky=tk.W, pady=2)
        self.step2_target = ttk.Combobox(
            self.params_frame, values=["Topt", "OGT"], state="readonly", width=10
        )
        saved_target = self.step_params.get("s2_target", "Topt")
        self.step2_target.set(saved_target)
        self.step2_target.grid(row=1, column=1, sticky=tk.W, pady=2, padx=5)

        # --- CV Folds ---
        tk.Label(self.params_frame, text="CV Folds:", font=("Arial", 9, "bold")).grid(
            row=2, column=0, sticky=tk.W, pady=2)
        self.step2_fold = ttk.Combobox(
            self.params_frame, values=["2", "3", "4", "5"], state="readonly", width=5
        )
        saved_fold = str(self.step_params.get("s2_fold", "3"))
        self.step2_fold.set(saved_fold)
        self.step2_fold.grid(row=2, column=1, sticky=tk.W, pady=2, padx=5)

        # --- Model Selection (Checkbuttons) ---
        tk.Label(self.params_frame, text="Models:", font=("Arial", 9, "bold")).grid(
            row=3, column=0, sticky=tk.NW, pady=2)

        models_frame = tk.Frame(self.params_frame)
        models_frame.grid(row=3, column=1, columnspan=2, sticky=tk.W, pady=2, padx=5)

        saved_models = self.step_params.get("s2_models",
                                            {"catboost": True, "lightgbm": True, "xgboost": True,
                                             "rf": True, "et": True})

        self.step2_model_vars = {}
        model_labels = [
            ("catboost", "CatBoost"),
            ("lightgbm", "LightGBM"),
            ("xgboost", "XGBoost"),
            ("rf", "Random Forest"),
            ("et", "Extra Trees"),
        ]
        for model_id, display_name in model_labels:
            var = tk.BooleanVar(value=saved_models.get(model_id, True))
            self.step2_model_vars[model_id] = var
            tk.Checkbutton(models_frame, text=display_name, variable=var,
                           font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 8))

        # Generate plots checkbox
        self.step2_plots = tk.BooleanVar(value=self.step_params.get("s2_plots", True))
        tk.Checkbutton(self.params_frame, text="Generate plots (Residual, Error, Feature Importance)",
                       variable=self.step2_plots, font=("Arial", 9)).grid(
            row=4, column=0, columnspan=3, sticky=tk.W, pady=5)

    # ==========================================================================
    # STEP 3: DIRECTED EVOLUTION CONFIGURATION
    # ==========================================================================

    def create_step3_params(self):
        """Create parameter inputs for Step 3 - Directed Evolution (three-column layout)"""
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        # --- Three-column container ---
        columns_frame = tk.Frame(self.params_frame)
        columns_frame.pack(fill=tk.X, pady=2)

        # ========== COLUMN 1 ==========
        col1 = tk.Frame(columns_frame)
        col1.pack(side=tk.LEFT, anchor=tk.N, padx=(0, 15))

        r = 0
        # UniProt ID + Fetch
        tk.Label(col1, text="UniProt ID:", font=("Arial", 9, "bold")).grid(
            row=r, column=0, sticky=tk.W, pady=2)
        uniprot_frame = tk.Frame(col1)
        uniprot_frame.grid(row=r, column=1, sticky=tk.W, pady=2, padx=5)
        self.step3_uniprot = tk.Entry(uniprot_frame, width=12)
        self.step3_uniprot.insert(0, self.step_params.get("s3_uniprot", "E5BBQ3"))
        self.step3_uniprot.pack(side=tk.LEFT)
        tk.Button(uniprot_frame, text="Fetch", width=5,
                  command=self.fetch_sequence).pack(side=tk.LEFT, padx=(4, 0))

        r += 1
        # Current Topt (read-only, fetched from CSV)
        tk.Label(col1, text="Topt (C):").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.step3_current_topt = tk.Entry(col1, width=12, state=tk.DISABLED,
                                           disabledbackground="#f0f0f0",
                                           disabledforeground="#333333")
        self.step3_current_topt.grid(row=r, column=1, sticky=tk.W, pady=2, padx=5)

        r += 1
        # Target Topt
        tk.Label(col1, text="Target Topt (C):", font=("Arial", 9, "bold")).grid(
            row=r, column=0, sticky=tk.W, pady=2)
        self.step3_target = tk.Entry(col1, width=12)
        self.step3_target.insert(0, self.step_params.get("s3_target", "35.0"))
        self.step3_target.grid(row=r, column=1, sticky=tk.W, pady=2, padx=5)

        # ========== COLUMN 2 ==========
        col2 = tk.Frame(columns_frame)
        col2.pack(side=tk.LEFT, anchor=tk.N, padx=(0, 15))

        r = 0
        # Catalytic Triad
        tk.Label(col2, text="Catalytic Triad:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.step3_catalytic = tk.Entry(col2, width=14)
        self.step3_catalytic.insert(0, self.step_params.get("s3_catalytic", "130,176,208"))
        self.step3_catalytic.grid(row=r, column=1, sticky=tk.W, pady=2, padx=5)

        r += 1
        # Oxyanion Hole
        tk.Label(col2, text="Oxyanion Hole:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.step3_oxyanion = tk.Entry(col2, width=14)
        self.step3_oxyanion.insert(0, self.step_params.get("s3_oxyanion", "60,131,132"))
        self.step3_oxyanion.grid(row=r, column=1, sticky=tk.W, pady=2, padx=5)

        r += 1
        # Substrate Binding
        tk.Label(col2, text="Substrate Binding:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.step3_substrate = tk.Entry(col2, width=14)
        self.step3_substrate.insert(0, self.step_params.get("s3_substrate", ""))
        self.step3_substrate.grid(row=r, column=1, sticky=tk.W, pady=2, padx=5)

        # ========== COLUMN 3 ==========
        col3 = tk.Frame(columns_frame)
        col3.pack(side=tk.LEFT, anchor=tk.N)

        r = 0
        # Max Iterations
        tk.Label(col3, text="Max Iterations:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.step3_iterations = tk.Entry(col3, width=10)
        self.step3_iterations.insert(0, self.step_params.get("s3_iterations", "150"))
        self.step3_iterations.grid(row=r, column=1, sticky=tk.W, pady=2, padx=5)

        r += 1
        # Population Size
        tk.Label(col3, text="Population Size:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.step3_population = tk.Entry(col3, width=10)
        self.step3_population.insert(0, self.step_params.get("s3_population", "300"))
        self.step3_population.grid(row=r, column=1, sticky=tk.W, pady=2, padx=5)

        r += 1
        # Mutation Rate
        tk.Label(col3, text="Base Mutation Rate:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.step3_mutation = tk.Entry(col3, width=10)
        self.step3_mutation.insert(0, self.step_params.get("s3_mutation", "0.05"))
        self.step3_mutation.grid(row=r, column=1, sticky=tk.W, pady=2, padx=5)

        # ========== INITIAL SEQUENCE (READ-ONLY) ==========
        tk.Label(self.params_frame, text="Initial Sequence (auto-detected from dataset):",
                 font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(10, 2))

        seq_frame = tk.Frame(self.params_frame)
        seq_frame.pack(fill=tk.X, pady=(0, 5))

        self.step3_sequence_text = tk.Text(seq_frame, height=4, wrap=tk.NONE,
                                           font=("Courier", 9), bg="#f0f0f0")

        seq_vscroll = tk.Scrollbar(seq_frame, orient=tk.VERTICAL,
                                   command=self.step3_sequence_text.yview)
        seq_hscroll = tk.Scrollbar(seq_frame, orient=tk.HORIZONTAL,
                                   command=self.step3_sequence_text.xview)

        self.step3_sequence_text.configure(yscrollcommand=seq_vscroll.set,
                                           xscrollcommand=seq_hscroll.set)

        seq_vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        seq_hscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.step3_sequence_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.step3_sequence_text.insert(tk.END,
            "(Sequence will be loaded from dataset CSV at runtime)")
        self.step3_sequence_text.configure(state=tk.DISABLED)

    def fetch_sequence(self):
        """Fetch sequence and topt from dataset CSV using UniProt ID."""
        uniprot_id = self.step3_uniprot.get().strip()
        if not uniprot_id:
            messagebox.showwarning("Warning", "Please enter a UniProt ID.")
            return

        adapt = self.step_params.get("s0_adaptation", "cold")
        dataset_csv = self.step_params.get("s0_output", self._file_names[adapt]["s0_output"])

        if not os.path.exists(dataset_csv):
            messagebox.showerror("Error",
                f"Dataset file not found: {dataset_csv}\n\n"
                "Please run Step 0 (Data Extraction) first.")
            return

        # Read CSV and lookup sequence + topt by uniprot_id
        import csv
        sequence = None
        topt_value = None
        for sep in [';', ',']:
            try:
                with open(dataset_csv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter=sep)
                    fields = reader.fieldnames or []
                    if 'uniprot_id' not in fields:
                        continue
                    for row in reader:
                        if row.get('uniprot_id', '').strip() == uniprot_id:
                            sequence = row.get('sequence', '').strip()
                            # Try topt column (case-insensitive)
                            for col in fields:
                                if col.lower() == 'topt':
                                    topt_value = row.get(col, '').strip()
                                    break
                            break
                if sequence:
                    break
            except Exception:
                continue

        if not sequence:
            messagebox.showerror("Error",
                f"UniProt ID '{uniprot_id}' not found in {dataset_csv}.\n\n"
                "Please check the ID and try again.")
            return

        # Populate Current Topt (read-only)
        self.step3_current_topt.configure(state=tk.NORMAL)
        self.step3_current_topt.delete(0, tk.END)
        if topt_value:
            self.step3_current_topt.insert(0, topt_value)
        else:
            self.step3_current_topt.insert(0, "N/A")
        self.step3_current_topt.configure(state=tk.DISABLED)

        # Populate the read-only sequence text widget
        self.step3_sequence_text.configure(state=tk.NORMAL)
        self.step3_sequence_text.delete("1.0", tk.END)
        self.step3_sequence_text.insert(tk.END, sequence)
        self.step3_sequence_text.configure(state=tk.DISABLED)

    # ==========================================================================
    # STEP 4: ESM-2 VALIDATION (COLAB) CONFIGURATION
    # ==========================================================================

    def create_step4_params(self):
        """Create parameter inputs for Step 4 - ESM-2 Validation (Colab)"""
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        # --- Two-column layout ---
        columns_frame = tk.Frame(self.params_frame)
        columns_frame.pack(fill=tk.BOTH, expand=True, pady=2)

        # ========== LEFT: Colab link + info ==========
        left_col = tk.Frame(columns_frame)
        left_col.pack(side=tk.LEFT, anchor=tk.N, padx=(0, 15), fill=tk.Y)

        tk.Label(left_col, text="Google Colab Notebook:", font=("Arial", 9, "bold")).pack(
            anchor=tk.W, pady=2)

        url = self.pipeline_steps[4]["url"]
        url_label = tk.Label(
            left_col,
            text=url[:60] + "...",
            font=("Arial", 8, "underline"),
            fg="#2980b9",
            cursor="hand2"
        )
        url_label.pack(anchor=tk.W, pady=2)
        url_label.bind("<Button-1>", lambda e: self.open_colab_chrome())

        info_text = (
            "Workflow:\n"
            "  1. Click 'Run Current Step' to open Colab\n"
            "  2. Press 'Copy' to copy the sequence\n"
            "  3. Paste (Ctrl+V) into the Colab notebook\n"
            "  4. Run all cells in the notebook\n"
            "  5. Download results and return here"
        )
        tk.Label(left_col, text=info_text, font=("Arial", 8), fg="#7f8c8d",
                 justify=tk.LEFT).pack(anchor=tk.W, pady=(10, 0))

        # ========== RIGHT: Sequence text + Copy button ==========
        right_col = tk.Frame(columns_frame)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(right_col, text="Generatede Variant Sequence to validate:", font=("Arial", 9, "bold")).pack(
            anchor=tk.W, pady=2)

        text_frame = tk.Frame(right_col)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.step4_sequence_text = tk.Text(text_frame, height=8, wrap=tk.WORD,
                                           font=("Courier", 9))

        s4_vscroll = tk.Scrollbar(text_frame, orient=tk.VERTICAL,
                                  command=self.step4_sequence_text.yview)
        self.step4_sequence_text.configure(yscrollcommand=s4_vscroll.set)

        s4_vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.step4_sequence_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Try to load best_sequence.txt from report folder
        seq_path = os.path.join("report", "best_sequence.txt")
        if os.path.exists(seq_path):
            with open(seq_path, 'r', encoding='utf-8') as f:
                self.step4_sequence_text.insert(tk.END, f.read().strip())
        else:
            self.step4_sequence_text.insert(tk.END,
                "(best_sequence.txt will be available after Step 3 completes)")

        # Copy button
        tk.Button(right_col, text="Copy to Clipboard", width=18,
                  command=self.copy_step4_sequence).pack(anchor=tk.W, pady=(5, 0))

    def open_colab_chrome(self):
        """Open Colab URL in Chrome """
        url = self.pipeline_steps[4]["url"]
        chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        if os.path.exists(chrome):
            subprocess.Popen([
                chrome,
                url,
                "--window-size=1200,800",
                "--window-position=100,100"
            ])
        else:
            webbrowser.open(url)

    def copy_step4_sequence(self):
        """Copy the sequence text to clipboard."""
        content = self.step4_sequence_text.get("1.0", tk.END).strip()
        if content and not content.startswith("("):
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            messagebox.showinfo("Copied", "Sequence copied to clipboard.")
        else:
            messagebox.showwarning("Warning", "No sequence available to copy.")

    # ==========================================================================
    # STEP 5: ESMFOLD STRUCTURE PREDICTION CONFIGURATION
    # ==========================================================================

    def create_step5_params(self):
        """Create parameter inputs for Step 5 - ESMFold Structure Prediction"""
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        # --- Two-column layout ---
        columns_frame = tk.Frame(self.params_frame)
        columns_frame.pack(fill=tk.BOTH, expand=True, pady=2)

        # ========== LEFT: Name + Info ==========
        left_col = tk.Frame(columns_frame)
        left_col.pack(side=tk.LEFT, anchor=tk.N, padx=(0, 15), fill=tk.Y)

        # Variant name
        tk.Label(left_col, text="Generated Variant Name:", font=("Arial", 9, "bold")).pack(
            anchor=tk.W, pady=2)
        self.step5_name = tk.Entry(left_col, width=25)
        self.step5_name.insert(0, self.step_params.get("s5_name", "My Variant"))
        self.step5_name.pack(anchor=tk.W, pady=(0, 8))

        # Info
        info_text = (
            "Predicts 3D structure via ESMFold API (Meta).\n"
            "No GPU required - runs on Meta's servers.\n\n"
            "Outputs (saved to esmfold_results/):\n"
            "  - PDB structure file\n"
            "  - Interactive 3D viewer (3Dmol.js)\n"
            "  - pLDDT confidence profile (PNG + HTML)"
        )
        tk.Label(left_col, text=info_text, font=("Arial", 8), fg="#7f8c8d",
                 justify=tk.LEFT).pack(anchor=tk.W, pady=(5, 0))

        # ========== RIGHT: Sequence text ==========
        right_col = tk.Frame(columns_frame)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(right_col, text="Generated Variant Sequence:", font=("Arial", 9, "bold")).pack(
            anchor=tk.W, pady=2)

        text_frame = tk.Frame(right_col)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.step5_sequence = tk.Text(text_frame, height=8, wrap=tk.WORD,
                                      font=("Courier", 9))
        s5_vscroll = tk.Scrollbar(text_frame, orient=tk.VERTICAL,
                                  command=self.step5_sequence.yview)
        self.step5_sequence.configure(yscrollcommand=s5_vscroll.set)

        s5_vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.step5_sequence.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Always read fresh from best_sequence.txt (may change between steps)
        seq_path = os.path.join("report", "best_sequence.txt")
        if os.path.exists(seq_path):
            with open(seq_path, 'r', encoding='utf-8') as f:
                self.step5_sequence.insert(tk.END, f.read().strip())
        else:
            self.step5_sequence.insert(tk.END,
                "(best_sequence.txt will be available after Step 3 completes)")
        
    def open_html_in_chrome(self, html_file):
        """Open a local HTML file in Chrome --app mode. Fallback to default browser."""
        chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        if os.path.exists(html_file):
            abs_path = "file:///" + os.path.abspath(html_file).replace("\\", "/")
            self.log(f"Opening: {html_file}")
            if os.path.exists(chrome):
                subprocess.Popen([chrome, "--app=" + abs_path, "--window-size=1200,800"])
            else:
                webbrowser.open(abs_path)
        else:
            self.log(f"WARNING: {html_file} not found")

    def open_step5_results(self):
        """Open ESMFold HTML results in Chrome --app mode after Step 5 completes."""
        name = self.step_params.get("s5_name", "Protein").strip()
        results_dir = "esmfold_results"
        self.open_html_in_chrome(os.path.join(results_dir, f"{name}_plddt_interactive.html"))
        self.open_html_in_chrome(os.path.join(results_dir, f"{name}_3d.html"))

    def browse_file(self, entry_widget):
        """Browse for file"""
        filename = filedialog.askopenfilename()
        if filename:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, filename)
    
    def previous_step(self):
        """Go to previous step"""
        if self.current_step > 0:
            self.save_current_step_params()
            self.current_step -= 1
            self.update_current_step()

    def next_step(self):
        """Go to next step"""
        if self.current_step < len(self.pipeline_steps) - 1:
            self.save_current_step_params()
            self.current_step += 1
            self.update_current_step()
    
    def update_current_step(self):
        """Update UI for current step"""
        step = self.pipeline_steps[self.current_step]
        self.current_step_label.config(text=f"Current Step: {step['number']} - {step['title']}")

        self.prev_btn.config(state=tk.NORMAL if self.current_step > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_step < len(self.pipeline_steps) - 1 else tk.DISABLED)

        # Update configuration panel and highlighting
        self.show_step_config(self.current_step)
        self.highlight_active_step()
    
    def update_step_status(self, step_num, status):
        """Update visual status of a step"""
        colors = {
            "pending": {"bg": "#95a5a6", "text": " Pending", "fg": "#95a5a6"},
            "running": {"bg": "#3498db", "text": " Running", "fg": "#3498db"},
            "completed": {"bg": "#27ae60", "text": " Completed", "fg": "#27ae60"},
            "failed": {"bg": "#e74c3c", "text": " Failed", "fg": "#e74c3c"}
        }
        
        self.step_labels[step_num]["number"].config(bg=colors[status]["bg"])
        self.step_labels[step_num]["status"].config(
            text=colors[status]["text"],
            fg=colors[status]["fg"]
        )
        self.step_status[step_num] = status
    
    def log(self, message):
        """Add message to console"""
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)
        self.root.update()
    
    def save_current_step_params(self):
        """Save parameters from active step's GUI fields to step_params"""
        step_num = self.current_step

        if step_num == 0:
            if hasattr(self, 'adaptation_type'):
                self.step_params["s0_adaptation"] = self.adaptation_type.get()
            if hasattr(self, 'db_file'):
                self.step_params["s0_db_file"] = self.db_file.get()
            if hasattr(self, 'step0_output'):
                self.step_params["s0_output"] = self.step0_output.get()

        elif step_num == 1:
            if hasattr(self, 'step1_input'):
                self.step_params["s1_input"] = self.step1_input.get()
            if hasattr(self, 'step1_output'):
                self.step_params["s1_output"] = self.step1_output.get()

        elif step_num == 2:
            if hasattr(self, 'step2_input'):
                self.step_params["s2_input"] = self.step2_input.get()
            if hasattr(self, 'step2_plots'):
                self.step_params["s2_plots"] = self.step2_plots.get()
            if hasattr(self, 'step2_target'):
                self.step_params["s2_target"] = self.step2_target.get()
            if hasattr(self, 'step2_fold'):
                self.step_params["s2_fold"] = self.step2_fold.get()
            if hasattr(self, 'step2_model_vars'):
                self.step_params["s2_models"] = {
                    mid: var.get() for mid, var in self.step2_model_vars.items()
                }

        elif step_num == 3:
            if hasattr(self, 'step3_target'):
                self.step_params["s3_target"] = self.step3_target.get()
            if hasattr(self, 'step3_current_topt'):
                self.step3_current_topt.configure(state=tk.NORMAL)
                self.step_params["s3_current_topt"] = self.step3_current_topt.get()
                self.step3_current_topt.configure(state=tk.DISABLED)
            if hasattr(self, 'step3_uniprot'):
                self.step_params["s3_uniprot"] = self.step3_uniprot.get()
            if hasattr(self, 'step3_catalytic'):
                self.step_params["s3_catalytic"] = self.step3_catalytic.get()
            if hasattr(self, 'step3_oxyanion'):
                self.step_params["s3_oxyanion"] = self.step3_oxyanion.get()
            if hasattr(self, 'step3_substrate'):
                self.step_params["s3_substrate"] = self.step3_substrate.get()
            if hasattr(self, 'step3_iterations'):
                self.step_params["s3_iterations"] = self.step3_iterations.get()
            if hasattr(self, 'step3_population'):
                self.step_params["s3_population"] = self.step3_population.get()
            if hasattr(self, 'step3_mutation'):
                self.step_params["s3_mutation"] = self.step3_mutation.get()
            if hasattr(self, 'step3_sequence_text'):
                self.step3_sequence_text.configure(state=tk.NORMAL)
                self.step_params["s3_sequence"] = self.step3_sequence_text.get("1.0", tk.END).strip()
                self.step3_sequence_text.configure(state=tk.DISABLED)

        elif step_num == 5:
            if hasattr(self, 'step5_sequence'):
                self.step_params["s5_sequence"] = self.step5_sequence.get("1.0", tk.END).strip()
            if hasattr(self, 'step5_name'):
                self.step_params["s5_name"] = self.step5_name.get()

    def build_command(self, step_num):
        """Build command for a step using current configuration"""
        step = self.pipeline_steps[step_num]
        adapt = self.step_params.get("s0_adaptation", "cold")
        fnames = self._file_names[adapt]

        if "url" in step:
            return None

        # --- Step 0: Data Extraction ---
        if step_num == 0:
            if adapt == "cold":
                script = step.get("script_cold", "0_extract_data_cold.py")
            else:
                script = step.get("script_heat", "0_extract_data_heat.py")
            db = self.step_params.get("s0_db_file", "brenda.sql")
            out = self.step_params.get("s0_output", fnames["s0_output"])
            return f'python {script} "{db}" "{out}"'

        # --- Step 1: Feature Calculation ---
        if step_num == 1:
            script = step.get("script", "1_calculate_enzyme_features.py")
            inp = self.step_params.get("s1_input", fnames["s1_input"])
            out = self.step_params.get("s1_output", fnames["s1_output"])
            return f'python {script} "{inp}" -o "{out}"'

        # --- Step 2: AutoML Training ---
        if step_num == 2:
            script = step.get("script", "2_train_model_automlV3.py")
            inp = self.step_params.get("s2_input", fnames["s2_input"])
            cmd = f'python {script} "{inp}"'
            if not self.step_params.get("s2_plots", True):
                cmd += " --no-plots"
            # Target variable
            target = self.step_params.get("s2_target", "Topt").lower()
            cmd += f" --target {target}"
            # CV folds
            fold = self.step_params.get("s2_fold", "3")
            cmd += f" --fold {fold}"
            # Selected models
            model_dict = self.step_params.get("s2_models",
                                              {"catboost": True, "lightgbm": True,
                                               "xgboost": True, "rf": True, "et": True})
            selected = [m for m, v in model_dict.items() if v]
            if selected:
                cmd += " --models " + " ".join(selected)
            return cmd

        # --- Step 3: Directed Evolution ---
        if step_num == 3:
            script = step.get("script", "3_directed_evolutionV5.py")
            cmd = f"python {script}"
            # Target Topt
            target = self.step_params.get("s3_target", "35.0")
            cmd += f" --target-topt {target}"
            # Current Topt (from dataset CSV, fetched via Fetch button)
            cur_topt = self.step_params.get("s3_current_topt", "").strip()
            if cur_topt and cur_topt != "N/A":
                cmd += f" --current-topt {cur_topt}"
            # UniProt ID
            uniprot = self.step_params.get("s3_uniprot", "E5BBQ3").strip()
            cmd += f" --uniprot-id {uniprot}"
            # Dataset CSV (based on cold/heat adaptation selection)
            dataset_csv = self.step_params.get("s0_output", fnames["s0_output"])
            cmd += f' --dataset-csv "{dataset_csv}"'
            # Catalytic triad
            cat = self.step_params.get("s3_catalytic", "130,176,208").strip()
            if cat:
                cmd += f' --catalytic-triad "{cat}"'
            # Oxyanion hole
            oxy = self.step_params.get("s3_oxyanion", "60,131,132").strip()
            if oxy:
                cmd += f' --oxyanion-hole "{oxy}"'
            # Substrate binding
            sub = self.step_params.get("s3_substrate", "").strip()
            if sub:
                cmd += f' --substrate-binding "{sub}"'
            # Iterations, population, mutation rate
            itr = self.step_params.get("s3_iterations", "150")
            cmd += f" --iterations {itr}"
            pop = self.step_params.get("s3_population", "300")
            cmd += f" --population {pop}"
            mut = self.step_params.get("s3_mutation", "0.05")
            cmd += f" --mutation-rate {mut}"
            # Sequence (fetched via Fetch button)
            seq = self.step_params.get("s3_sequence", "").strip()
            if seq and not seq.startswith("("):
                cmd += f' --sequence "{seq}"'
            return cmd

        # --- Step 5: ESMFold Structure Prediction ---
        if step_num == 5:
            script = step.get("script", "5_esmfold_structure_prediction.py")
            seq = self.step_params.get("s5_sequence", "").strip()
            name = self.step_params.get("s5_name", "Protein").strip()
            cmd = f'python {script}'
            if seq and not seq.startswith("("):
                cmd += f' --sequence "{seq}"'
            if name:
                cmd += f' --name "{name}"'
            return cmd

        # Fallback
        script = step.get("script", "")
        return f"python {script}"
    
    def run_step_worker(self, step_num):
        """Worker function to run a step"""
        step = self.pipeline_steps[step_num]
        
        # Special handling for Colab
        if "url" in step:
            self.log(f"\n{'='*70}")
            self.log(f"Opening Google Colab for ESM-2 validation...")
            self.log(f"URL: {step['url']}")
            self.log(f"{'='*70}\n")
            chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
            if os.path.exists(chrome):
                subprocess.Popen([
                    chrome,
                    "--app=" + step["url"],
                    "--window-size=1200,800",
                    "--window-position=100,100"
                ])
            else:
                webbrowser.open(step["url"])
            self.update_step_status(step_num, "completed")
            self.reset_controls()
            messagebox.showinfo(
                "ESM-2 Validation",
                "Google Colab opened in Chrome.\nPaste the sequence from clipboard and run all cells."
            )
            return
        
        cmd = self.build_command(step_num)
        if not cmd:
            return

        self.log(f"\n{'='*70}")
        self.log(f"STEP {step_num} STARTED")
        self.log(f"Command: {cmd}")
        self.log(f"{'='*70}\n")

        try:
            self.process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.log(line.rstrip())

            self.process.wait()

            if self.process.returncode == 0:
                self.log(f"\n{chr(10003)} STEP {step_num} COMPLETED SUCCESSFULLY")
                self.update_step_status(step_num, "completed")
                self.progress['value'] = (step_num + 1) / len(self.pipeline_steps) * 100

                # Open HTML reports in Chrome --app mode
                if step_num == 2:
                    self.open_html_in_chrome(os.path.join("report", "AutoML_Results_Report.html"))
                elif step_num == 3:
                    self.open_html_in_chrome(os.path.join("report", "Directed_Evolution_Report.html"))
                elif step_num == 5:
                    self.open_step5_results()
            else:
                self.log(f"\n{chr(10007)} STEP {step_num} FAILED (exit code: {self.process.returncode})")
                self.update_step_status(step_num, "failed")

        except Exception as e:
            self.log(f"\n{chr(10007)} ERROR: {str(e)}")
            self.update_step_status(step_num, "failed")

        self.reset_controls()
    
    def run_current_step(self):
        """Run current step - save parameters and execute"""
        # First save the active step's parameters
        self.save_current_step_params()

        self.update_step_status(self.current_step, "running")
        self.run_step_btn.config(state=tk.DISABLED)
        self.run_all_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        self.worker_thread = Thread(target=self.run_step_worker, args=(self.current_step,))
        self.worker_thread.start()
    
    def run_all_steps(self):
        """Run all steps sequentially"""
        if messagebox.askyesno("Run All", "Run all pipeline steps?"):
            self.current_step = 0
            self.update_current_step()
            self.run_current_step()
    
    def stop_execution(self):
        """Stop current execution"""
        if self.process:
            self.process.terminate()
            self.log("\n Execution stopped by user")
            self.update_step_status(self.current_step, "failed")
            self.reset_controls()
    
    def reset_controls(self):
        """Re-enable controls"""
        self.run_step_btn.config(state=tk.NORMAL)
        self.run_all_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def export_log(self):
        """Export console to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.console.get(1.0, tk.END))
            messagebox.showinfo("Success", f"Log exported to {filename}")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
