import threading
import tkinter as tk
from tkinter import ttk, font as tkfont

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---- Color palette (modern dark-accent light theme) ----
COLORS = {
    "bg":         "#F5F7FA",
    "surface":    "#FFFFFF",
    "border":     "#E1E5EB",
    "text":       "#1F2937",
    "muted":      "#6B7280",
    "primary":    "#4F46E5",
    "primary_hv": "#4338CA",
    "accent":     "#10B981",
    "danger":     "#EF4444",
    "warn":       "#F59E0B",
}


class PPLDetector:
    """PPL-based anomaly detector for educational purposes."""

    def __init__(self, model_id: str = "gpt2"):
        print(f"Loading model '{model_id}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def calculate_ppl(self, text: str) -> float:
        """
        Calculate the perplexity (PPL) of the input text.
        Higher values indicate text that is more "unpredictable" to the model.
        """
        if not text.strip():
            return 0.0

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"], labels=inputs["input_ids"]
            )
            ppl = torch.exp(outputs.loss).item()

        return ppl

    def detect_injection(self, text: str, threshold: float = 100.0):
        """
        Return (is_anomalous, ppl).
        If ppl exceeds the threshold the text is flagged as anomalous.
        """
        ppl = self.calculate_ppl(text)
        return ppl > threshold, ppl


class PPLApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.detector = None

        self.root.title("PPL Anomaly Detector  ·  Educational")
        self.root.geometry("780x720")
        self.root.minsize(680, 640)
        self.root.configure(bg=COLORS["bg"])

        self._setup_fonts()
        self._setup_styles()
        self._build_ui()

    # ---------- styling ----------

    def _setup_fonts(self):
        family = "Segoe UI"
        self.f_title   = tkfont.Font(family=family, size=18, weight="bold")
        self.f_sub     = tkfont.Font(family=family, size=10)
        self.f_section = tkfont.Font(family=family, size=11, weight="bold")
        self.f_body    = tkfont.Font(family=family, size=10)
        self.f_mono    = tkfont.Font(family="Consolas", size=11)
        self.f_metric  = tkfont.Font(family=family, size=17, weight="bold")
        self.f_verdict = tkfont.Font(family=family, size=13, weight="bold")

    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # Base
        style.configure(".",
            background=COLORS["bg"],
            foreground=COLORS["text"],
            font=self.f_body,
        )
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("Card.TFrame",
            background=COLORS["surface"],
            relief="flat",
            borderwidth=1,
        )
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"])
        style.configure("Card.TLabel", background=COLORS["surface"], foreground=COLORS["text"])
        style.configure("Section.TLabel",
            background=COLORS["surface"],
            foreground=COLORS["text"],
            font=self.f_section,
        )
        style.configure("Muted.TLabel",
            background=COLORS["surface"],
            foreground=COLORS["muted"],
            font=self.f_sub,
        )
        style.configure("Title.TLabel",
            background=COLORS["bg"],
            foreground=COLORS["text"],
            font=self.f_title,
        )
        style.configure("Subtitle.TLabel",
            background=COLORS["bg"],
            foreground=COLORS["muted"],
            font=self.f_sub,
        )
        style.configure("Metric.TLabel",
            background=COLORS["surface"],
            foreground=COLORS["primary"],
            font=self.f_metric,
        )
        style.configure("Verdict.TLabel",
            background=COLORS["surface"],
            foreground=COLORS["muted"],
            font=self.f_verdict,
        )
        style.configure("Status.TLabel",
            background=COLORS["surface"],
            foreground=COLORS["muted"],
            font=self.f_sub,
            padding=(10, 6),
        )

        # Entry
        style.configure("Modern.TEntry",
            fieldbackground=COLORS["surface"],
            background=COLORS["surface"],
            foreground=COLORS["text"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["border"],
            darkcolor=COLORS["border"],
            padding=6,
        )
        style.map("Modern.TEntry",
            bordercolor=[("focus", COLORS["primary"])],
            lightcolor=[("focus", COLORS["primary"])],
            darkcolor=[("focus", COLORS["primary"])],
        )

        # Buttons
        style.configure("Primary.TButton",
            background=COLORS["primary"],
            foreground="#FFFFFF",
            font=self.f_section,
            padding=(16, 8),
            borderwidth=0,
            focusthickness=0,
        )
        style.map("Primary.TButton",
            background=[("active", COLORS["primary_hv"]),
                        ("disabled", "#C7D2FE")],
            foreground=[("disabled", "#FFFFFF")],
        )

        style.configure("Secondary.TButton",
            background=COLORS["surface"],
            foreground=COLORS["primary"],
            font=self.f_section,
            padding=(14, 7),
            borderwidth=1,
            bordercolor=COLORS["primary"],
            focusthickness=0,
        )
        style.map("Secondary.TButton",
            background=[("active", "#EEF2FF"),
                        ("disabled", "#F3F4F6")],
            foreground=[("disabled", COLORS["muted"])],
            bordercolor=[("disabled", COLORS["border"])],
        )

    # ---------- UI ----------

    def _card(self, parent, px: int = 18, py: int = 16):
        """Create a card-style container with a 1px border and internal padding.

        Returns (outer, content) where outer should be packed into ``parent``
        and widgets should be added to ``content``.
        """
        outer = tk.Frame(parent, bg=COLORS["border"], highlightthickness=0, bd=0)
        surface = tk.Frame(outer, bg=COLORS["surface"])
        surface.pack(fill="both", expand=True, padx=1, pady=1)
        content = tk.Frame(surface, bg=COLORS["surface"])
        content.pack(fill="both", expand=True, padx=px, pady=py)
        return outer, content

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=(24, 20, 24, 16))
        container.pack(fill="both", expand=True)

        # ---- Header ----
        header = ttk.Frame(container)
        header.pack(fill="x", pady=(0, 18))
        ttk.Label(header, text="PPL Anomaly Detector", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Measure text perplexity with a causal language model. For educational use.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        # ---- Model card ----
        model_outer, model_inner = self._card(container)
        model_outer.pack(fill="x", pady=(0, 14))

        ttk.Label(model_inner, text="Model", style="Section.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(model_inner,
            text="Hugging Face model ID (e.g. gpt2, distilgpt2).",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 10))

        self.model_var = tk.StringVar(value="gpt2")
        entry = ttk.Entry(model_inner, textvariable=self.model_var, style="Modern.TEntry", font=self.f_mono)
        entry.grid(row=2, column=0, sticky="ew", padx=(0, 10))
        model_inner.columnconfigure(0, weight=1)

        self.load_btn = ttk.Button(model_inner, text="Load Model", style="Primary.TButton", command=self._load_model)
        self.load_btn.grid(row=2, column=1, sticky="e")

        # ---- Input card ----
        input_outer, input_inner = self._card(container)
        input_outer.pack(fill="both", expand=True, pady=(0, 14))

        ttk.Label(input_inner, text="Input Text", style="Section.TLabel").pack(anchor="w")
        ttk.Label(input_inner,
            text="Paste or type the text you want to evaluate.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 10))

        text_wrap = tk.Frame(input_inner, bg=COLORS["border"])
        text_wrap.pack(fill="both", expand=True)
        text_inner = tk.Frame(text_wrap, bg=COLORS["surface"])
        text_inner.pack(fill="both", expand=True, padx=1, pady=1)

        self.text_input = tk.Text(text_inner,
            height=9,
            wrap="word",
            relief="flat",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            insertbackground=COLORS["primary"],
            selectbackground="#DBEAFE",
            selectforeground=COLORS["text"],
            font=self.f_mono,
            padx=12,
            pady=10,
            bd=0,
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(text_inner, orient="vertical", command=self.text_input.yview)
        self.text_input.configure(yscrollcommand=scrollbar.set)
        self.text_input.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Threshold + action row
        ctrl = tk.Frame(input_inner, bg=COLORS["surface"])
        ctrl.pack(fill="x", pady=(12, 0))

        ttk.Label(ctrl, text="Threshold", style="Card.TLabel", font=self.f_section).pack(side="left")
        self.threshold_var = tk.StringVar(value="100.0")
        ttk.Entry(ctrl, textvariable=self.threshold_var, style="Modern.TEntry",
                  font=self.f_mono, width=10).pack(side="left", padx=(10, 0))

        self.calc_btn = ttk.Button(
            ctrl, text="Calculate PPL", style="Primary.TButton",
            command=self._on_calculate, state="disabled",
        )
        self.calc_btn.pack(side="right")

        # ---- Result card ----
        result_outer, result_inner = self._card(container)
        result_outer.pack(fill="x", pady=(0, 14))

        ttk.Label(result_inner, text="Result", style="Section.TLabel").pack(anchor="w")

        result_row = tk.Frame(result_inner, bg=COLORS["surface"])
        result_row.pack(fill="x", pady=(10, 4))

        # PPL metric block
        ppl_block = tk.Frame(result_row, bg=COLORS["surface"])
        ppl_block.pack(side="left", fill="x", expand=True, anchor="w")
        ttk.Label(ppl_block, text="PERPLEXITY", style="Muted.TLabel").pack(anchor="w")
        self.ppl_label = ttk.Label(ppl_block, text="—", style="Metric.TLabel", wraplength=300)
        self.ppl_label.pack(anchor="w", pady=(2, 0))

        # Spacer
        tk.Frame(result_row, bg=COLORS["surface"], width=40).pack(side="left")

        # Verdict block
        verdict_block = tk.Frame(result_row, bg=COLORS["surface"])
        verdict_block.pack(side="left")
        ttk.Label(verdict_block, text="VERDICT", style="Muted.TLabel").pack(anchor="w")

        self.verdict_row = tk.Frame(verdict_block, bg=COLORS["surface"])
        self.verdict_row.pack(anchor="w", pady=(4, 0))
        self.verdict_dot = tk.Canvas(self.verdict_row, width=14, height=14,
                                     bg=COLORS["surface"], highlightthickness=0)
        self.verdict_dot.pack(side="left")
        self._set_verdict_dot(COLORS["muted"])
        self.judge_label = ttk.Label(self.verdict_row, text="Awaiting input",
                                     style="Verdict.TLabel")
        self.judge_label.pack(side="left", padx=(8, 0))

        # ---- Status bar ----
        status_outer, status_inner = self._card(container, px=12, py=6)
        status_outer.pack(fill="x")
        self.status_var = tk.StringVar(value="Load a model to get started.")
        ttk.Label(status_inner, textvariable=self.status_var, style="Status.TLabel").pack(fill="x")

    def _set_verdict_dot(self, color: str):
        self.verdict_dot.delete("all")
        self.verdict_dot.create_oval(2, 2, 12, 12, fill=color, outline=color)

    # ---------- callbacks ----------

    def _load_model(self):
        self.load_btn.config(state="disabled")
        self.calc_btn.config(state="disabled")
        self.status_var.set("Loading model…")
        self.root.update_idletasks()

        def _load():
            try:
                detector = PPLDetector(model_id=self.model_var.get())
                self.detector = detector
                self.root.after(0, lambda: self.status_var.set("Model loaded successfully."))
            except Exception as e:
                msg = str(e)
                self.root.after(0, lambda: self.status_var.set(f"Load failed: {msg}"))
            finally:
                self.root.after(0, self._restore_buttons)

        threading.Thread(target=_load, daemon=True).start()

    def _restore_buttons(self):
        self.load_btn.config(state="normal")
        # Re-enable calculate only if a detector is available.
        self.calc_btn.config(state="normal" if self.detector is not None else "disabled")

    def _on_calculate(self):
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            self.status_var.set("Please enter some text.")
            return

        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            self.status_var.set("Threshold must be a number.")
            return

        self.calc_btn.config(state="disabled")
        self.status_var.set("Calculating…")
        self.root.update_idletasks()

        def _calc():
            try:
                is_anomalous, ppl = self.detector.detect_injection(text, threshold=threshold)
                self.root.after(0, lambda: self._apply_result(ppl, is_anomalous))
                self.root.after(0, lambda: self.status_var.set("Calculation complete."))
            except Exception as e:
                msg = str(e)
                self.root.after(0, lambda: self.status_var.set(f"Error: {msg}"))
            finally:
                self.root.after(
                    0,
                    lambda: self.calc_btn.config(
                        state="normal" if self.detector is not None else "disabled"
                    ),
                )

        threading.Thread(target=_calc, daemon=True).start()

    def _apply_result(self, ppl: float, is_anomalous: bool):
        self.ppl_label.config(text=f"{ppl:.2f}")
        if is_anomalous:
            self.judge_label.config(text="Anomalous", foreground=COLORS["danger"])
            self._set_verdict_dot(COLORS["danger"])
        else:
            self.judge_label.config(text="Normal", foreground=COLORS["accent"])
            self._set_verdict_dot(COLORS["accent"])


if __name__ == "__main__":
    root = tk.Tk()
    PPLApp(root)
    root.mainloop()