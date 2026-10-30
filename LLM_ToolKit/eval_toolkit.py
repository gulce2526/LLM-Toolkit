import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ALL_ROLES = ["ŞÜPHELİ", "YETKİLİ", "MUHABİR", "MAĞDUR", "İLGİSİZ"]
SUSPECT = "ŞÜPHELİ"


def to_binary(series):
    """Collapse roles to binary: 1 if ŞÜPHELİ (suspect), else 0."""
    return (series.str.strip().str.upper() == SUSPECT.upper()).astype(int)


class EvalToolkit:
    def __init__(self, root):
        self.root = root
        root.title("LLM Role-Detection Evaluation Toolkit")
        root.geometry("1000x700")

        self.gold_path = None
        self.llm_files = {}   # {name: path}

        # ---- left panel: file loading ----
        left = tk.Frame(root, padx=10, pady=10)
        left.pack(side="left", fill="y")

        tk.Label(left, text="GOLD LABELS", font=("Arial", 11, "bold")).pack(anchor="w")
        tk.Button(left, text="Load Gold CSV", command=self.load_gold).pack(fill="x", pady=4)
        self.gold_label = tk.Label(left, text="(none)", fg="gray")
        self.gold_label.pack(anchor="w")

        tk.Label(left, text="\nLLM OUTPUT CSVs", font=("Arial", 11, "bold")).pack(anchor="w")
        tk.Button(left, text="Add LLM CSV", command=self.add_llm).pack(fill="x", pady=4)
        self.llm_listbox = tk.Listbox(left, height=8, width=28)
        self.llm_listbox.pack(pady=4)
        tk.Button(left, text="Remove selected", command=self.remove_llm).pack(fill="x")

        tk.Button(left, text="COMPARE ALL", command=self.compare,
                  bg="#d070d0", font=("Arial", 11, "bold")).pack(fill="x", pady=20)

        # ---- right panel: tabs for results ----
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.metrics_tab = tk.Frame(self.tabs)
        self.cm_tab = tk.Frame(self.tabs)
        self.tabs.add(self.metrics_tab, text="Metrics")
        self.tabs.add(self.cm_tab, text="Confusion Matrix")

        self.metrics_text = tk.Text(self.metrics_tab, font=("Consolas", 10), wrap="none")
        self.metrics_text.pack(fill="both", expand=True)

        # confusion matrix: model selector + canvas
        cm_top = tk.Frame(self.cm_tab)
        cm_top.pack(fill="x")
        tk.Label(cm_top, text="Model:").pack(side="left")
        self.cm_model = ttk.Combobox(cm_top, state="readonly")
        self.cm_model.pack(side="left", padx=5)
        self.cm_model.bind("<<ComboboxSelected>>", lambda e: self.draw_confusion())
        self.cm_canvas_frame = tk.Frame(self.cm_tab)
        self.cm_canvas_frame.pack(fill="both", expand=True)

        self.results = {}   # {name: merged_dataframe}

    def load_gold(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            self.gold_path = path
            self.gold_label.config(text=path.split("/")[-1], fg="green")

    def add_llm(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            name = path.split("/")[-1].replace(".csv", "")
            self.llm_files[name] = path
            self.llm_listbox.insert("end", name)

    def remove_llm(self):
        sel = self.llm_listbox.curselection()
        if sel:
            name = self.llm_listbox.get(sel[0])
            del self.llm_files[name]
            self.llm_listbox.delete(sel[0])

    def compare(self):
        if not self.gold_path:
            messagebox.showerror("Error", "Load a gold CSV first.")
            return
        if not self.llm_files:
            messagebox.showerror("Error", "Add at least one LLM CSV.")
            return

        gold = pd.read_csv(self.gold_path)
        self.results = {}
        out = []
        out.append(f"{'Model':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}  {'Matched':>8}")
        out.append("-" * 60)

        for name, path in self.llm_files.items():
            pred = pd.read_csv(path)
            merged = gold.merge(pred, on=["article_id", "person"], suffixes=("_true", "_pred"))
            if len(merged) == 0:
                out.append(f"{name:<20}  (no matching rows — check article_id/person)")
                continue

            self.results[name] = merged
            y_true = to_binary(merged["role_true"])
            y_pred = to_binary(merged["role_pred"])

            acc = accuracy_score(y_true, y_pred)
            # zero_division=0 avoids errors if a class is absent
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            out.append(f"{name:<20} {acc:>6.2f} {prec:>6.2f} {rec:>6.2f} {f1:>6.2f}  {len(merged):>8}")

        out.append("\n(Binary scoring: ŞÜPHELİ = positive, everything else = negative)")
        out.append("Precision = of those flagged ŞÜPHELİ, how many were right")
        out.append("Recall    = of real ŞÜPHELİ, how many were caught")

        # best by F1
        if self.results:
            best = max(self.results, key=lambda n: f1_score(
                to_binary(self.results[n]["role_true"]),
                to_binary(self.results[n]["role_pred"]), zero_division=0))
            out.append(f"\nBest model (by F1): {best}")

        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", "\n".join(out))

        # populate confusion-matrix model selector
        self.cm_model["values"] = list(self.results.keys())
        if self.results:
            self.cm_model.current(0)
            self.draw_confusion()

    def draw_confusion(self):
        name = self.cm_model.get()
        if name not in self.results:
            return
        merged = self.results[name]

        # clear old canvas
        for w in self.cm_canvas_frame.winfo_children():
            w.destroy()

        y_true = merged["role_true"].str.strip().str.upper()
        y_pred = merged["role_pred"].str.strip().str.upper()
        labels = [r.upper() for r in ALL_ROLES]
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="PuRd")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(ALL_ROLES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(ALL_ROLES, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {name}")
        # write counts in cells
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.cm_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    EvalToolkit(root)
    root.mainloop()