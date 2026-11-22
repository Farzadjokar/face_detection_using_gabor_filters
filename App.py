import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageOps, ImageTk

import viz
from gabor_face import PCAPNN, GaborFaceDetector, face_filters

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.npz")
FILETYPES = [("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gabor Face Detector")
        self.geometry("1150x740")
        self.minsize(860, 540)

        self.detector = None
        self.pil = None            # original image (RGB)
        self.gray = None           # float32 [0,1]
        self.cands = None          # cached raw windows (slow scan result)
        self.cands_minface = None
        self.boxes = []
        self.shown = None          # PIL image currently on the canvas
        self.photo = None          # keeps a reference to the PhotoImage
        self.queue = queue.Queue()
        self.busy = False

        self.var_thr = tk.DoubleVar(value=0.80)
        self.var_nb = tk.IntVar(value=1)
        self.var_ov = tk.DoubleVar(value=0.30)
        self.var_min = tk.IntVar(value=40)
        self.var_raw = tk.BooleanVar(value=False)
        self.var_status = tk.StringVar(value="Open an image to start.")

        self._build_ui()
        self._load_model()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        side = ttk.Frame(self, padding=12)
        side.grid(row=0, column=0, sticky="ns")

        ttk.Label(side, text="Gabor Face Detector", font=("TkDefaultFont", 14, "bold")).pack(anchor="w")
        ttk.Label(side, text="Gabor features + PCA + PNN", foreground="#666").pack(anchor="w", pady=(0, 10))

        ttk.Button(side, text="Open image...", command=self.open_image).pack(fill="x", pady=2)
        self.btn_detect = ttk.Button(side, text="Detect faces", command=self.detect)
        self.btn_detect.pack(fill="x", pady=2)
        ttk.Button(side, text="Save result...", command=self.save_result).pack(fill="x", pady=2)

        box = ttk.LabelFrame(side, text="Settings", padding=8)
        box.pack(fill="x", pady=10)
        self._slider(box, "Confidence threshold", self.var_thr, 0.50, 0.99, 0.01, self._refilter)
        self._slider(box, "Min. overlapping hits", self.var_nb, 1, 10, 1, self._refilter)
        self._slider(box, "Overlap (NMS IoU)", self.var_ov, 0.10, 0.60, 0.05, self._refilter)
        self._slider(box, "Min. face size (px, needs re-scan)", self.var_min, 20, 200, 5, None)
        ttk.Checkbutton(box, text="Show raw candidate windows", variable=self.var_raw,
                        command=self._refilter).pack(anchor="w", pady=(6, 0))

        viewbox = ttk.LabelFrame(side, text="Look inside", padding=8)
        viewbox.pack(fill="x")
        ttk.Button(viewbox, text="Gabor filter bank", command=self.show_bank).pack(fill="x", pady=2)
        ttk.Button(viewbox, text="Gabor responses of image", command=self.show_responses).pack(fill="x", pady=2)

        self.progress = ttk.Progressbar(side, mode="determinate", maximum=1.0)
        self.progress.pack(fill="x", pady=(14, 2))
        ttk.Label(side, textvariable=self.var_status, wraplength=230, justify="left").pack(anchor="w")

        self.canvas = tk.Canvas(self, background="#222", highlightthickness=0)
        self.canvas.grid(row=0, column=1, sticky="nsew")
        self.canvas.bind("<Configure>", lambda e: self._show(self.shown))

    def _slider(self, parent, text, var, lo, hi, step, cmd):
        ttk.Label(parent, text=text).pack(anchor="w")
        s = tk.Scale(parent, from_=lo, to=hi, resolution=step, orient="horizontal", variable=var,
                     length=230, showvalue=True, highlightthickness=0)
        if cmd:
            s.bind("<ButtonRelease-1>", lambda e: cmd())
            s.bind("<KeyRelease>", lambda e: cmd())
        s.pack(fill="x", pady=(0, 4))

    # ------------------------------------------------------------------ model
    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Model missing", "model.npz not found.\nRun   python train.py   first.")
            self.var_status.set("model.npz missing - run train.py")
            return
        try:
            self.detector = GaborFaceDetector(PCAPNN.load(MODEL_PATH), face_filters())
        except Exception as e:                      # corrupted / incompatible model file
            messagebox.showerror("Model error", f"Could not load model.npz:\n{e}")

    # ------------------------------------------------------------------ actions
    def open_image(self):
        path = filedialog.askopenfilename(title="Choose an image", filetypes=FILETYPES)
        if not path:
            return
        try:
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception as e:
            messagebox.showerror("Cannot open image", str(e))
            return
        self.pil = img
        self.gray = np.asarray(img.convert("L"), np.float32) / 255.0
        self.cands, self.boxes = None, []
        self._show(img)
        self.var_status.set(f"{os.path.basename(path)}  ({img.width}x{img.height}). Press 'Detect faces'.")

    def detect(self):
        if self.detector is None:
            return messagebox.showwarning("No model", "model.npz could not be loaded.")
        if self.pil is None:
            return messagebox.showinfo("No image", "Open an image first.")
        if self.busy:
            return
        self.busy = True
        self.btn_detect.state(["disabled"])
        self.progress["value"] = 0
        self.var_status.set("Scanning image pyramid ...")
        gray, min_face = self.gray, int(self.var_min.get())

        def work():
            t = time.time()
            try:
                c = self.detector.candidates(gray, 0.5, min_face=min_face,
                                             progress=lambda f: self.queue.put(("progress", f)))
                self.queue.put(("done", c, min_face, time.time() - t))
            except Exception as e:
                self.queue.put(("error", str(e)))

        threading.Thread(target=work, daemon=True).start()
        self.after(60, self._poll)

    def _poll(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg[0] == "progress":
                    self.progress["value"] = msg[1]
                elif msg[0] == "done":
                    self.cands, self.cands_minface, self.elapsed = msg[1], msg[2], msg[3]
                    self._finish()
                    return
                else:
                    self._finish()
                    messagebox.showerror("Detection failed", msg[1])
                    return
        except queue.Empty:
            pass
        self.after(60, self._poll)

    def _finish(self):
        self.busy = False
        self.btn_detect.state(["!disabled"])
        self.progress["value"] = 1.0
        if self.cands is not None:
            self._refilter()

    def _refilter(self):
        """Fast: re-apply threshold / NMS / min-hits to the cached candidates."""
        if self.cands is None or self.pil is None:
            return
        thr, ov, nb = float(self.var_thr.get()), float(self.var_ov.get()), int(self.var_nb.get())
        self.boxes = GaborFaceDetector.filter(self.cands, thr, ov, nb)
        raw = [c for c in self.cands if c[4] >= thr] if self.var_raw.get() else None
        self._show(viz.draw_detections(self.pil, self.boxes, raw))
        n = len(self.boxes)
        self.var_status.set(f"{n} face{'s' if n != 1 else ''} found  |  scan took {self.elapsed:.1f}s  "
                            f"|  {len(self.cands)} raw windows")

    def save_result(self):
        if self.shown is None:
            return messagebox.showinfo("Nothing to save", "Open an image and run detection first.")
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            self.shown.save(path)
            self.var_status.set(f"Saved {os.path.basename(path)}")

    def show_bank(self):
        self._popup("Gabor filter bank (4 scales x 6 orientations, Manjunath & Ma)", viz.bank_image())

    def show_responses(self):
        if self.gray is None:
            return messagebox.showinfo("No image", "Open an image first.")
        self._popup("Image | Gabor magnitude at 0, 45, 90, 135 degrees",
                    viz.response_image(self.gray, face_filters()))

    def _popup(self, title, pil):
        win = tk.Toplevel(self)
        win.title(title)
        win.photo = ImageTk.PhotoImage(pil)          # keep a reference on the window
        ttk.Label(win, image=win.photo).pack(padx=8, pady=8)

    # ------------------------------------------------------------------ drawing
    def _show(self, pil):
        self.shown = pil
        self.canvas.delete("all")
        if pil is None:
            return
        cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        s = min(cw / pil.width, ch / pil.height, 4.0)
        resized = pil.resize((max(1, int(pil.width * s)), max(1, int(pil.height * s))), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(cw // 2, ch // 2, image=self.photo)


if __name__ == "__main__":
    App().mainloop()