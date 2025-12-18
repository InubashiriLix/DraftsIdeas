"""
实验六：图像处理（单文件 GUI）
- 加载任意图片
- 去噪：高斯/中值/双边滤波
- 模糊图像直方图均衡化
- 二值化阈值分割并提取轮廓
运行：python exp6_image_processing.py
依赖：opencv-python、numpy、Pillow（仅用来在 Tk 中显示预览）
"""
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


def ensure_odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


class ImageApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("实验六：图像处理")
        self.root.geometry("1050x720")

        self.image_path: str | None = None
        self.image: np.ndarray | None = None
        self.processed: np.ndarray | None = None

        # UI state
        self.denoise_method = tk.StringVar(value="Median")
        self.kernel_var = tk.IntVar(value=5)
        self.sigma_var = tk.IntVar(value=50)
        self.threshold_var = tk.IntVar(value=128)
        self.use_otsu = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value="请先加载一张图片。")

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.root); top.pack(fill="x", padx=10, pady=8)
        ttk.Button(top, text="打开图片", command=self.load_image).pack(side="left")
        ttk.Button(top, text="保存处理结果", command=self.save_image).pack(side="left", padx=8)
        self.path_label = ttk.Label(top, text="未选择文件")
        self.path_label.pack(side="left", padx=10)

        controls = ttk.Frame(self.root); controls.pack(fill="x", padx=10)
        self._build_denoise_frame(controls)
        self._build_equalize_frame(controls)
        self._build_threshold_frame(controls)

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", pady=6)

        previews = ttk.Frame(self.root); previews.pack(fill="both", expand=True, padx=10, pady=4)
        self.orig_label = ttk.Label(previews, text="原图预览", relief="groove", anchor="center")
        self.proc_label = ttk.Label(previews, text="处理结果预览", relief="groove", anchor="center")
        self.orig_label.pack(side="left", expand=True, fill="both", padx=(0, 6))
        self.proc_label.pack(side="left", expand=True, fill="both")

        ttk.Label(self.root, textvariable=self.status_var, anchor="w").pack(fill="x", padx=10, pady=(4, 8))

    def _build_denoise_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="去噪滤波"); frame.pack(side="left", padx=6, pady=4, fill="x")
        ttk.Label(frame, text="方法：").grid(row=0, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.denoise_method, values=["Gaussian", "Median", "Bilateral"], width=10, state="readonly").grid(row=0, column=1, padx=4, pady=2)
        ttk.Label(frame, text="核大小 (奇数)：").grid(row=1, column=0, sticky="w")
        ttk.Scale(frame, from_=3, to=15, orient="horizontal", variable=self.kernel_var).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Label(frame, text="强度/σ：").grid(row=2, column=0, sticky="w")
        ttk.Scale(frame, from_=10, to=150, orient="horizontal", variable=self.sigma_var).grid(row=2, column=1, sticky="ew", padx=4)
        ttk.Button(frame, text="应用去噪", command=self.apply_denoise).grid(row=3, column=0, columnspan=2, pady=4, sticky="ew")
        frame.columnconfigure(1, weight=1)

    def _build_equalize_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="直方图均衡化"); frame.pack(side="left", padx=6, pady=4, fill="x")
        ttk.Label(frame, text="用于模糊或亮度不均的图像").grid(row=0, column=0, padx=4, pady=2)
        ttk.Button(frame, text="均衡化亮度", command=self.apply_equalize).grid(row=1, column=0, padx=4, pady=4, sticky="ew")

    def _build_threshold_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="阈值分割 + 轮廓"); frame.pack(side="left", padx=6, pady=4, fill="both", expand=True)
        ttk.Label(frame, text="阈值：").grid(row=0, column=0, sticky="w")
        ttk.Scale(frame, from_=0, to=255, orient="horizontal", variable=self.threshold_var).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Checkbutton(frame, text="Otsu 自适应阈值", variable=self.use_otsu).grid(row=1, column=0, columnspan=2, sticky="w", padx=2, pady=2)
        ttk.Button(frame, text="二值化并提取轮廓", command=self.apply_threshold).grid(row=2, column=0, columnspan=2, pady=4, sticky="ew")
        frame.columnconfigure(1, weight=1)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("错误", "无法读取该文件，请选择图片格式。")
            return
        self.image_path = path
        self.image = img
        self.processed = None
        self.path_label.config(text=os.path.basename(path))
        self.status_var.set(f"已加载：{path}  尺寸：{img.shape[1]}x{img.shape[0]}")
        self._refresh_previews()

    def save_image(self):
        if self.processed is None:
            messagebox.showinfo("提示", "请先生成处理结果。")
            return
        default = "processed.png"
        if self.image_path:
            base = os.path.splitext(os.path.basename(self.image_path))[0]
            default = f"{base}_processed.png"
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=default, filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp")])
        if not path:
            return
        cv2.imwrite(path, self.processed)
        self.status_var.set(f"已保存结果到：{path}")

    def apply_denoise(self):
        if self.image is None:
            messagebox.showinfo("提示", "请先打开一张图片。")
            return
        k = ensure_odd(self.kernel_var.get())
        method = self.denoise_method.get()
        if method == "Gaussian":
            out = cv2.GaussianBlur(self.image, (k, k), 0)
        elif method == "Median":
            out = cv2.medianBlur(self.image, k)
        else:  # Bilateral
            sigma = max(1, int(self.sigma_var.get()))
            out = cv2.bilateralFilter(self.image, k, sigma, sigma // 2)
        self.processed = out
        self.status_var.set(f"去噪完成：{method}，核大小 {k}")
        self._refresh_previews()

    def apply_equalize(self):
        if self.image is None:
            messagebox.showinfo("提示", "请先打开一张图片。")
            return
        img = self.image
        if img.ndim == 2 or img.shape[2] == 1:
            eq = cv2.equalizeHist(img)
        else:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        self.processed = eq
        self.status_var.set("直方图均衡化完成。")
        self._refresh_previews()

    def apply_threshold(self):
        if self.image is None:
            messagebox.showinfo("提示", "请先打开一张图片。")
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thresh_val = int(self.threshold_var.get())
        flag = cv2.THRESH_BINARY
        if self.use_otsu.get():
            flag |= cv2.THRESH_OTSU
            thresh_val = 0
        _, mask = cv2.threshold(gray, thresh_val, 255, flag)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        drawn = self.image.copy()
        cv2.drawContours(drawn, contours, -1, (0, 255, 0), 2)
        combined = np.hstack([overlay, drawn])

        self.processed = combined
        self.status_var.set(f"阈值分割完成，检测到轮廓 {len(contours)} 个。")
        self._refresh_previews()

    def _refresh_previews(self):
        self._show_image_on_label(self.image, self.orig_label, placeholder="原图预览")
        self._show_image_on_label(self.processed, self.proc_label, placeholder="处理结果预览")

    def _show_image_on_label(self, img: np.ndarray | None, label: ttk.Label, placeholder: str):
        if img is None:
            label.config(image="", text=placeholder)
            label.image = None
            return
        preview = self._to_preview_image(img, max_size=460)
        tk_img = ImageTk.PhotoImage(preview)
        label.config(image=tk_img, text="")
        label.image = tk_img  # keep reference

    def _to_preview_image(self, img: np.ndarray, max_size: int = 480) -> Image.Image:
        if img.ndim == 2 or img.shape[2] == 1:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((max_size, max_size))
        return pil_img


def main():
    root = tk.Tk()
    ImageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
