# KolektorSDD 缺陷检测：从下载到训练到推理（端到端示例）

这个项目给你一个最小可复现的工业缺陷检测流程（面向 KolektorSDD）：
- 自动下载（或手动放置）数据集
- 扫描并建立索引（image / mask / label）
- 统一 resize（默认 512x1408）
- 训练：二分类（缺陷/正常）+ 可选分割（U-Net）
- 推理：输出缺陷概率、分割mask、轮廓、可视化叠加图

> 注意：KolektorSDD 许可为 CC BY-NC-SA；商用请自行联系数据集作者。
> 数据集下载链接可能会变化；如果脚本下载失败，可按下方“手动下载”方式处理。

---

## 0. 环境准备

推荐 Python 3.10+（3.11 也可）

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 1. 下载数据集

### 方式 A：脚本自动下载（推荐）

```bash
python scripts/download_ksdd.py --out data/raw
```

脚本会尝试多个官方/镜像 URL（可在脚本里自己追加）。

### 方式 B：手动下载（兜底）

请从 ViCoS 官方页面下载 “fine annotations” 版本，得到一个 zip 文件，然后放到：
- `data/raw/` 目录下（文件名随意）

然后运行解压：

```bash
python scripts/unpack_ksdd.py --raw data/raw --out data/ksdd
```

---

## 2. 建立索引 + 预处理（resize、划分 train/val/test）

```bash
python scripts/prepare_ksdd.py --data_root data/ksdd --out data/processed --img_h 1408 --img_w 512
```

输出：
- `data/processed/index.csv`：每一行一张图（image_path, mask_path, label, split）
- `data/processed/images/`：resize 后的图
- `data/processed/masks/`：resize 后的mask（如果能找到）

---

## 3. 训练（二分类：缺陷/正常）

```bash
python scripts/train_cls.py --data data/processed --epochs 10 --batch 8 --lr 1e-4 --out_dir outputs/cls
```

训练完成后会保存：
- `outputs/cls/best.pt`
- 训练日志：loss/acc 等

---

## 4. （可选）训练分割（U-Net，输出缺陷mask）

如果数据集里能成功匹配到 mask（fine annotations），可以训练分割：

```bash
python scripts/train_seg.py --data data/processed --epochs 20 --batch 4 --lr 1e-3 --out_dir outputs/seg
```

输出：
- `outputs/seg/best.pt`

---

## 5. 推理（单张/文件夹）+ 轮廓提取可视化

单张图：
```bash
python scripts/infer.py --image path/to/one.png --cls_ckpt outputs/cls/best.pt --seg_ckpt outputs/seg/best.pt --out_dir outputs/infer
```

文件夹：
```bash
python scripts/infer.py --image_dir path/to/folder --cls_ckpt outputs/cls/best.pt --seg_ckpt outputs/seg/best.pt --out_dir outputs/infer
```

输出示例：
- `*_overlay.png`：原图+mask+轮廓叠加
- `*_mask.png`：预测mask（二值）
- `*_contours.png`：轮廓图
- `results.csv`：每张图的缺陷概率/是否异常

---

## 6. 经典CV基线（不训练也能跑）

```bash
python scripts/cv_baseline.py --image path/to/one.png --out_dir outputs/cv
```

会输出一套：均衡化/梯度/阈值/形态学/轮廓 结果，用于写实验报告“关键方法”。

---

## 常见问题

1) 如果 `prepare_ksdd.py` 找不到 mask：
- 你可能下载了 “box annotations” 版本或结构不同
- 先看看 `data/ksdd` 内部目录结构，然后按脚本提示修正匹配规则（scripts/prepare_ksdd.py 里有注释）

2) GPU：
- 默认 CPU 也能跑，只是慢一点。
- 有 CUDA 的话 PyTorch 会自动使用。

---

