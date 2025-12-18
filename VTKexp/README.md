# VTK 实验代码

依赖：Python 3.10+，`pip install vtk`（测试于 VTK 9.3.x）。

## 运行方式
在本目录执行：

- 实验一：模型建模与交互
  - `python exp1_basic_model.py --shape cone` 或 `python exp1_basic_model.py --stl model.stl`
  - TrackballActor 交互：左旋转 / 中平移 / 右缩放。

- 实验二：四元数旋转
  - `python exp2_quaternion_sliders.py [--stl model.stl]`
  - 窗口左侧四个滑块设定 w,x,y,z，实时归一化并更新姿态。

- 实验三：Bezier 曲线
  - `python exp3_bezier.py`
  - 展示单段三次曲线及在 P3 处 C1 光滑拼接的第二段；可在脚本内修改控制点。

- 实验四：B 样条曲线
  - `python exp4_bspline_interactive.py`
  - 左键点击添加特征点；蓝色为均匀 B 样条，橙色为准均匀（弦长参数）。按 `u` 撤销，`r` 重置。

- 实验五：光照效果
  - `python exp5_lighting.py [--stl model.stl]`
  - 三种光源：1 环境光、2 漫反射、3 镜面高光，可按数字键开关。

- 实验六：图像处理
  - `python exp6_image_processing.py`
  - Tk GUI，支持去噪滤波、直方图均衡化、阈值二值分割和轮廓提取（依赖 opencv-python、numpy、Pillow）。

## 说明
- 所有窗口均可缩放；若使用 STL，模型单位需自洽。
- 若需截图，可在 VTK 窗口菜单选择保存，或在代码中添加 `vtkWindowToImageFilter`。
- 如需整合到单一 GUI 或添加更多控制，请告知需求。
