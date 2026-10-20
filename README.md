# Perception Algorithm Practice

面向计算机视觉、图像信号处理与智能感知算法岗位的 Python 编程实践项目。

## 项目目标

- 夯实 Python、NumPy 与科学计算基础。
- 逐步学习 SciPy、OpenCV、PyTorch 和图像处理算法。
- 通过可运行代码、测试、实验记录和 Git 提交形成可展示的项目成果。
- 后续扩展到图像鲁棒性、点云处理和多传感器融合。

## 当前学习进度

- 2026-10-19：创建项目、配置 Python 虚拟环境，完成 NumPy 数组、`shape`、`ndim`、`dtype` 和 `axis` 基础练习。
- 2026-10-20：实现数组索引、布尔筛选、广播、中心裁剪、归一化、标准化和批量距离计算，并使用 `unittest` 测试。

## 今日模块：NumPy 索引、切片与广播

`src/numpy_indexing.py` 包含以下函数：

- `normalize_array`：Min-Max 归一化；常量数组返回全零数组。
- `standardize_array`：使用总体标准差（`ddof=0`）进行标准化；常量数组返回全零数组。
- `remove_outliers`：使用 Z-score 移除异常值。
- `pairwise_distances`：通过广播计算样本两两之间的欧氏距离矩阵。
- `select_by_threshold`：使用布尔索引筛选大于阈值的元素。
- `clip_array`：将数值限制到指定区间。
- `get_center_crop`：从二维灰度图或三维彩色图复制中心裁剪区域。
- `broadcast_add`：演示数组与偏置的广播加法，例如 `(H, W, C) + (C,)`。

## 运行方式

1. 打开 PowerShell 并进入项目目录。
2. 激活虚拟环境：

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. 运行 NumPy 基础示例：

   ```powershell
   python .\src\numpy_basics.py
   ```

4. 运行今天的全部测试：

   ```powershell
   python -m unittest -v
   ```

## 学习收获

- 基础切片通常返回与原数组共享内存的视图；布尔索引通常返回独立副本。
- 广播按维度从右向左匹配：维度相等或其中之一为 `1` 时可以运算。
- 向量化能将双重循环的成对距离计算表达为一次批量数组运算。
- 图像裁剪、阈值分割、通道偏置校正和特征距离计算都建立在这些数组操作之上。

## 项目结构

```text
src/    # 可运行的学习代码
tests/  # unittest 自动化测试
notes/  # 每日学习笔记与复盘
```
