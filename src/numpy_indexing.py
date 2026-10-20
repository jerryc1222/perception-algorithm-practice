"""NumPy 索引、切片、布尔筛选、广播与向量化练习函数。"""

import numpy as np


def _as_float_array(arr, name="arr"):
    """将输入转为浮点数组，并拒绝不能转换为数值的输入。"""
    try:
        return np.asarray(arr, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是可转换为浮点数的数组。") from error


def _validate_finite(arr, name="arr"):
    """确认数组中不存在 NaN 或无穷大。"""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 不能包含 NaN 或无穷大。")


def _validate_finite_number(value, name):
    """确认单个参数是有限数值，并以 float 形式返回。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限数值。")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是有限数值。") from error
    if not np.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值。")
    return number


def normalize_array(arr):
    """使用 Min-Max 归一化将数值映射到 [0, 1]。

    空数组返回空浮点数组；含 NaN 或无穷大的数组抛出 ValueError；
    恒定数组没有可缩放范围，因此返回同形状的全零数组。
    """
    data = _as_float_array(arr)
    if data.size == 0:
        return data.copy()
    _validate_finite(data)

    minimum = data.min()
    maximum = data.max()
    if maximum == minimum:
        return np.zeros_like(data, dtype=float)
    return (data - minimum) / (maximum - minimum)


def standardize_array(arr):
    """使用总体标准差将数组标准化为均值 0、标准差 1。

    总体标准差使用 ``np.std(..., ddof=0)``，即把当前数组看作完整总体。
    空数组返回空浮点数组；非有限值抛出 ValueError；恒定数组返回全零数组。
    """
    data = _as_float_array(arr)
    if data.size == 0:
        return data.copy()
    _validate_finite(data)

    mean = data.mean()
    population_std = data.std(ddof=0)
    if population_std == 0:
        return np.zeros_like(data, dtype=float)
    return (data - mean) / population_std


def remove_outliers(arr, threshold=2.0):
    """使用 Z-score 保留绝对值不超过阈值的元素。

    返回一维浮点数组。空数组返回空数组；恒定数组的 Z-score 全视为 0，
    因而保留全部元素。阈值必须是大于等于 0 的有限数值。
    """
    limit = _validate_finite_number(threshold, "threshold")
    if limit < 0:
        raise ValueError("threshold 必须大于或等于 0。")

    data = _as_float_array(arr)
    if data.size == 0:
        return data.reshape(-1).copy()
    _validate_finite(data)

    population_std = data.std(ddof=0)
    flat_data = data.reshape(-1)
    if population_std == 0:
        return flat_data.copy()

    z_scores = (flat_data - data.mean()) / population_std
    return flat_data[np.abs(z_scores) <= limit]


def pairwise_distances(X):
    """返回二维样本数组 X 的两两欧氏距离矩阵。

    X 的形状必须是 ``(n_samples, n_features)``。实现通过广播构建形状为
    ``(n_samples, n_samples, n_features)`` 的特征差，再沿最后一维求范数。
    空样本返回形状为 ``(0, 0)`` 的矩阵；非有限值和非二维输入抛出 ValueError。
    """
    data = _as_float_array(X, "X")
    if data.ndim != 2:
        raise ValueError("X 必须是形状为 (n_samples, n_features) 的二维数组。")
    _validate_finite(data, "X")

    sample_count = data.shape[0]
    if sample_count == 0:
        return np.empty((0, 0), dtype=float)

    differences = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    return np.sqrt(np.sum(differences**2, axis=2))


def select_by_threshold(arr, threshold):
    """使用布尔索引返回 arr 中严格大于 threshold 的元素。

    输入被展平后筛选，因此返回一维数组。threshold 必须是有限数值。
    """
    limit = _validate_finite_number(threshold, "threshold")
    data = _as_float_array(arr)
    return data.reshape(-1)[data.reshape(-1) > limit]


def clip_array(arr, min_value, max_value):
    """返回限制在 [min_value, max_value] 内的新数组，不修改原数组。

    上下界必须是有限数值，且 min_value 不能大于 max_value。
    """
    lower = _validate_finite_number(min_value, "min_value")
    upper = _validate_finite_number(max_value, "max_value")
    if lower > upper:
        raise ValueError("min_value 不能大于 max_value。")
    data = _as_float_array(arr)
    return np.clip(data, lower, upper)


def get_center_crop(image, crop_height, crop_width):
    """复制并返回二维或三维图像数组的中心裁剪区域。

    对于奇数尺寸差，额外的一行或列保留在底部或右侧。函数返回副本，
    因此修改裁剪结果不会修改原图。
    """
    data = np.asarray(image)
    if data.ndim not in (2, 3):
        raise ValueError("image 必须是二维灰度图或三维彩色图数组。")
    if isinstance(crop_height, bool) or isinstance(crop_width, bool):
        raise ValueError("裁剪尺寸必须是正整数。")
    if not isinstance(crop_height, (int, np.integer)) or not isinstance(
        crop_width, (int, np.integer)
    ):
        raise ValueError("裁剪尺寸必须是正整数。")
    if crop_height <= 0 or crop_width <= 0:
        raise ValueError("裁剪尺寸必须是正整数。")

    height, width = data.shape[:2]
    if crop_height > height or crop_width > width:
        raise ValueError("裁剪尺寸不能超过图像的高和宽。")

    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return data[top : top + crop_height, left : left + crop_width].copy()


def broadcast_add(arr, bias):
    """返回 arr 与 bias 的广播加法结果，不修改 arr。

    例如图像的形状为 ``(H, W, C)``、通道偏置的形状为 ``(C,)`` 时，
    同一个 C 维偏置会加到每一个像素。形状不兼容时抛出清晰的 ValueError。
    """
    data = np.asarray(arr)
    offset = np.asarray(bias)
    try:
        return data + offset
    except ValueError as error:
        raise ValueError(
            "无法广播相加："
            f"arr 的形状为 {data.shape}，bias 的形状为 {offset.shape}。"
        ) from error
