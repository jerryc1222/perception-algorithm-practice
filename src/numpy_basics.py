# NumPy 基础练习：数组创建、属性、axis 与 Python 列表对比

import numpy as np


# 1. 创建一维 NumPy 数组
one_dimensional_array = np.array([10, 20, 30, 40])

print("=== 一维数组 ===")
print("数组内容：", one_dimensional_array)
print("数组类型：", type(one_dimensional_array))
print("shape：", one_dimensional_array.shape)
print("ndim：", one_dimensional_array.ndim)
print("dtype：", one_dimensional_array.dtype)
print()

# 2. 创建二维 NumPy 数组：两行、三列
two_dimensional_array = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
    ]
)

print("=== 二维数组 ===")
print("数组内容：")
print(two_dimensional_array)
print("shape：", two_dimensional_array.shape)
print("ndim：", two_dimensional_array.ndim)
print("dtype：", two_dimensional_array.dtype)
print()

# 3. axis 的初步理解
# axis=0：沿“行”的方向压缩，因此得到每一列的结果
print("=== axis 示例 ===")
print("按列求和，axis=0：", two_dimensional_array.sum(axis=0))

# axis=1：沿“列”的方向压缩，因此得到每一行的结果
print("按行求和，axis=1：", two_dimensional_array.sum(axis=1))
print()

# 4. 用 NumPy 创建指定形状、指定数据类型的数组
zero_array = np.zeros((2, 3), dtype=np.float32)

print("=== zeros 创建的数组 ===")
print(zero_array)
print("shape：", zero_array.shape)
print("dtype：", zero_array.dtype)
print()

# 5. Python 列表与 NumPy 数组的区别
python_list = [10, 20, 30, 40]

print("=== Python 列表与 NumPy 数组 ===")
print("Python 列表乘以 2：", python_list * 2)
print("NumPy 数组乘以 2：", one_dimensional_array * 2)