"""
baseline.interpolation —— 传统插值基线

四种基于空间插值的流速预测方法:
    - nearest_x: 按 x_norm 最近邻
    - linear_x:  基于 x_norm 线性插值
    - idw_x:     按横向距离反距离加权
    - idw_xd:    在 [x_norm, d_norm] 二维空间反距离加权
"""

from baseline.interpolation.methods import nearest_x, linear_x, idw_x, idw_xd
from baseline.interpolation.adapter import (
    InterpolationAdapter,
    NearestXAdapter,
    LinearXAdapter,
    IDWXAdapter,
    IDWXDAdapter,
)
