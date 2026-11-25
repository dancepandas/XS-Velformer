"""
data 包 —— 数据预处理与数据集构建

提供从 sections.npz 加载数据、特征归一化、自监督样本拆分和批处理功能。

主要组件:
- SectionPreprocessor: 加载 npz, 特征归一化, 全局统计量, 时间划分 (§12)
- SectionDataset: 自监督训练数据集, 5种采样策略 (§6.2, §12.8)
- collate_fn: 变长序列批处理与 Padding (§12.9)
"""

from .preprocess import SectionPreprocessor, SectionSample
from .dataset import SectionDataset, collate_fn
