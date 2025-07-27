"""
ProtRepr: 蛋白质表示学习框架

基于 ProteinTensor 的蛋白质深度学习工具包，专注于蛋白质结构表示、
预测和功能分析。使用 PyTorch 作为后端以获得最佳性能。
"""

__version__ = "0.3.0"
__author__ = "ProtRepr Team"

# 核心数据结构
from .core.atom14 import Atom14
# from .core.atom37 import Atom37  # TODO: 实现
# from .core.frame import Frame    # TODO: 实现

# 批量处理功能
from .batch_processing import (
    BatchPDBToAtom14Converter,
    convert_single_worker,
    save_statistics
)

__all__ = [
    # 核心类
    "Atom14",
    # "Atom37",  # TODO: 实现后启用
    # "Frame",   # TODO: 实现后启用
    
    # 批量处理
    "BatchPDBToAtom14Converter",
    "convert_single_worker", 
    "save_statistics",
    
    # 元数据
    "__version__",
    "__author__",
] 