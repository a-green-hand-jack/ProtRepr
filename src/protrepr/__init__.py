"""
ProtRepr: 蛋白质表示学习框架

基于 ProteinTensor 的蛋白质深度学习工具包，专注于蛋白质结构表示、
预测和功能分析。使用 PyTorch 作为后端以获得最佳性能。
"""

# 🚀 首先设置警告过滤器，必须在任何其他导入之前！
import warnings
warnings.filterwarnings('ignore', message='.*Used element.*')
warnings.filterwarnings('ignore', category=UserWarning, module='Bio.PDB.*')

__version__ = "0.3.0"
__author__ = "ProtRepr Team"

# 核心数据结构
from .core.atom14 import Atom14
from .core.atom37 import Atom37  
from .core.frame import Frame    

# 批量处理功能
from .batch_processing import (
    BatchPDBToAtom14Converter,
    BatchPDBToAtom37Converter,
    save_statistics
)

__all__ = [
    # 核心类
    "Atom14",
    "Atom37",  
    "Frame",   
    
    # 批量处理
    "BatchPDBToAtom14Converter",
    "BatchPDBToAtom37Converter", 
    "save_statistics",
    
    # 元数据
    "__version__",
    "__author__",
] 