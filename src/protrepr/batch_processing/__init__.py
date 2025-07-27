"""
批量处理功能模块

本模块提供高性能的批量数据处理功能，支持：
- PDB/CIF 文件的批量转换为各种蛋白质表示格式
- 多进程并行处理，充分利用多核CPU资源
- 统一的进度跟踪和错误处理机制
- 支持 Atom14、Atom37 和 Frame 表示格式

主要组件：
- BatchPDBToAtom14Converter: 批量 PDB 到 Atom14 转换器
- BatchPDBToAtom37Converter: 批量 PDB 到 Atom37 转换器
- save_statistics: 统计信息保存工具
"""

from .atom14_batch_converter import BatchPDBToAtom14Converter, save_statistics
from .atom37_batch_converter import BatchPDBToAtom37Converter

__all__ = [
    "BatchPDBToAtom14Converter",
    "BatchPDBToAtom37Converter", 
    "save_statistics",
] 