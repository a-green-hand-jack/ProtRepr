"""
批量处理功能模块

本模块提供高性能的批量数据处理功能，支持：
- PDB/CIF 文件的批量转换为各种蛋白质表示格式
- 蛋白质表示格式的批量转换为结构文件
- 多进程并行处理，充分利用多核CPU资源
- 统一的进度跟踪和错误处理机制
- 支持 Atom14、Atom37 和 Frame 表示格式

主要组件：
- BatchPDBToAtom14Converter: 批量 PDB/CIF 到 Atom14 转换器
- BatchPDBToAtom37Converter: 批量 PDB/CIF 到 Atom37 转换器
- BatchPDBToFrameConverter: 批量 PDB/CIF 到 Frame 转换器
- BatchAtom14ToCIFConverter: 批量 Atom14 到 CIF/PDB 转换器
- BatchAtom37ToCIFConverter: 批量 Atom37 到 CIF/PDB 转换器
- BatchFrameToCIFConverter: 批量 Frame 到 CIF/PDB 转换器
- save_statistics: 统计信息保存工具
"""

from .cif_to_atom14_converter import BatchPDBToAtom14Converter, save_statistics
from .cif_to_atom37_converter import BatchPDBToAtom37Converter
from .frame_batch_converter import BatchPDBToFrameConverter
from .atom14_to_cif_converter import BatchAtom14ToCIFConverter
from .atom37_to_cif_converter import BatchAtom37ToCIFConverter
from .frame_to_cif_converter import BatchFrameToCIFConverter

__all__ = [
    "BatchPDBToAtom14Converter",
    "BatchPDBToAtom37Converter",
    "BatchPDBToFrameConverter",
    "BatchAtom14ToCIFConverter", 
    "BatchAtom37ToCIFConverter",
    "BatchFrameToCIFConverter",
    "save_statistics",
] 