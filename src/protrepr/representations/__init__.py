"""
蛋白质表示转换工具模块

包含三种核心蛋白质三维结构表示方法的转换工具函数：
- base_converter: 基础转换器，包含所有转换器的共同功能
- atom14_converter: Atom14 数据类的转换工具和辅助函数
- atom37_converter: Atom37 数据类的转换工具和辅助函数  
- frame_converter: Frame 数据类的转换工具和辅助函数

所有转换工具都基于 PyTorch 张量，支持 GPU 加速和自动微分。
这些模块主要提供底层的转换逻辑，而主要的 API 通过数据类暴露。
"""

from . import base_converter
from . import atom14_converter
from . import atom37_converter
from . import frame_converter

__all__ = [
    "base_converter",
    "atom14_converter",
    "atom37_converter", 
    "frame_converter",
] 