"""
核心功能模块

包含 ProtRepr 框架的核心数据类和基础功能：
- Atom14: 紧凑型原子表示数据类
- Atom37: 固定大小重原子表示数据类  
- Frame: 刚体坐标系表示数据类

所有数据类都支持：
- @dataclass 装饰器，便于属性访问
- 与 ProteinTensor 的双向转换
- PyTorch 原生支持和 GPU 加速
- 完整的验证和设备管理功能
"""

from .atom14 import Atom14
from .atom37 import Atom37
from .frame import Frame

__all__ = [
    "Atom14",
    "Atom37", 
    "Frame",
] 