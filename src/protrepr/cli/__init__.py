"""
ProtRepr 命令行工具模块

本模块提供了一系列命令行工具，用于在ProtRepr的三种核心表示格式
(Atom14, Atom37, Frame) 和结构文件之间进行转换。

可用命令:
- protrepr-struct-to-atom14: 结构文件 → Atom14
- protrepr-struct-to-atom37: 结构文件 → Atom37  
- protrepr-struct-to-frame: 结构文件 → Frame
- protrepr-atom14-to-struct: Atom14 → 结构文件
- protrepr-atom37-to-struct: Atom37 → 结构文件
- protrepr-frame-to-struct: Frame → 结构文件

所有命令都支持批量处理、并行计算和详细的统计报告。
"""

__all__ = [] 