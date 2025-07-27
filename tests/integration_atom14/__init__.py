"""
Atom14 集成测试包

本包提供对 ProtRepr Atom14 实现的完整端到端测试，包括：
- CIF/PDB 文件读取与转换
- Atom14 表示的 NumPy 和 PyTorch 格式保存/加载
- Atom14 到 CIF/PDB 的重建和验证
- 数据一致性和精度验证
- 性能基准测试

测试覆盖完整的数据流：
CIF/PDB → ProteinTensor → Atom14 → NPZ/PT → Atom14 → CIF/PDB
"""

from .test_atom14_end_to_end import *

__all__ = [
    "TestAtom14EndToEnd",
    "test_complete_workflow",
    "test_data_consistency",
    "test_file_formats_roundtrip"
] 