#!/usr/bin/env python3
"""
警告过滤器使用示例

本脚本展示如何在使用 ProtRepr 时过滤 BioPython 的非关键警告。
"""

import sys
from pathlib import Path

# 将项目根目录下的 src 目录添加到 Python 解释器的搜索路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# 应用警告过滤器 - 在任何其他导入之前
from protrepr.utils.warnings_filter import apply_default_filters
apply_default_filters()

# 现在可以正常导入和使用 ProtRepr，不会看到 BioPython 警告
try:
    import protrepr  # type: ignore
    print("✅ 警告过滤器已应用，BioPython 警告已被抑制")
    print("现在可以正常使用 ProtRepr 功能而不会看到大量警告信息")
    
    # 示例：如果你有蛋白质文件，可以这样使用 ProteinTensor
    # from protein_tensor import load_structure
    # protein_tensor = load_structure("path/to/your/protein.pdb", backend="torch")
    # print(f"加载蛋白质文件成功: {protein_tensor.n_atoms} 个原子")
    
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所需的依赖包")

# 如果需要在调试时恢复警告
# from protrepr.utils.warnings_filter import reset_warnings
# reset_warnings()
# print("警告过滤器已重置，现在会显示所有警告") 