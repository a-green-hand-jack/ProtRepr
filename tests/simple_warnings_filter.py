#!/usr/bin/env python3
"""
简单的警告过滤器

直接过滤 BioPython 的 PDB 构建警告。
在你的代码最开始处导入并调用这个函数。
"""

import warnings


def suppress_biopython_pdb_warnings():
    """
    抑制 BioPython PDB 解析时的元素推断警告。
    
    这些警告是由于 PDB 文件中原子缺少元素信息导致的，
    BioPython 会根据原子名称自动推断元素类型，这是正常行为。
    """
    # 过滤所有包含 "Used element" 的警告信息
    warnings.filterwarnings('ignore', message='.*Used element.*')
    
    # 过滤 Bio.PDB 模块的所有用户警告
    warnings.filterwarnings('ignore', category=UserWarning, module='Bio.PDB.*')
    
    print("✅ BioPython PDB 警告已被过滤")


if __name__ == "__main__":
    suppress_biopython_pdb_warnings()
    print("警告过滤器设置完成。")
    print("将此函数复制到你的代码开头即可过滤 BioPython 警告。")
    
    # 示例用法
    print("\n示例用法：")
    print("```python")
    print("import warnings")
    print("warnings.filterwarnings('ignore', message='.*Used element.*')")
    print("warnings.filterwarnings('ignore', category=UserWarning, module='Bio.PDB.*')")
    print("")
    print("# 现在可以正常使用你的代码，不会看到 BioPython 警告")
    print("from protein_tensor import load_structure")
    print("# ... 你的其他代码")
    print("```") 