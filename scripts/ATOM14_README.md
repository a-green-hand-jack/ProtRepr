# Scripts 目录

本目录包含 ProtRepr 项目的命令行工具，为用户提供便捷的蛋白质数据处理接口。

## 设计原则

- **轻量包装器**: 脚本文件只负责命令行参数解析和用户界面，核心实现位于 `src/protrepr/` 中
- **简洁易用**: 提供直观的命令行接口，支持常见的数据处理任务
- **模块化**: 每个脚本专注于一个特定的功能
- **双向转换**: 支持正向转换（PDB/CIF → Atom14/Atom37）和反向转换（Atom14/Atom37 → CIF/PDB）

## 可用工具

### `batch_pdb_to_atom14.py`

批量将 PDB/CIF 文件转换为 ProtRepr Atom14 格式的命令行工具。

**核心实现**: `src/protrepr/batch_processing/atom14_batch_converter.py`

#### 功能特性

- 🚀 **高性能**: 支持多进程并行处理
- 📁 **灵活输入**: 支持单文件或目录批量处理
- 🔄 **PyTorch 原生**: 仅支持 PyTorch PT 格式，无 NumPy 依赖
- 💾 **格式选择**: 支持 Atom14 实例或字典格式保存
- 📊 **详细统计**: 提供完整的转换统计和错误报告
- 🎯 **精确控制**: 可配置设备、工作进程数、目录结构保持等

#### 基本用法

```bash
# 转换单个文件 (保存为 Atom14 实例)
python batch_pdb_to_atom14.py protein.pdb output_dir

# 保存为字典格式
python batch_pdb_to_atom14.py protein.pdb output_dir --save-as-dict

# 批量转换目录中的所有结构文件
python batch_pdb_to_atom14.py /path/to/pdb_files /path/to/output

# 使用多进程加速处理
python batch_pdb_to_atom14.py /path/to/pdb_files /path/to/output --workers 8

# 保存转换统计信息
python batch_pdb_to_atom14.py proteins/ output/ --save-stats stats.json
```

#### 新特性示例 (v2.0)

```bash
# 保存为 Atom14 实例 (默认，推荐用于直接加载使用)
python batch_pdb_to_atom14.py /data/proteins /data/atom14_instances

# 保存为字典格式 (与旧版本兼容)
python batch_pdb_to_atom14.py /data/proteins /data/atom14_dicts --save-as-dict

# 结合并行处理和字典格式
python batch_pdb_to_atom14.py /data/proteins /data/output --workers 8 --save-as-dict
```

#### 高级选项

- `--workers, -w`: 并行工作进程数（默认：CPU核心数的一半）
- `--no-preserve-structure`: 不保持目录结构，所有输出文件放在同一目录
- `--device`: 计算设备（`cpu` 或 `cuda`）
- `--save-as-dict`: 保存为字典格式而非 Atom14 实例
- `--save-stats`: 保存详细统计信息到 JSON 文件
- `--verbose, -v`: 详细输出模式

#### 输出数据结构

**Atom14 实例格式** (默认，推荐):
```python
# 直接加载和使用
from protrepr.core.atom14 import Atom14
atom14 = Atom14.load("output.pt")

# 访问属性
print(f"残基数: {atom14.num_residues}")
print(f"坐标形状: {atom14.coords.shape}")

# 转换为 CIF 文件验证
atom14.to_cif("verify.cif")
```

**字典格式** (与旧版本兼容):
```python
import torch
data = torch.load("output.pt")

# 数据字段
coords = data['coords']          # (num_residues, 14, 3)
atom_mask = data['atom_mask']    # (num_residues, 14)
res_mask = data['res_mask']      # (num_residues,)
# ... 其他字段
metadata = data['metadata']      # 包含格式版本、设备信息等

# 重构为 Atom14 实例
from protrepr.core.atom14 import Atom14
atom14 = Atom14.load("output.pt")  # 自动识别格式
```

#### 性能优化建议

1. **保存格式选择**: 
   - 实例格式：更快的加载速度，直接可用的 API
   - 字典格式：更好的版本兼容性，更小的文件尺寸
2. **并行处理**: 对于大量文件，使用 `--workers` 参数
3. **设备选择**: 如有 GPU，使用 `--device cuda`
4. **内存管理**: 处理超大蛋白质时，减少工作进程数

#### 版本变更 (v2.0)

- ✅ **新增**: `--save-as-dict` 参数控制保存格式
- ✅ **简化**: 移除 NPZ 格式支持，专注 PyTorch 生态
- ✅ **优化**: 使用 `Atom14.save()` 方法，统一保存逻辑
- ✅ **增强**: 默认保存完整 Atom14 实例，提供更好的用户体验

#### 错误处理

- 自动跳过无法解析的文件
- 详细的错误日志记录
- 完整的失败文件列表
- 非零退出码表示处理过程中有失败

#### 故障排除

- **内存不足**: 减少 `--workers` 数量
- **CUDA 错误**: 检查 GPU 可用性，或回退到 CPU
- **文件权限**: 确保对输入文件有读权限，对输出目录有写权限
- **依赖问题**: 确保 `protein-tensor` 库正确安装
- **格式兼容**: 旧版本生成的字典格式可以正常加载为 Atom14 实例

#### 完整使用示例

```bash
# 1. 基本转换 (推荐)
python scripts/batch_pdb_to_atom14.py /tests/data /tests/atom14_e2e 

# 2. 高性能批量处理
python scripts/batch_pdb_to_atom14.py /data/large_dataset /data/output \
    --workers 16 \
    --device cpu \
    --save-stats processing_stats.json

# 3. 兼容模式 (字典格式)
python scripts/batch_pdb_to_atom14.py /data/proteins /data/legacy_output \
    --save-as-dict \
    --no-preserve-structure

# 4. 验证工作流
python scripts/batch_pdb_to_atom14.py sample.pdb output/ --verbose
# 然后使用 Python 加载和验证：
# >>> from protrepr.core.atom14 import Atom14
# >>> atom14 = Atom14.load("output/sample.pt")
# >>> atom14.to_cif("verification.cif")
```

### `batch_atom14_to_cif.py` (新增)

批量将 ProtRepr Atom14 格式文件转换为 CIF 或 PDB 结构文件的反向转换工具。

**核心实现**: `src/protrepr/batch_processing/atom14_to_cif_converter.py`

#### 功能特性

- 🔄 **反向转换**: 将 Atom14 PT 文件转换回可视化的结构文件
- 📁 **多格式支持**: 输出 CIF 或 PDB 格式
- 🚀 **高性能**: 支持多进程并行处理
- 📊 **详细统计**: 提供完整的转换统计和错误报告
- 🎯 **精确控制**: 可配置工作进程数、目录结构保持等

#### 基本用法

```bash
# 转换为 CIF 格式 (默认)
python batch_atom14_to_cif.py /path/to/atom14_files /path/to/output

# 转换为 PDB 格式
python batch_atom14_to_cif.py /path/to/atom14_files /path/to/output --format pdb

# 批量转换目录中的所有 Atom14 文件
python batch_atom14_to_cif.py /data/atom14_pt_files /data/cif_output

# 使用多进程加速处理
python batch_atom14_to_cif.py /data/atom14_files /data/output --workers 8

# 保存转换统计信息
python batch_atom14_to_cif.py atom14_files/ cif_output/ --save-stats reverse_stats.json
```

#### 高级选项

- `--format, -f`: 输出格式（`cif` 或 `pdb`，默认：`cif`）
- `--workers, -w`: 并行工作进程数（默认：CPU核心数的一半）
- `--no-preserve-structure`: 不保持目录结构，所有输出文件放在同一目录
- `--save-stats`: 保存详细统计信息到 JSON 文件
- `--verbose, -v`: 详细输出模式

#### 使用场景

1. **结果可视化**: 将训练好的模型输出转换为可在 PyMOL/ChimeraX 中查看的格式
2. **质量检查**: 验证 Atom14 数据的完整性和正确性
3. **数据交换**: 与其他不支持 Atom14 格式的工具进行数据交换
4. **发布共享**: 将研究结果转换为标准格式供他人使用

#### 完整使用示例

```bash
# 1. 基本反向转换 (CIF 格式)
python scripts/batch_atom14_to_cif.py /data/atom14_files /data/cif_output

# 2. 转换为 PDB 格式用于可视化
python scripts/batch_atom14_to_cif.py /results/atom14 /results/visualization \
    --format pdb \
    --workers 8

# 3. 完整工作流验证
python scripts/batch_atom14_to_cif.py atom14_sample.pt verification_output/ \
    --format cif \
    --verbose \
    --save-stats verification_stats.json

# 4. 批量处理实验结果
python scripts/batch_atom14_to_cif.py /experiments/atom14_predictions /publish/structures \
    --format cif \
    --no-preserve-structure
```

#### 输出验证

转换完成后，可以使用以下方式验证输出：

```bash
# 使用 PyMOL 查看 CIF 文件
pymol output.cif

# 使用 ChimeraX 查看 PDB 文件
chimerax output.pdb

# 或者使用 Python 验证
python -c "
from protein_tensor import load_structure
protein = load_structure('output.cif')
print(f'残基数: {protein.n_residues}')
print(f'原子数: {protein.n_atoms}')
"
```

## 开发指南

### 添加新工具

1. 在 `src/protrepr/` 中实现核心功能
2. 在 `scripts/` 中创建轻量命令行包装器
3. 在 `tests/test_converter/` 中添加相应测试
4. 更新本 README

### 设计准则

- 遵循 Unix 哲学: 一个工具做好一件事
- 提供清晰的帮助信息和错误消息
- 支持标准的输入输出重定向
- 使用一致的命令行参数约定

## 相关文档

- [批量处理模块文档](../src/protrepr/batch_processing/)
- [Atom14 核心实现](../src/protrepr/core/atom14.py)
- [测试用例](../tests/test_converter/)
- [项目主文档](../README.md) 