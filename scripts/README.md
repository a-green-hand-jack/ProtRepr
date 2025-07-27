# Scripts 目录

本目录包含 ProtRepr 项目的命令行工具，为用户提供便捷的蛋白质数据处理接口。

## 设计原则

- **轻量包装器**: 脚本文件只负责命令行参数解析和用户界面，核心实现位于 `src/protrepr/` 中
- **简洁易用**: 提供直观的命令行接口，支持常见的数据处理任务
- **模块化**: 每个脚本专注于一个特定的功能

## 可用工具

### `batch_pdb_to_atom14.py`

批量将 PDB/CIF 文件转换为 ProtRepr Atom14 格式的命令行工具。

**核心实现**: `src/protrepr/batch_processing/atom14_batch_converter.py`

#### 功能特性

- 🚀 **高性能**: 支持多进程并行处理
- 📁 **灵活输入**: 支持单文件或目录批量处理
- 🔄 **多格式支持**: 输入支持 PDB、CIF、ENT、mmCIF；输出支持 NPZ、PyTorch PT
- 📊 **详细统计**: 提供完整的转换统计和错误报告
- 🎯 **精确控制**: 可配置设备、工作进程数、目录结构保持等

#### 基本用法

```bash
# 转换单个文件
python batch_pdb_to_atom14.py protein.pdb output_dir

# 批量转换目录中的所有结构文件
python batch_pdb_to_atom14.py /path/to/pdb_files /path/to/output

# 使用多进程加速处理
python batch_pdb_to_atom14.py /data/proteins /data/atom14 --workers 8

# 输出为 PyTorch 格式
python batch_pdb_to_atom14.py input.cif output_dir --format pt

# 保存转换统计信息
python batch_pdb_to_atom14.py proteins/ output/ --save-stats stats.json
```

#### 高级选项

- `--workers, -w`: 并行工作进程数（默认：CPU核心数的一半）
- `--no-preserve-structure`: 不保持目录结构，所有输出文件放在同一目录
- `--device`: 计算设备（`cpu` 或 `cuda`）
- `--format, -f`: 输出格式（`npz` 或 `pt`）
- `--save-stats`: 保存详细统计信息到 JSON 文件
- `--verbose, -v`: 详细输出模式

#### 输出数据结构

**NPZ 格式** (推荐用于数据存储):
```python
data = np.load("output.npz")
# 包含: coords, atom_mask, res_mask, chain_ids, residue_types,
#      residue_indices, chain_residue_indices, residue_names, 
#      atom_names, num_residues, num_chains
```

**PyTorch 格式** (推荐用于模型训练):
```python
data = torch.load("output.pt")
# 包含上述所有数据 + metadata 字典
metadata = data['metadata']  # 格式版本、设备信息等
```

#### 性能优化建议

1. **并行处理**: 对于大量文件，使用 `--workers` 参数
2. **设备选择**: 如有 GPU，使用 `--device cuda` 
3. **格式选择**: NPZ 格式文件更小，PT 格式加载更快
4. **内存管理**: 处理超大蛋白质时，减少工作进程数

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