# Frame 批量处理工具

本目录包含用于 **Frame 表示法**的批量转换工具。Frame 是一种基于刚体变换的蛋白质表示方法，使用每个残基的旋转矩阵和平移向量（CA原子坐标）来代替传统的原子坐标表示。

## 🎯 功能概览

Frame 批处理工具提供两个核心功能：

1. **PDB/CIF → Frame**: 将蛋白质结构文件批量转换为 Frame 表示
2. **Frame → CIF/PDB**: 将 Frame 表示批量转换回结构文件

## 📋 可用脚本

### 1. `batch_pdb_to_frame.py` - PDB/CIF 到 Frame 转换

将 PDB/CIF 文件批量转换为 Frame 格式（.pt 文件）。

**基本语法:**
```bash
python batch_pdb_to_frame.py input_path output_dir [options]
```

**常用示例:**
```bash
# 基本转换 (保存为 Frame 实例)
python batch_pdb_to_frame.py /data/pdb_files /data/frame_output

# 保存为字典格式
python batch_pdb_to_frame.py /data/pdb_files /data/frame_output --save-as-dict

# 使用 8 个并行进程
python batch_pdb_to_frame.py /data/pdb_files /data/frame_output --workers 8

# 使用 GPU 加速
python batch_pdb_to_frame.py /data/pdb_files /data/frame_output --device cuda

# 扁平输出结构（不保持目录层次）
python batch_pdb_to_frame.py /data/pdb_files /data/frame_output --no-preserve-structure
```

**支持的输入格式:**
- `.pdb` - 标准 PDB 格式
- `.ent` - PDB 实体文件
- `.cif` - mmCIF 格式
- `.mmcif` - macromolecular CIF 格式

### 2. `batch_frame_to_cif.py` - Frame 到 CIF/PDB 转换

将 Frame 格式文件批量转换为 CIF 或 PDB 文件。

**基本语法:**
```bash
python batch_frame_to_cif.py input_path output_dir [options]
```

**常用示例:**
```bash
# 转换为 CIF 格式 (默认)
python batch_frame_to_cif.py /data/frame_files /data/cif_output

# 转换为 PDB 格式
python batch_frame_to_cif.py /data/frame_files /data/pdb_output --format pdb

# 使用并行处理
python batch_frame_to_cif.py /data/frame_files /data/output --workers 8

# 扁平输出结构
python batch_frame_to_cif.py /data/frame_files /data/output --no-preserve-structure
```

## 📊 参数详解

### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input_path` | 输入文件或目录路径 | 必需 |
| `output_dir` | 输出目录路径 | 必需 |
| `--workers, -w` | 并行工作进程数 | CPU核心数的一半 |
| `--no-preserve-structure` | 不保持目录结构，扁平化输出 | False |
| `--recursive, -r` | 递归搜索子目录 | True |
| `--save-stats` | 保存统计信息到 JSON 文件 | 无 |
| `--verbose, -v` | 详细输出模式 | False |

### PDB/CIF → Frame 特有参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--device` | 计算设备 (cpu/cuda) | cpu |
| `--save-as-dict` | 保存为字典格式而非实例 | False |

### Frame → CIF/PDB 特有参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--format, -f` | 输出格式 (cif/pdb) | cif |

## 🚀 使用示例

### 场景1: 单文件转换
```bash
# 转换单个 PDB 文件为 Frame
python batch_pdb_to_frame.py protein.pdb output_dir

# 转换单个 Frame 文件为 CIF
python batch_frame_to_cif.py frame.pt output_dir
```

### 场景2: 批量转换整个目录
```bash
# 转换目录中所有结构文件为 Frame 格式
python batch_pdb_to_frame.py /data/pdbs /data/frames --workers 16

# 转换目录中所有 Frame 文件为 PDB 格式
python batch_frame_to_cif.py /data/frames /data/pdbs --format pdb --workers 16
```

### 场景3: 高性能批量处理
```bash
# 使用 GPU 加速 + 高并行度 + 统计信息保存
python batch_pdb_to_frame.py \
    /data/large_dataset \
    /data/frame_output \
    --device cuda \
    --workers 32 \
    --save-stats conversion_stats.json \
    --verbose
```

### 场景4: 往返转换验证
```bash
# 第一步: PDB → Frame
python batch_pdb_to_frame.py original_pdbs frame_intermediate --workers 8

# 第二步: Frame → PDB (验证)
python batch_frame_to_cif.py frame_intermediate reconstructed_pdbs --format pdb --workers 8
```

## 📈 性能优化建议

### 1. 并行处理
- **CPU密集型**: 设置 `--workers` 为 CPU 核心数
- **IO密集型**: 可设置为 CPU 核心数的 1.5-2 倍
- **内存限制**: 减少并行数以避免内存溢出

### 2. 设备选择
- **小批量** (< 100 文件): CPU 即可
- **大批量** (> 1000 文件): 推荐使用 GPU
- **超大批量** (> 10000 文件): GPU + 高并行度

### 3. 存储格式
- **开发/调试**: 使用 `--save-as-dict` 便于检查
- **生产/推理**: 使用实例格式 (默认) 性能更好

## 📋 输出文件结构

### 保持目录结构 (默认)
```
输入:
/data/proteins/
├── group1/
│   ├── protein1.pdb
│   └── protein2.cif
└── group2/
    └── protein3.pdb

输出:
/data/frames/
├── group1/
│   ├── protein1.pt
│   └── protein2.pt
└── group2/
    └── protein3.pt
```

### 扁平化结构 (`--no-preserve-structure`)
```
输入:
/data/proteins/
├── group1/protein1.pdb
├── group1/protein2.cif
└── group2/protein3.pdb

输出:
/data/frames/
├── protein1.pt
├── protein2.pt
└── protein3.pt
```

## 📊 统计信息

使用 `--save-stats` 参数可以保存详细的转换统计信息到 JSON 文件：

```json
{
  "total": 1000,
  "success": 995,
  "failed": 5,
  "failed_files": ["protein_x.pdb", "protein_y.cif"],
  "results": [
    {
      "input_file": "protein1.pdb",
      "output_file": "protein1.pt",
      "success": true,
      "processing_time": 0.123,
      "num_residues": 150,
      "num_atoms": 600,
      "num_chains": 1
    }
  ],
  "converter_settings": {
    "device": "cuda",
    "workers": 16,
    "preserve_structure": true
  }
}
```

## ⚠️ 注意事项

### 1. Frame 表示的特性
- **只保留主链原子**: N, CA, C, O
- **残基级表示**: 每个残基用一个刚体变换表示
- **数据压缩**: 比完整原子坐标更紧凑

### 2. 转换精度
- **往返误差**: Frame → PDB → Frame 可能有小量精度损失
- **残基过滤**: 缺少主链原子的残基会被过滤
- **侧链丢失**: Frame 不包含侧链信息

### 3. 性能考虑
- **GPU 内存**: 大型蛋白质可能需要更多 GPU 内存
- **并行上限**: 过多并行进程可能导致内存不足
- **IO 瓶颈**: 高速存储有助于提升整体性能

## 🐛 故障排除

### 常见错误及解决方案

1. **内存不足**
   ```bash
   # 减少并行进程数
   python batch_pdb_to_frame.py input output --workers 2
   ```

2. **CUDA 不可用**
   ```bash
   # 回退到 CPU
   python batch_pdb_to_frame.py input output --device cpu
   ```

3. **文件格式不支持**
   ```bash
   # 检查输入文件扩展名是否为 .pdb, .cif, .ent, .mmcif
   ls -la input_dir/*.{pdb,cif,ent,mmcif}
   ```

4. **权限错误**
   ```bash
   # 确保输出目录有写权限
   chmod 755 output_dir
   ```

## 📚 相关文档

- [Frame 表示法原理](../docs/frame_implementation_completion_report.md)
- [ProtRepr 项目文档](../README.md)
- [Atom14/Atom37 批处理工具](./ATOM14_README.md)

## 🤝 技术支持

如遇到问题，请：
1. 检查输入文件格式和路径
2. 使用 `--verbose` 模式获取详细日志
3. 查看统计信息中的 `failed_files` 列表
4. 参考上述故障排除指南 