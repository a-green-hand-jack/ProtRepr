# ProtRepr: 蛋白质表示学习框架

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

ProtRepr 是一个基于开源库 [ProteinTensor](https://github.com/a-green-hand-jack/ProteinTensor) 的二次开发项目，专注于蛋白质表示学习、结构预测和功能分析的深度学习研究与应用框架。

## 🎯 核心目标

充分利用 `ProteinTensor` 提供的蛋白质结构到张量的转换能力，构建一个专注于蛋白质表示学习的深度学习框架。所有开发工作都围绕如何更高效地将蛋白质结构数据与 PyTorch 生态中的先进模型（如 GNNs, Transformers, SE(3)-Equivariant Networks）相结合。

## 🚀 核心功能

### 三种标准蛋白质表示方法

1. **Atom37 表示法** - 基于残基的固定大小重原子表示
   - 每个残基用 `(37, 3)` 坐标张量表示所有重原子位置
   - 涵盖20种标准氨基酸的所有重原子类型
   - 配套 `atom37_mask` 标识真实原子位置

2. **Atom14 表示法** - 紧凑型原子表示
   - 每个残基用 `(14, 3)` 坐标张量表示关键原子
   - 包含主链原子（N, Cα, C, O）和重要侧链原子
   - 更节省内存的同时保留关键几何信息

3. **Frame 表示法** - 基于残基的刚体坐标系
   - 为每个残基定义局部刚体变换 `(translation, rotation)`
   - 支持 SE(3)-equivariant 网络的核心需求
   - 通过主链原子的 Gram-Schmidt 正交化计算

## 🔧 技术特色

- **PyTorch-Native**: 所有计算直接在 GPU 上完成，避免不必要的数据传输
- **强制 PyTorch 后端**: 确保与深度学习工作流的无缝集成
- **高性能**: 优化的张量操作，支持批处理和自动微分
- **可扩展**: 模块化设计，易于集成新的表示方法和模型架构
- **命令行工具**: 提供完整的命令行工具套件，支持批量处理

## 📦 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- Git SSH 访问权限（用于安装 ProteinTensor）

### 从 GitHub 安装

```bash
# 使用 uv（推荐）
uv pip install git+ssh://git@github.com/a-green-hand-jack/ProtRepr.git

# 使用 pip
pip install git+ssh://git@github.com/a-green-hand-jack/ProtRepr.git
```

### 开发安装

```bash
# 克隆仓库
git clone git@github.com:a-green-hand-jack/ProtRepr.git
cd ProtRepr

# 创建虚拟环境
uv venv
source .venv/bin/activate  # Linux/macOS

# 开发模式安装
uv pip install -e .

# 安装开发依赖
uv pip install -e ".[dev]"
```

**注意**: 项目依赖的 `protein-tensor` 库将从 GitHub 仓库直接安装。

## 🛠️ 命令行工具

安装后，ProtRepr 提供了一套完整的命令行工具，支持在三种表示格式和结构文件之间进行批量转换。

### 可用命令

#### 结构文件 → 表示格式

```bash
# 结构文件转换为 Atom14 格式
protrepr-struct-to-atom14 /path/to/structures /path/to/output

# 结构文件转换为 Atom37 格式
protrepr-struct-to-atom37 /path/to/structures /path/to/output

# 结构文件转换为 Frame 格式
protrepr-struct-to-frame /path/to/structures /path/to/output
```

#### 表示格式 → 结构文件

```bash
# Atom14 转换为结构文件
protrepr-atom14-to-struct /path/to/atom14_files /path/to/output

# Atom37 转换为结构文件
protrepr-atom37-to-struct /path/to/atom37_files /path/to/output

# Frame 转换为结构文件
protrepr-frame-to-struct /path/to/frame_files /path/to/output
```

### 支持的文件格式

- **输入结构文件**: `.pdb`, `.cif`, `.ent`, `.mmcif`
- **表示格式文件**: `.pt` (PyTorch 格式)
- **输出结构文件**: `.cif`, `.pdb`

### 命令行工具示例

```bash
# 基本转换
protrepr-struct-to-atom14 protein.pdb output/

# 批量转换，使用8个并行进程
protrepr-struct-to-atom37 /data/structures/ /data/atom37/ --workers 8

# 转换为PDB格式
protrepr-atom14-to-struct /data/atom14/ /data/pdbs/ --format pdb

# 使用GPU加速
protrepr-struct-to-frame /data/structures/ /data/frames/ --device cuda

# 保存为字典格式而非类实例
protrepr-struct-to-atom14 /data/structures/ /data/output/ --save-as-dict

# 保存统计信息
protrepr-struct-to-atom37 /data/structures/ /data/output/ --save-stats stats.json

# 详细输出模式
protrepr-atom37-to-struct /data/atom37/ /data/output/ --verbose
```

### 通用参数

所有命令行工具都支持以下参数：

- `--workers, -w`: 并行工作进程数（默认：CPU核心数的一半）
- `--no-preserve-structure`: 不保持目录结构，平铺输出
- `--save-stats`: 保存详细的转换统计信息到JSON文件
- `--verbose, -v`: 详细输出模式
- `--help, -h`: 显示帮助信息

## 💻 Python API 使用

### 快速开始

```python
import torch
from protein_tensor import load_structure
from protrepr import Atom14, Atom37, Frame

# 加载蛋白质结构（强制使用 PyTorch 后端）
protein_pt = load_structure("protein.pdb", backend='torch')

# 创建三种不同的表示格式
atom14 = Atom14.from_protein_tensor(protein_pt)
atom37 = Atom37.from_protein_tensor(protein_pt)
frame = Frame.from_protein_tensor(protein_pt)

print(f"Atom14 shape: {atom14.coords.shape}")      # (num_residues, 14, 3)
print(f"Atom37 shape: {atom37.coords.shape}")      # (num_residues, 37, 3)
print(f"Frame translations: {frame.translations.shape}")  # (num_residues, 3)
print(f"Frame rotations: {frame.rotations.shape}")        # (num_residues, 3, 3)
```

### 核心类详细用法

#### 1. Atom14 类

```python
# 创建和加载
atom14 = Atom14.from_protein_tensor(protein_pt)

# 属性访问
coords = atom14.coords              # (num_residues, 14, 3) 原子坐标
atom_mask = atom14.atom_mask        # (num_residues, 14) 原子掩码
res_mask = atom14.res_mask          # (num_residues,) 残基掩码
chain_ids = atom14.chain_ids        # (num_residues,) 链标识符

# 几何操作
backbone = atom14.get_backbone_coords()    # 获取主链原子坐标 (N, CA, C, O)
sidechain = atom14.get_sidechain_coords()  # 获取侧链原子坐标

# 链操作
chain_residues = atom14.get_chain_residues('A')  # 获取A链的残基

# 设备管理
atom14_gpu = atom14.to_device(torch.device("cuda"))
atom14_cpu = atom14.to_device(torch.device("cpu"))

# 保存和加载
atom14.save("atom14_data.pt")  # 保存为实例
atom14.save("atom14_dict.pt", save_as_instance=False)  # 保存为字典

loaded_atom14 = Atom14.load("atom14_data.pt")

# 转换回ProteinTensor
protein_tensor = atom14.to_protein_tensor()

# 导出为CIF文件
atom14.to_cif("output.cif")
```

#### 2. Atom37 类

```python
# 创建和加载
atom37 = Atom37.from_protein_tensor(protein_pt)

# 属性访问（类似Atom14，但有37个原子位置）
coords = atom37.coords              # (num_residues, 37, 3)
atom_mask = atom37.atom_mask        # (num_residues, 37)

# 获取特定残基的原子
residue_atoms = atom37.get_residue_atoms(0)  # 获取第0个残基的所有原子
ca_coord = residue_atoms['CA']      # 获取CA原子坐标

# 计算质心
center_of_mass = atom37.compute_center_of_mass()  # (num_residues, 3)

# 几何操作
backbone = atom37.get_backbone_coords()    # (num_residues, 4, 3)
sidechain = atom37.get_sidechain_coords()  # (num_residues, 33, 3)

# 保存和加载
atom37.save("atom37_data.pt")
loaded_atom37 = Atom37.load("atom37_data.pt")

# 导出为CIF文件
atom37.to_cif("output.cif")
```

#### 3. Frame 类

```python
# 创建和加载
frame = Frame.from_protein_tensor(protein_pt)

# 属性访问
translations = frame.translations   # (num_residues, 3) CA原子坐标
rotations = frame.rotations         # (num_residues, 3, 3) 旋转矩阵
res_mask = frame.res_mask          # (num_residues,) 残基掩码

# 重建主链坐标
backbone_coords = frame.get_backbone_coords()  # 从刚体变换重建主链

# 获取局部坐标系中的标准原子位置
local_coords = frame.get_local_coordinates()
n_local = local_coords['N']    # N原子在局部坐标系中的位置
ca_local = local_coords['CA']  # CA原子在局部坐标系中的位置

# 保存和加载
frame.save("frame_data.pt")
loaded_frame = Frame.load("frame_data.pt")

# 导出为CIF文件
frame.to_cif("output.cif")
```

### 数据验证和属性

所有核心类都提供以下标准属性和方法：

```python
# 设备信息
device = atom14.device               # 张量所在设备
batch_shape = atom14.batch_shape     # 批量维度形状
num_residues = atom14.num_residues   # 残基数量
num_chains = atom14.num_chains       # 链数量

# 数据验证
atom14.validate()  # 验证数据一致性和有效性

# 类型信息
print(f"数据类型: {type(atom14)}")
print(f"坐标类型: {type(atom14.coords)}")  # torch.Tensor
```

### 批量处理 API

除了命令行工具，你也可以在Python中使用批量处理功能：

```python
from protrepr.batch_processing import (
    BatchPDBToAtom14Converter,
    BatchAtom14ToCIFConverter,
    save_statistics
)

# 创建批量转换器
converter = BatchPDBToAtom14Converter(
    n_workers=8,
    device='cuda',
    save_as_instance=True
)

# 执行批量转换
statistics = converter.convert_batch(
    input_path='/path/to/structures',
    output_dir='/path/to/output'
)

# 保存统计信息
save_statistics(statistics, 'conversion_stats.json')

print(f"成功转换: {statistics['success']} 个文件")
print(f"转换失败: {statistics['failed']} 个文件")
```

## ⚠️ 重要注意事项

### 1. PyTorch 后端要求

**关键**: ProtRepr 项目强制要求使用 PyTorch 后端。在使用 `protein_tensor.load_structure()` 时，必须显式指定 `backend='torch'`：

```python
# 正确用法
protein_pt = load_structure("protein.pdb", backend='torch')

# 错误用法（会导致后续操作失败）
protein_np = load_structure("protein.pdb", backend='numpy')
```

### 2. 设备管理

- 所有张量操作都支持GPU加速
- 使用 `.to_device()` 方法在CPU和GPU之间转移数据
- 命令行工具支持 `--device cuda` 参数

### 3. 内存注意事项

- Atom37 比 Atom14 占用更多内存（37 vs 14 个原子位置）
- Frame 表示最节省内存（仅存储刚体变换）
- 大批量处理时建议适当调整 `--workers` 参数

### 4. 文件格式兼容性

- 支持的输入格式：PDB、CIF、ENT、MMCIF
- 输出的PyTorch文件使用 `.pt` 扩展名
- CIF输出完全兼容标准格式

## 📁 项目结构

```
ProtRepr/
├── src/protrepr/                    # 核心库代码
│   ├── __init__.py                 # API 导出
│   ├── cli/                        # 命令行工具
│   │   ├── struct_to_atom14.py    # 结构→Atom14 工具
│   │   ├── atom14_to_struct.py    # Atom14→结构 工具
│   │   └── ...                    # 其他CLI工具
│   ├── core/                       # 核心数据类
│   │   ├── atom14.py              # Atom14 数据类
│   │   ├── atom37.py              # Atom37 数据类
│   │   └── frame.py               # Frame 数据类
│   ├── batch_processing/          # 批量处理模块
│   ├── representations/           # 转换工具函数
│   └── utils/                     # 工具函数
├── tests/                         # 测试代码
├── scripts/                       # 开发脚本
├── docs/                          # 文档
├── notebooks/                     # 教程和示例
└── pyproject.toml                 # 项目配置
```

## 🧪 运行测试

```bash
# 运行所有测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov=src/protrepr --cov-report=html

# 查看 HTML 覆盖率报告
open htmlcov/index.html
```

## 📝 开发

本项目遵循严格的代码质量标准：

- **类型注解**: 100% 的函数和方法都有完整的类型注解
- **文档**: Google 风格的 Docstrings
- **日志**: 使用 `logging` 模块而非 `print`
- **路径管理**: 统一使用 `pathlib`
- **代码风格**: 遵循 PEP 8，使用 `black` 和 `isort` 格式化

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

本项目基于 [ProteinTensor](https://github.com/a-green-hand-jack/ProteinTensor) 开发，感谢原作者的杰出工作。

## 📧 联系

如有问题或建议，请通过 [Issues](https://github.com/your-org/protrepr/issues) 联系我们。 