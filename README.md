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

- **PyTorch-Native**: 所有计算支持在 GPU 上完成，但是不强制要求使用 GPU，支持 CPU 计算,而且经测试 CPU 计算已经足够高效
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
# 使用 ssh 安装

## 使用 uv（推荐）
uv pip install git+ssh://git@github.com/a-green-hand-jack/ProtRepr.git
## 使用 pip

pip install git+ssh://git@github.com/a-green-hand-jack/ProtRepr.git

# 使用 https 安装

## 使用 uv（推荐）
uv pip install git+https://github.com/a-green-hand-jack/ProtRepr.git

## 使用 pip
pip install git+https://github.com/a-green-hand-jack/ProtRepr.git
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
protein_pt = load_structure("protein.pdb")

# 创建三种不同的表示格式
atom14 = Atom14.from_protein_tensor(protein_pt)
atom37 = Atom37.from_protein_tensor(protein_pt)
frame = Frame.from_protein_tensor(protein_pt)

print(f"Atom14 shape: {atom14.coords.shape}")      # (num_residues, 14, 3)
print(f"Atom37 shape: {atom37.coords.shape}")      # (num_residues, 37, 3)
print(f"Frame translations: {frame.translations.shape}")  # (num_residues, 3)
print(f"Frame rotations: {frame.rotations.shape}")        # (num_residues, 3, 3)
```

### 核心类完整 API 参考

#### 1. Atom14 类 - 紧凑型原子表示

##### 数据属性 (Data Attributes)

```python
# 核心坐标和掩码数据
coords: torch.Tensor                    # (..., num_residues, 14, 3) - 原子坐标
atom_mask: torch.Tensor                 # (..., num_residues, 14) - 原子掩码 (1=真实, 0=填充)
res_mask: torch.Tensor                  # (..., num_residues) - 残基掩码 (1=标准, 0=非标准)

# 蛋白质元数据
chain_ids: torch.Tensor                 # (..., num_residues) - 链标识符编码
residue_types: torch.Tensor             # (..., num_residues) - 残基类型编号 (0-19)
residue_indices: torch.Tensor           # (..., num_residues) - 全局残基编号 (支持链间gap)
chain_residue_indices: torch.Tensor     # (..., num_residues) - 链内局部编号
residue_names: torch.Tensor             # (..., num_residues) - 残基名称编码
atom_names: torch.Tensor                # (14,) - atom14 原子名称编码

# 可选属性
b_factors: Optional[torch.Tensor]       # (..., num_residues, 14) - B因子
occupancies: Optional[torch.Tensor]     # (..., num_residues, 14) - 占用率
```

##### 属性方法 (Properties)

```python
device: torch.device                    # 张量所在设备
batch_shape: torch.Size                 # 批量维度形状 (...)
num_residues: int                       # 残基数量
num_chains: int                         # 链数量
```

##### 类方法 (Class Methods)

```python
# 创建和加载
Atom14.from_protein_tensor(protein_tensor, device=None)  # 从 ProteinTensor 创建
Atom14.load(filepath, map_location=None)                 # 从文件加载
```

##### 实例方法 (Instance Methods)

```python
# 核心转换方法
to_protein_tensor() -> ProteinTensor    # 转换回 ProteinTensor
to_device(device) -> Atom14             # 移动到指定设备
validate() -> None                      # 验证数据一致性

# 几何查询方法
get_backbone_coords() -> torch.Tensor   # 获取主链坐标 (..., num_residues, 4, 3)
get_sidechain_coords() -> torch.Tensor  # 获取侧链坐标 (..., num_residues, 10, 3)
get_chain_residues(chain_id) -> torch.Tensor  # 获取指定链的残基索引

# 数据持久化
save(filepath, save_as_instance=True)   # 保存数据到文件
to_cif(output_path)                     # 导出为 CIF 文件
```

##### 使用示例

```python
# 创建和基本操作
atom14 = Atom14.from_protein_tensor(protein_pt)
print(f"形状: {atom14.coords.shape}")           # (num_residues, 14, 3)
print(f"设备: {atom14.device}")                 # cpu 或 cuda

# 几何操作
backbone = atom14.get_backbone_coords()         # N, CA, C, O 原子坐标
sidechain = atom14.get_sidechain_coords()       # CB 及其他侧链原子
chain_a_residues = atom14.get_chain_residues(0) # A链残基索引

# 设备管理和保存
atom14_gpu = atom14.to_device(torch.device("cuda"))
atom14.save("data.pt", save_as_instance=True)   # 保存完整实例
atom14.save("data_dict.pt", save_as_instance=False)  # 保存字典格式
```

#### 2. Atom37 类 - 完整重原子表示

##### 数据属性 (Data Attributes)

```python
# 核心坐标和掩码数据 (与 Atom14 相同结构，但有37个原子位置)
coords: torch.Tensor                    # (..., num_residues, 37, 3) - 原子坐标
atom_mask: torch.Tensor                 # (..., num_residues, 37) - 原子掩码
res_mask: torch.Tensor                  # (..., num_residues) - 残基掩码

# 蛋白质元数据 (与 Atom14 相同)
chain_ids: torch.Tensor                 # (..., num_residues) - 链标识符
residue_types: torch.Tensor             # (..., num_residues) - 残基类型编号
residue_indices: torch.Tensor           # (..., num_residues) - 全局残基编号
chain_residue_indices: torch.Tensor     # (..., num_residues) - 链内局部编号
residue_names: torch.Tensor             # (..., num_residues) - 残基名称编码
atom_names: torch.Tensor                # (37,) - atom37 原子名称编码

# 可选属性
b_factors: Optional[torch.Tensor]       # (..., num_residues, 37) - B因子
occupancies: Optional[torch.Tensor]     # (..., num_residues, 37) - 占用率
```

##### 属性方法 (Properties)

```python
device: torch.device                    # 张量所在设备
batch_shape: torch.Size                 # 批量维度形状
num_residues: int                       # 残基数量
num_chains: int                         # 链数量
num_atoms_per_residue: int              # 每残基原子数 (固定为37)
```

##### 类方法和实例方法 (与 Atom14 相同)

```python
# 类方法
Atom37.from_protein_tensor(protein_tensor, device=None)
Atom37.load(filepath, map_location=None)

# 基础实例方法 (与 Atom14 相同)
to_protein_tensor(), to_device(), validate(), save(), to_cif()
get_backbone_coords(), get_sidechain_coords(), get_chain_residues()
```

##### Atom37 特有方法

```python
# 残基级别操作
get_residue_atoms(residue_idx: int) -> Dict[str, torch.Tensor]
    # 获取指定残基的所有原子，返回 {'CA': coord, 'N': coord, ...}

compute_center_of_mass() -> torch.Tensor
    # 计算每个残基的质心坐标 (..., num_residues, 3)
```

##### 使用示例

```python
atom37 = Atom37.from_protein_tensor(protein_pt)
print(f"形状: {atom37.coords.shape}")           # (num_residues, 37, 3)

# 获取主链和侧链 (比 Atom14 更完整)
backbone = atom37.get_backbone_coords()         # (..., num_residues, 4, 3)
sidechain = atom37.get_sidechain_coords()       # (..., num_residues, 33, 3)

# 残基级别操作
first_residue = atom37.get_residue_atoms(0)     # 第一个残基的所有原子
ca_coord = first_residue['CA']                  # CA 原子坐标
center_of_mass = atom37.compute_center_of_mass() # 每个残基的质心
```

#### 3. Frame 类 - 刚体坐标系表示

##### 数据属性 (Data Attributes)

```python
# 刚体变换数据
translations: torch.Tensor              # (..., num_residues, 3) - CA 原子坐标
rotations: torch.Tensor                 # (..., num_residues, 3, 3) - 局部坐标系旋转矩阵
res_mask: torch.Tensor                  # (..., num_residues) - 残基掩码

# 蛋白质元数据 (与其他类相同，但无 atom_names)
chain_ids: torch.Tensor                 # (..., num_residues) - 链标识符
residue_types: torch.Tensor             # (..., num_residues) - 残基类型编号
residue_indices: torch.Tensor           # (..., num_residues) - 全局残基编号
chain_residue_indices: torch.Tensor     # (..., num_residues) - 链内局部编号
residue_names: torch.Tensor             # (..., num_residues) - 残基名称编码

# 可选属性
b_factors: Optional[torch.Tensor]       # (..., num_residues) - 残基级别B因子
```

##### 属性方法 (Properties)

```python
device: torch.device                    # 张量所在设备
batch_shape: torch.Size                 # 批量维度形状
num_residues: int                       # 残基数量
num_chains: int                         # 链数量
```

##### 类方法和基础实例方法 (与其他类相同)

```python
# 类方法
Frame.from_protein_tensor(protein_tensor, device=None)
Frame.load(filepath, map_location=None)

# 基础实例方法
to_protein_tensor(), to_device(), validate(), save(), to_cif()
get_chain_residues()
```

##### Frame 特有方法

```python
get_backbone_coords() -> torch.Tensor
    # 从刚体变换重建主链坐标 (..., num_residues, 4, 3)
    # 注意：这是通过几何重建的，不是直接存储的坐标

get_local_coordinates() -> Dict[str, torch.Tensor]
    # 获取标准局部坐标系中的原子位置
    # 返回: {'N': local_pos, 'CA': local_pos, 'C': local_pos, 'O': local_pos}
```

##### 使用示例

```python
frame = Frame.from_protein_tensor(protein_pt)
print(f"平移形状: {frame.translations.shape}")    # (num_residues, 3)
print(f"旋转形状: {frame.rotations.shape}")       # (num_residues, 3, 3)

# 刚体变换操作
backbone_reconstructed = frame.get_backbone_coords()  # 重建主链坐标
local_coords = frame.get_local_coordinates()          # 标准局部坐标

# 获取局部坐标系中的标准位置
n_local = local_coords['N']      # N 原子在局部坐标系中的位置
ca_local = local_coords['CA']    # CA 原子 (原点)
c_local = local_coords['C']      # C 原子在局部坐标系中的位置
```

### 公共属性和通用操作

所有三个核心类都支持以下标准操作：

```python
# 设备和形状信息
device = instance.device                # torch.device - 张量所在设备
batch_shape = instance.batch_shape      # torch.Size - 批量维度形状
num_residues = instance.num_residues    # int - 残基数量
num_chains = instance.num_chains        # int - 链数量

# 数据验证
instance.validate()                     # 验证数据一致性和有效性

# 设备管理
instance_gpu = instance.to_device(torch.device("cuda"))
instance_cpu = instance.to_device(torch.device("cpu"))

# 数据持久化
instance.save("data.pt")                # 保存为实例
instance.save("data_dict.pt", save_as_instance=False)  # 保存为字典
loaded = ClassName.load("data.pt")      # 加载数据

# 格式转换
protein_tensor = instance.to_protein_tensor()  # 转换回 ProteinTensor
instance.to_cif("output.cif")           # 导出为 CIF 文件

# 链操作
chain_residues = instance.get_chain_residues(0)  # 获取指定链的残基
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

**关键**: ProtRepr 项目强制要求使用 PyTorch 后端。在使用 `protein_tensor.load_structure()` 时，无需调整,因为这个方法默认返回 `torch.Tensor` 类型的张量。

```python
protein_pt = load_structure("protein.pdb")
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
  - 但是,目前仅测试了 PDB 和 CIF 格式,对其他格式的兼容性未知,请谨慎使用.
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