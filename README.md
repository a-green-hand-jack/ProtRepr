# ProtRepr: 蛋白质表示学习框架

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

ProtRepr 是一个基于开源库 [ProteinTensor](https://github.com/a-green-hand-jack/ProteinTensor) 的二次开发项目，专注于蛋白质表示学习、结构预测和功能分析的深度学习研究与应用框架。

## 🎯 核心目标

充分利用 `ProteinTensor` 提供的蛋白质结构到张量的转换能力，构建一个专注于蛋白质表示学习的深度学习框架。所有开发工作都围绕如何更高效地将蛋白质结构数据与 PyTorch 生态中的先进模型（如 GNNs, Transformers, SE(3)-Equivariant Networks）相结合。

## 🚀 核心功能

### 三种标准蛋白质表示方法

1. **atom37 表示法** - 基于残基的固定大小重原子表示
   - 每个残基用 `(37, 3)` 坐标张量表示所有重原子位置
   - 涵盖20种标准氨基酸的所有重原子类型
   - 配套 `atom37_mask` 标识真实原子位置

2. **atom14 表示法** - 紧凑型原子表示
   - 每个残基用 `(14, 3)` 坐标张量表示关键原子
   - 包含主链原子（N, Cα, C, O）和重要侧链原子
   - 更节省内存的同时保留关键几何信息

3. **frame 表示法** - 基于残基的刚体坐标系
   - 为每个残基定义局部刚体变换 `(translation, rotation)`
   - 支持 SE(3)-equivariant 网络的核心需求
   - 通过主链原子的 Gram-Schmidt 正交化计算

## 🔧 技术特色

- **PyTorch-Native**: 所有计算直接在 GPU 上完成，避免不必要的数据传输
- **强制 PyTorch 后端**: 确保与深度学习工作流的无缝集成
- **高性能**: 优化的张量操作，支持批处理和自动微分
- **可扩展**: 模块化设计，易于集成新的表示方法和模型架构

## 📦 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- Git SSH 访问权限（用于安装 ProteinTensor）

### 使用 uv（推荐）

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows

# 安装项目（开发模式）
uv pip install -e .

# 安装开发依赖
uv pip install -e ".[dev]"
```

### 使用 pip

```bash
pip install protrepr
```

**注意**: 项目依赖的 `protein-tensor` 库将从 GitHub 仓库直接安装：
```
git+ssh://git@github.com/a-green-hand-jack/ProteinTensor.git
```

## 🚀 快速开始

```python
from protein_tensor import load_structure
from protrepr import Atom14, Atom37, Frame

# 加载蛋白质结构（强制使用 PyTorch 后端）
protein_pt = load_structure("protein.pdb", backend='torch')

# 方式一：创建 Atom14 表示
atom14 = Atom14.from_protein_tensor(protein_pt)
print(f"Atom14 coordinates shape: {atom14.coords.shape}")  # (num_residues, 14, 3)
print(f"Atom14 mask shape: {atom14.mask.shape}")          # (num_residues, 14)

# 方式二：创建 Atom37 表示
atom37 = Atom37.from_protein_tensor(protein_pt)
print(f"Atom37 coordinates shape: {atom37.coords.shape}")  # (num_residues, 37, 3)
print(f"Atom37 mask shape: {atom37.mask.shape}")          # (num_residues, 37)

# 方式三：创建 Frame 表示
frame = Frame.from_protein_tensor(protein_pt)
print(f"Frame translations shape: {frame.translations.shape}")  # (num_residues, 3)
print(f"Frame rotations shape: {frame.rotations.shape}")        # (num_residues, 3, 3)

# 双向转换：转换回 ProteinTensor
reconstructed_pt = atom14.to_protein_tensor()

# 直接属性访问
backbone_coords = atom37.get_backbone_coords()  # 获取主链原子坐标
ca_coords = frame.translations                  # CA 原子坐标（平移向量）

# 设备管理
atom14_gpu = atom14.to_device(torch.device("cuda"))
atom37_cpu = atom37.to_device(torch.device("cpu"))
```

## 📁 项目结构

```
ProtRepr/
├── src/protrepr/                    # 核心库代码
│   ├── __init__.py                 # API 导出
│   ├── core/                       # 核心数据类
│   │   ├── atom14.py              # Atom14 数据类
│   │   ├── atom37.py              # Atom37 数据类
│   │   └── frame.py               # Frame 数据类
│   ├── representations/           # 转换工具函数
│   │   ├── atom14_converter.py   # Atom14 转换工具
│   │   ├── atom37_converter.py   # Atom37 转换工具
│   │   └── frame_converter.py    # Frame 转换工具
│   └── utils/                     # 工具函数
│       └── geometry.py            # 几何计算工具
├── tests/                         # 测试代码
├── scripts/                       # 可执行脚本
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