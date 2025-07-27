# ProtRepr 文档

欢迎使用 ProtRepr，一个基于 ProteinTensor 的蛋白质表示学习框架！

## 🎯 项目概述

ProtRepr 是一个专注于蛋白质表示学习、结构预测和功能分析的深度学习研究与应用框架。它基于开源库 [ProteinTensor](https://github.com/a-green-hand-jack/ProteinTensor) 进行二次开发，充分利用其蛋白质结构到张量的转换能力。

## 🚀 核心特性

### 三种标准蛋白质表示方法

1. **atom37 表示法** - 基于残基的固定大小重原子表示
2. **atom14 表示法** - 紧凑型原子表示
3. **frame 表示法** - 基于残基的刚体坐标系

### 技术优势

- **PyTorch-Native**: 所有计算直接在 GPU 上完成
- **强制 PyTorch 后端**: 确保与深度学习工作流的无缝集成
- **高性能**: 优化的张量操作，支持批处理和自动微分
- **可扩展**: 模块化设计，易于集成新的表示方法

## 📖 文档导航

- [使用指南](usage.md) - 详细的使用说明和API文档
- [开发指南](development.md) - 开发环境设置和贡献指南
- [API 参考](api/index.md) - 完整的API文档

## 🚀 快速开始

### 安装

```bash
# 使用 uv（推荐）
uv venv
source .venv/bin/activate
uv pip install -e .

# 或使用 pip
pip install protrepr
```

### 基本使用

```python
from protein_tensor import load_structure
from protrepr import EnhancedProteinTensor

# 加载蛋白质结构（强制使用 PyTorch 后端）
protein_pt = load_structure("protein.pdb", backend='torch')

# 创建增强的蛋白质张量对象
enhanced_protein = EnhancedProteinTensor(protein_pt)

# 生成不同的表示方法
atom37_coords, atom37_mask = enhanced_protein.to_atom37()
atom14_coords, atom14_mask = enhanced_protein.to_atom14()
translations, rotations = enhanced_protein.to_frames()
```

## 🔬 应用场景

- **结构预测**: 为蛋白质折叠预测模型提供标准化输入
- **功能分析**: 支持蛋白质功能预测和分类任务
- **药物设计**: 用于分子对接和药物发现研究
- **进化分析**: 支持蛋白质进化和比较研究

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](../LICENSE) 文件。

## 🙏 致谢

本项目基于 [ProteinTensor](https://github.com/a-green-hand-jack/ProteinTensor) 开发，感谢原作者的杰出工作。 