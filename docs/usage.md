# 使用指南

本指南将详细介绍如何使用 ProtRepr 框架进行蛋白质表示学习。

## 📦 安装指南

### 系统要求

- Python 3.8 或更高版本
- PyTorch 2.0 或更高版本
- Git SSH 访问权限（用于安装 ProteinTensor）
- CUDA（可选，用于 GPU 加速）

### 使用 uv 安装（推荐）

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows

# 安装项目
uv pip install -e .

# 安装开发依赖（用于开发和测试）
uv pip install -e ".[dev]"
```

### 使用 pip 安装

```bash
# 从 PyPI 安装
pip install protrepr

# 或从源码安装
git clone https://github.com/your-org/protrepr.git
cd protrepr
pip install -e .
```

**注意**: 项目依赖的 `protein-tensor` 库将从 GitHub 仓库直接安装：
```
git+ssh://git@github.com/a-green-hand-jack/ProteinTensor.git
```

## 🚀 基本使用

### 1. 加载蛋白质结构

```python
from protein_tensor import load_structure
from protrepr import Atom14, Atom37, Frame
import torch

# 加载 PDB 文件（强制使用 PyTorch 后端）
protein_pt = load_structure("examples/protein.pdb", backend='torch')

# 验证使用了正确的后端
assert isinstance(protein_pt.coordinates, torch.Tensor)
```

### 2. 创建和使用 Atom14 表示

```python
# 从 ProteinTensor 创建 Atom14 实例
atom14 = Atom14.from_protein_tensor(protein_pt)

print(f"Atom14 坐标形状: {atom14.coords.shape}")     # (num_residues, 14, 3)
print(f"Atom14 掩码形状: {atom14.mask.shape}")       # (num_residues, 14)
print(f"残基数量: {atom14.num_residues}")
print(f"链数量: {atom14.num_chains}")

# 检查真实原子的数量
num_real_atoms = atom14.mask.sum().item()
print(f"真实原子数量: {num_real_atoms}")

# 访问元数据
print(f"残基名称: {atom14.residue_names[:5]}")  # 前5个残基
print(f"原子名称: {atom14.atom_names}")         # 14个标准原子名称

# 获取主链和侧链坐标
backbone_coords = atom14.get_backbone_coords()  # (num_residues, 4, 3)
sidechain_coords = atom14.get_sidechain_coords()  # (num_residues, 10, 3)

# 双向转换
reconstructed_pt = atom14.to_protein_tensor()
```

### 3. 创建和使用 Atom37 表示

```python
# 从 ProteinTensor 创建 Atom37 实例
atom37 = Atom37.from_protein_tensor(protein_pt)

print(f"Atom37 坐标形状: {atom37.coords.shape}")     # (num_residues, 37, 3)
print(f"Atom37 掩码形状: {atom37.mask.shape}")       # (num_residues, 37)

# Atom37 比 Atom14 包含更多原子信息
print(f"内存使用比较: Atom37 vs Atom14 = {37/14:.2f}x")

# 获取特定残基的原子信息
residue_atoms = atom37.get_residue_atoms(0)  # 第一个残基
ca_coord = residue_atoms['CA']  # 获取CA原子坐标

# 计算残基质心
center_of_mass = atom37.compute_center_of_mass()  # (num_residues, 3)

# 获取主链和侧链坐标
backbone_coords = atom37.get_backbone_coords()    # (num_residues, 4, 3)
sidechain_coords = atom37.get_sidechain_coords()  # (num_residues, 33, 3)
```

### 4. 创建和使用 Frame 表示

```python
# 从 ProteinTensor 创建 Frame 实例
frame = Frame.from_protein_tensor(protein_pt)

print(f"平移向量形状: {frame.translations.shape}")  # (num_residues, 3)
print(f"旋转矩阵形状: {frame.rotations.shape}")     # (num_residues, 3, 3)

# 验证旋转矩阵的有效性
det = torch.det(frame.rotations)
print(f"旋转矩阵行列式（应接近1）: {det.mean():.4f}")

# 应用刚体变换到坐标
sample_coords = torch.randn(frame.num_residues, 10, 3)  # 示例坐标
transformed_coords = frame.apply_transform(sample_coords)

# 计算相对变换
relative_transforms = frame.compute_relative_frames()

# Frame 间的插值
frame2 = Frame.from_protein_tensor(protein_pt)  # 另一个Frame
alpha = 0.5  # 插值系数
interpolated_frame = frame.interpolate_frames(frame2, alpha)

# 重建主链坐标
backbone_coords = frame.compute_backbone_coords()  # (num_residues, 4, 3)
```

## 🔧 高级用法

### 1. 设备管理

```python
# 检查可用设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建数据类后移动到指定设备
atom14 = Atom14.from_protein_tensor(protein_pt)
atom37 = Atom37.from_protein_tensor(protein_pt)
frame = Frame.from_protein_tensor(protein_pt)

# 移动到 GPU
atom14_gpu = atom14.to_device(device)
atom37_gpu = atom37.to_device(device)
frame_gpu = frame.to_device(device)

print(f"Atom14 设备: {atom14_gpu.device}")
print(f"Atom37 设备: {atom37_gpu.device}")
print(f"Frame 设备: {frame_gpu.device}")

# 移动回 CPU
atom14_cpu = atom14_gpu.to_device(torch.device("cpu"))
```

### 2. 批量处理

```python
# 处理多个蛋白质结构
pdb_files = ["protein1.pdb", "protein2.pdb", "protein3.pdb"]
batch_atom37 = []
batch_frames = []

for pdb_file in pdb_files:
    protein_pt = load_structure(pdb_file, backend='torch')
    
    # 创建不同的表示
    atom37 = Atom37.from_protein_tensor(protein_pt)
    frame = Frame.from_protein_tensor(protein_pt)
    
    batch_atom37.append(atom37)
    batch_frames.append(frame)

# 批量设备管理
gpu_batch = [atom37.to_device(device) for atom37 in batch_atom37]

# 批量坐标提取（需要相同的残基数量）
if all(atom37.num_residues == batch_atom37[0].num_residues for atom37 in batch_atom37):
    batch_coords = torch.stack([atom37.coords for atom37 in batch_atom37])
    batch_masks = torch.stack([atom37.mask for atom37 in batch_atom37])
    print(f"批量坐标形状: {batch_coords.shape}")  # (batch_size, num_residues, 37, 3)
```

### 3. 与深度学习模型集成

```python
import torch.nn as nn

class ProteinClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # 使用 Atom37 表示法作为输入
        self.atom_embedding = nn.Linear(37 * 3, 512)
        self.residue_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=6
        )
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, atom37: Atom37):
        # 展平坐标 (batch, num_residues, 37, 3) -> (batch, num_residues, 37*3)
        coords = atom37.coords
        mask = atom37.mask
        
        flattened = coords.flatten(start_dim=-2)  # (batch, num_residues, 111)
        
        # 嵌入每个残基
        embedded = self.atom_embedding(flattened)  # (batch, num_residues, 512)
        
        # Transformer 编码
        # 注意：Transformer 期望 (seq_len, batch, feature)
        embedded = embedded.transpose(0, 1)  # (num_residues, batch, 512)
        encoded = self.residue_encoder(embedded)  # (num_residues, batch, 512)
        
        # 全局池化
        pooled = encoded.mean(dim=0)  # (batch, 512)
        
        # 分类
        logits = self.classifier(pooled)  # (batch, num_classes)
        return logits

# 使用示例
model = ProteinClassifier(num_classes=10)
atom37 = Atom37.from_protein_tensor(protein_pt)

# 添加批次维度
atom37_batch = atom37.coords.unsqueeze(0)  # 模拟批次
atom37.coords = atom37_batch
atom37.mask = atom37.mask.unsqueeze(0)

# 前向传播
logits = model(atom37)
```

### 4. SE(3)-Equivariant 网络集成

```python
class SE3EquivariantModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 Frame 表示进行 SE(3) 等变处理
        
    def forward(self, frame: Frame):
        # 使用旋转和平移信息
        translations = frame.translations  # (num_residues, 3)
        rotations = frame.rotations        # (num_residues, 3, 3)
        
        # SE(3)-equivariant 处理
        # ...
        
        return processed_frame

# Frame 表示特别适合 SE(3)-equivariant 网络
frame = Frame.from_protein_tensor(protein_pt)
model = SE3EquivariantModel()
result = model(frame)
        
        # 应用掩码并池化
        masked_embedded = embedded * atom37_mask.unsqueeze(-1)
        pooled = masked_embedded.mean(dim=1)  # 全局平均池化
        
        return self.classifier(pooled)

# 使用示例
model = ProteinClassifier(num_classes=10)
atom37_coords, atom37_mask = enhanced_protein.to_atom37()

# 添加批次维度
coords_batch = atom37_coords.unsqueeze(0)
mask_batch = atom37_mask.unsqueeze(0)

# 前向传播
predictions = model(coords_batch, mask_batch)
```

## 🛠️ 工具函数

### 几何计算

```python
from protrepr.utils.geometry import (
    compute_distances, 
    compute_bond_angles, 
    compute_dihedral_angles
)

# 计算原子间距离
coords1 = torch.randn(3)  # 第一个原子
coords2 = torch.randn(3)  # 第二个原子
distance = compute_distances(coords1, coords2)

# 计算键角
coords3 = torch.randn(3)  # 第三个原子
bond_angle = compute_bond_angles(coords1, coords2, coords3)

# 计算二面角
coords4 = torch.randn(3)  # 第四个原子
dihedral_angle = compute_dihedral_angles(coords1, coords2, coords3, coords4)
```

## 📊 性能优化

### 1. 内存优化

```python
# 对于大型蛋白质，选择合适的表示方法
num_residues = enhanced_protein.num_residues

if num_residues > 1000:
    # 大型蛋白质使用 atom14 节省内存
    coords, mask = enhanced_protein.to_atom14()
else:
    # 小型蛋白质使用 atom37 获得更多信息
    coords, mask = enhanced_protein.to_atom37()
```

### 2. GPU 加速

```python
# 确保所有计算在 GPU 上进行
if torch.cuda.is_available():
    # ProtRepr 会自动利用 GPU 加速
    # 无需手动转移张量
    pass
```

## ❗ 常见问题

### Q: 为什么必须使用 PyTorch 后端？

A: ProtRepr 专为深度学习工作流设计，强制使用 PyTorch 后端可以：
- 避免不必要的 CPU/GPU 数据传输
- 确保所有操作支持自动微分
- 提供最佳的性能表现

### Q: 如何处理缺失原子？

A: ProtRepr 通过掩码机制处理缺失原子：
- atom37/atom14 表示法中，缺失原子用零向量填充
- 相应的掩码位置标记为 `False`
- 在模型中使用掩码来忽略这些位置

### Q: 支持哪些蛋白质文件格式？

A: ProtRepr 支持 ProteinTensor 支持的所有格式：
- PDB 文件 (.pdb)
- mmCIF 文件 (.cif)
- 其他 ProteinTensor 支持的格式

## 📚 更多资源

- [API 参考文档](api/index.md)
- [开发指南](development.md)
- [示例代码](../scripts/example_usage.py)
- [GitHub 仓库](https://github.com/your-org/protrepr) 