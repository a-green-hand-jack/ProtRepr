"""
pytest 配置文件和共享 fixtures

为 ProtRepr 测试套件提供共享的 fixtures 和配置。
支持新的数据类架构：Atom14, Atom37, Frame。
"""

import pytest
import torch
from typing import Tuple, Dict, Any, List
from pathlib import Path
import tempfile
import numpy as np

# TODO: 在实现相应模块后取消注释
# from protrepr import Atom14, Atom37, Frame


def pytest_configure(config):
    """
    pytest 配置钩子，在测试开始前执行全局配置。
    """
    # 设置 PyTorch 默认设置
    torch.set_default_dtype(torch.float32)
    
    # 设置随机种子确保测试的可重现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置 PyTorch 计算模式
    torch.set_grad_enabled(False)  # 测试中通常不需要梯度


@pytest.fixture
def device() -> torch.device:
    """提供计算设备 fixture。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture 
def sample_protein_data(device: torch.device) -> Dict[str, Any]:
    """
    提供示例蛋白质数据的 fixture。
    
    Returns:
        包含坐标、原子类型、残基信息等的字典
    """
    num_residues = 10
    num_atoms = 50  # 总原子数
    
    # 生成示例坐标数据
    coordinates = torch.randn(num_atoms, 3, device=device) * 10.0
    
    # 原子类型（避免同一残基内重复的原子类型）
    atom_types = []
    atoms_per_residue = num_atoms // num_residues
    
    for res_idx in range(num_residues):
        # 为每个残基分配不重复的原子类型
        # 主要使用前几个常见原子类型（N, CA, C, O, CB等）
        residue_atom_types = list(range(min(atoms_per_residue, 15)))  # 使用前15个原子类型
        # 如果需要更多原子，随机填充其他类型
        while len(residue_atom_types) < atoms_per_residue:
            extra_type = int(torch.randint(15, 37, (1,)).item())
            if extra_type not in residue_atom_types:
                residue_atom_types.append(extra_type)
        
        atom_types.extend(residue_atom_types[:atoms_per_residue])
    
    atom_types = torch.tensor(atom_types, device=device, dtype=torch.long)
    
    # 残基索引（每个原子属于哪个残基）
    residue_indices = torch.repeat_interleave(
        torch.arange(num_residues, device=device), 
        num_atoms // num_residues
    )
    
    # 残基类型（氨基酸类型）- 每个原子对应一个残基类型
    residue_types_per_residue = torch.randint(0, 20, (num_residues,), device=device)
    residue_types = torch.repeat_interleave(
        residue_types_per_residue, 
        num_atoms // num_residues
    )
    
    # 链标识符 - 每个原子对应一个链ID
    chain_ids = torch.zeros(num_atoms, device=device, dtype=torch.long)
    
    return {
        "coordinates": coordinates,
        "atom_types": atom_types,
        "residue_indices": residue_indices,
        "residue_types": residue_types,
        "chain_ids": chain_ids,
        "num_residues": num_residues,
        "num_atoms": num_atoms,
    }


@pytest.fixture
def sample_residue_names() -> List[str]:
    """提供示例残基名称列表。"""
    return ["ALA", "GLY", "SER", "THR", "VAL", "LEU", "ILE", "MET", "PHE", "TYR"]


@pytest.fixture
def sample_atom14_data(device: torch.device, sample_residue_names: List[str]) -> Dict[str, Any]:
    """
    提供示例 Atom14 数据的 fixture。
    
    Returns:
        包含新 Atom14 API 所需数据的字典
    """
    num_residues = len(sample_residue_names)
    
    # Atom14 坐标 (num_residues, 14, 3)
    coords = torch.randn(num_residues, 14, 3, device=device) * 10.0
    
    # 分离的掩码
    atom_mask = torch.rand(num_residues, 14, device=device) > 0.3  # 70% 的原子存在
    res_mask = torch.ones(num_residues, device=device, dtype=torch.bool)  # 所有残基都是标准残基
    
    # 链和残基信息
    chain_ids = torch.zeros(num_residues, device=device, dtype=torch.long)
    residue_types = torch.arange(num_residues, device=device) % 20  # 限制在 0-19 范围内
    residue_indices = torch.arange(num_residues, device=device)  # 全局残基编号
    chain_residue_indices = torch.arange(num_residues, device=device)  # 链内局部编号
    
    # 张量化的名称（整数编码）
    residue_names_tensor = torch.arange(num_residues, device=device) % 20  # 对应 residue_types
    atom_names_tensor = torch.arange(14, device=device, dtype=torch.long)  # 0-13 对应 atom14 标准
    
    return {
        "coords": coords,
        "atom_mask": atom_mask,
        "res_mask": res_mask,
        "chain_ids": chain_ids,
        "residue_types": residue_types,
        "residue_indices": residue_indices,
        "chain_residue_indices": chain_residue_indices,
        "residue_names": residue_names_tensor,
        "atom_names": atom_names_tensor,
    }


@pytest.fixture
def sample_atom37_data(device: torch.device, sample_residue_names: List[str]) -> Dict[str, Any]:
    """
    提供示例 Atom37 数据的 fixture。
    
    Returns:
        包含 Atom37 表示所需数据的字典
    """
    num_residues = len(sample_residue_names)
    
    # Atom37 坐标 (num_residues, 37, 3)
    coords = torch.randn(num_residues, 37, 3, device=device) * 10.0
    
    # Atom37 掩码 (num_residues, 37)
    mask = torch.rand(num_residues, 37, device=device) > 0.4  # 60% 的原子存在
    
    # 链和残基信息
    chain_ids = torch.zeros(num_residues, device=device, dtype=torch.long)
    residue_types = torch.arange(num_residues, device=device)
    residue_indices = torch.arange(num_residues, device=device)
    
    # 原子名称（37个标准原子）
    atom_names = [
        "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "CD", "CD1", 
        "CD2", "CE", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2", 
        "NZ", "NH1", "NH2", "ND1", "ND2", "NE", "NE1", "NE2", "OE1", 
        "OE2", "OG", "OG1", "OH", "SD", "SG", "OD1", "OD2", "OXT"
    ]
    
    return {
        "coords": coords,
        "mask": mask,
        "chain_ids": chain_ids,
        "residue_types": residue_types,
        "residue_indices": residue_indices,
        "residue_names": sample_residue_names,
        "atom_names": atom_names,
    }


@pytest.fixture
def sample_frame_data(device: torch.device, sample_residue_names: List[str]) -> Dict[str, Any]:
    """
    提供示例 Frame 数据的 fixture。
    
    Returns:
        包含 Frame 表示所需数据的字典
    """
    num_residues = len(sample_residue_names)
    
    # 平移向量 (num_residues, 3)
    translations = torch.randn(num_residues, 3, device=device) * 10.0
    
    # 生成有效的旋转矩阵 (num_residues, 3, 3)
    # 使用 QR 分解确保正交性
    random_matrices = torch.randn(num_residues, 3, 3, device=device)
    Q, R = torch.linalg.qr(random_matrices)
    
    # 确保行列式为 +1 (右手坐标系)
    det = torch.det(Q)
    Q[det < 0] *= -1
    rotations = Q
    
    # 链和残基信息
    chain_ids = torch.zeros(num_residues, device=device, dtype=torch.long)
    residue_types = torch.arange(num_residues, device=device)
    residue_indices = torch.arange(num_residues, device=device)
    
    return {
        "translations": translations,
        "rotations": rotations,
        "chain_ids": chain_ids,
        "residue_types": residue_types,
        "residue_indices": residue_indices,
        "residue_names": sample_residue_names,
    }


@pytest.fixture
def mock_protein_tensor(sample_protein_data: Dict[str, torch.Tensor]):
    """
    提供模拟的 ProteinTensor 对象。
    
    注意：这是一个简化的模拟，实际实现时需要替换为真实的 ProteinTensor
    """
    class MockProteinTensor:
        def __init__(self, data: Dict[str, torch.Tensor]):
            self.coordinates = data["coordinates"]
            self.atom_types = data["atom_types"]
            self.residue_indices = data["residue_indices"]
            self.residue_types = data["residue_types"]
            self.chain_ids = data["chain_ids"]
            self.num_residues = data["num_residues"]
            self.num_atoms = data["num_atoms"]
            # 添加 residue_numbers 属性（通常与 residue_indices 相同）
            self.residue_numbers = data["residue_indices"]
            # 添加兼容性属性
            self.n_atoms = data["num_atoms"]
            self.n_residues = data["num_residues"]
        
        def to_torch(self):
            """转换为 torch 格式的数据字典"""
            return {
                'coordinates': self.coordinates,
                'atom_types': self.atom_types,
                'residue_types': self.residue_types,
                'chain_ids': self.chain_ids,
                'residue_numbers': self.residue_numbers
            }
        
        def _tensor_to_numpy(self, tensor):
            """将张量转换为 numpy 数组（模拟 ProteinTensor 的方法）"""
            if hasattr(tensor, 'cpu'):
                return tensor.cpu().numpy()
            return tensor
    
    return MockProteinTensor(sample_protein_data)


@pytest.fixture
def test_data_dir() -> Path:
    """提供测试数据目录的路径。"""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_output_dir():
    """提供临时输出目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_backbone_coords(device: torch.device) -> torch.Tensor:
    """
    提供示例主链原子坐标 (N, CA, C, O)。
    
    Returns:
        形状为 (num_residues, 4, 3) 的主链原子坐标
    """
    num_residues = 5
    
    # 生成合理的主链几何结构
    backbone_coords = torch.zeros(num_residues, 4, 3, device=device)
    
    for i in range(num_residues):
        # CA 原子位置
        ca_pos = torch.tensor([i * 3.8, 0.0, 0.0], device=device)
        
        # N 原子位置（相对于CA）
        n_pos = ca_pos + torch.tensor([-1.46, 0.0, 0.0], device=device)
        
        # C 原子位置（相对于CA）
        c_pos = ca_pos + torch.tensor([1.53, 0.0, 0.0], device=device)
        
        # O 原子位置（相对于C）
        o_pos = c_pos + torch.tensor([1.23, 1.0, 0.0], device=device)
        
        backbone_coords[i] = torch.stack([n_pos, ca_pos, c_pos, o_pos])
    
    return backbone_coords


@pytest.fixture(autouse=True)
def reset_torch_state():
    """
    自动 fixture，在每个测试后重置 PyTorch 状态。
    """
    yield
    # 测试后清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 