"""
基础转换器模块

本模块包含所有蛋白质表示转换器的共同功能，包括：
- 共同常量定义
- 张量化名称处理函数
- 向量化辅助函数
- 临时 ProteinTensor 类
- 抽象基类定义
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import torch
from protein_tensor import ProteinTensor, save_structure

logger = logging.getLogger(__name__)

# ================================
# 共同常量定义
# ================================

# 残基名称到索引的映射
RESIDUE_NAME_TO_IDX = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19
}

# 索引到残基名称的映射
IDX_TO_RESIDUE_NAME = {v: k for k, v in RESIDUE_NAME_TO_IDX.items()}

# 标准键长和键角常量
STANDARD_BOND_LENGTHS = {
    "CA_CB": 1.526,  # CA-CB 键长 (Å)
    "CA_N": 1.458,   # CA-N 键长 (Å) 
    "CA_C": 1.525,   # CA-C 键长 (Å)
}

STANDARD_BOND_ANGLES = {
    "N_CA_CB": 110.5,  # N-CA-CB 键角 (度)
    "C_CA_CB": 110.1,  # C-CA-CB 键角 (度)
}

# 链间间隔设置
CHAIN_GAP = 200  # 不同链之间的残基编号间隔

# ================================
# 共同张量化名称处理函数
# ================================

def create_residue_name_tensor(residue_names: List[str], device: torch.device) -> torch.Tensor:
    """
    将残基名称列表转换为张量（整数编码）。
    
    Args:
        residue_names: 残基名称列表
        device: 目标设备
        
    Returns:
        torch.Tensor: 编码后的残基名称张量
    """
    residue_indices = []
    for name in residue_names:
        if name in RESIDUE_NAME_TO_IDX:
            residue_indices.append(RESIDUE_NAME_TO_IDX[name])
        else:
            # 静默处理未知残基，使用 UNK (20)
            residue_indices.append(20)  # UNK 残基
    
    return torch.tensor(residue_indices, dtype=torch.long, device=device)


def decode_residue_names(residue_tensor: torch.Tensor) -> List[str]:
    """
    将残基名称张量解码为字符串列表。
    
    Args:
        residue_tensor: 编码的残基名称张量
        
    Returns:
        List[str]: 解码后的残基名称列表
    """
    names = []
    for idx in residue_tensor.cpu().numpy():
        if idx in IDX_TO_RESIDUE_NAME:
            names.append(IDX_TO_RESIDUE_NAME[idx])
        else:
            names.append("UNK")
    return names


# ================================
# 共同向量化辅助函数
# ================================

def find_residue_boundaries_vectorized(chain_ids: torch.Tensor, residue_numbers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用向量化操作找到每个残基的边界。
    
    Args:
        chain_ids: 链ID张量 (num_atoms,)
        residue_numbers: 残基编号张量 (num_atoms,)
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (残基起始索引, 残基结束索引)
    """
    num_atoms = len(chain_ids)
    device = chain_ids.device
    
    # 创建残基唯一标识符
    # 使用高位存放chain_id，低位存放residue_number
    max_residue_num = residue_numbers.max().item() + 1
    residue_ids = chain_ids * max_residue_num + residue_numbers
    
    # 找到残基变化的位置
    # 在开头添加一个不同的值，确保第一个残基被检测到
    padded_ids = torch.cat([residue_ids[:1] - 1, residue_ids])
    changes = (padded_ids[1:] != padded_ids[:-1])
    
    # 残基起始位置
    residue_starts = torch.nonzero(changes, as_tuple=True)[0]
    
    # 残基结束位置（下一个残基的开始位置）
    residue_ends = torch.cat([residue_starts[1:], torch.tensor([num_atoms], device=device)])
    
    return residue_starts, residue_ends


def compute_chain_info_vectorized(chain_ids: torch.Tensor, residue_starts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    使用向量化操作计算链信息和残基编号。
    
    Args:
        chain_ids: 链ID张量 (num_atoms,)
        residue_starts: 残基起始索引 (num_residues,)
        
    Returns:
        Tuple containing:
            unique_chains: 唯一链ID (num_chains,)
            chain_residue_counts: 每条链的残基数量 (num_chains,)
            residue_chain_ids: 每个残基的链ID (num_residues,)
    """
    device = chain_ids.device
    num_residues = len(residue_starts)
    
    # 获取每个残基的链ID
    residue_chain_ids = chain_ids[residue_starts]
    
    # 获取唯一的链ID（保持顺序）
    unique_chains, inverse_indices = torch.unique(residue_chain_ids, return_inverse=True, sorted=True)
    
    # 计算每条链的残基数量
    chain_residue_counts = torch.bincount(inverse_indices)
    
    return unique_chains, chain_residue_counts, residue_chain_ids


def compute_global_residue_indices_vectorized(
    residue_chain_ids: torch.Tensor,
    unique_chains: torch.Tensor,
    chain_residue_counts: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用向量化操作计算全局残基编号（包含链间gap）。
    
    Args:
        residue_chain_ids: 每个残基的链ID (num_residues,)
        unique_chains: 唯一链ID (num_chains,)
        chain_residue_counts: 每条链的残基数量 (num_chains,)
        
    Returns:
        Tuple containing:
            global_residue_indices: 全局残基编号 (num_residues,)
            chain_residue_indices: 链内残基编号 (num_residues,)
    """
    device = residue_chain_ids.device
    num_residues = len(residue_chain_ids)
    num_chains = len(unique_chains)
    
    # 计算每条链的全局起始编号
    chain_start_indices = torch.zeros(num_chains, device=device, dtype=torch.long)
    
    current_start = 1  # 从1开始编号
    for i in range(num_chains):
        chain_start_indices[i] = current_start
        if i < num_chains - 1:  # 不是最后一条链
            current_start += chain_residue_counts[i] + CHAIN_GAP
    
    # 为每个残基分配全局编号
    global_residue_indices = torch.zeros(num_residues, device=device, dtype=torch.long)
    chain_residue_indices = torch.zeros(num_residues, device=device, dtype=torch.long)
    
    # 为每条链单独处理
    for chain_idx, chain_id in enumerate(unique_chains.tolist()):
        chain_mask = (residue_chain_ids == chain_id)
        chain_residue_count = chain_residue_counts[chain_idx].item()
        start_index = chain_start_indices[chain_idx].item()
        
        # 生成这条链的全局编号和链内编号
        chain_global_indices = torch.arange(
            start_index, start_index + chain_residue_count, 
            device=device, dtype=torch.long
        )
        chain_local_indices = torch.arange(
            chain_residue_count, device=device, dtype=torch.long
        )
        
        # 分配到对应位置
        global_residue_indices[chain_mask] = chain_global_indices
        chain_residue_indices[chain_mask] = chain_local_indices
    
    return global_residue_indices, chain_residue_indices


# ================================
# 临时 ProteinTensor 类
# ================================

class TempProteinTensor:
    """
    临时 ProteinTensor 对象，用于转换回 ProteinTensor 格式。
    """
    
    def __init__(self, coords, atom_types, residue_types, chain_ids, residue_numbers):
        self.coordinates = coords
        self.atom_types = atom_types
        self.residue_types = residue_types
        self.chain_ids = chain_ids
        self.residue_numbers = residue_numbers
        self.n_atoms = len(coords)
        self.n_residues = len(set((c, r) for c, r in zip(chain_ids, residue_numbers)))
    
    def _tensor_to_numpy(self, tensor):
        """将张量转换为 numpy 数组（模拟 ProteinTensor 的方法）"""
        if hasattr(tensor, 'cpu'):
            return tensor.cpu().numpy()
        return tensor
    
    def save_structure(self, output_path: str, format_type: str = "cif"):
        """保存结构到文件"""
        save_structure(self, output_path, format_type=format_type)  # type: ignore


# ================================
# 共同工具函数
# ================================

def save_protein_tensor_to_cif(
    protein_tensor: ProteinTensor,
    output_path: Union[str, Path],
    title: str = "ProtRepr Reconstructed Structure"
) -> None:
    """
    将 ProteinTensor 数据保存为 CIF 文件。
    
    Args:
        protein_tensor: ProteinTensor 实例或兼容对象
        output_path: 输出文件路径  
        title: 结构标题
    """
    logger.info(f"将 ProteinTensor 数据保存到 CIF 文件: {output_path}")
    save_structure(protein_tensor, output_path, format_type="cif")
    logger.info(f"CIF 文件保存成功: {output_path}")


# ================================
# 抽象基类定义
# ================================

class BaseProteinConverter(ABC):
    """
    蛋白质表示转换器的抽象基类。
    
    定义了所有转换器必须实现的接口。
    """
    
    @abstractmethod
    def protein_tensor_to_representation(
        self, 
        protein_tensor: ProteinTensor, 
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        将 ProteinTensor 转换为特定表示格式。
        
        Args:
            protein_tensor: 输入的 ProteinTensor 对象
            device: 目标设备
            
        Returns:
            Tuple: 转换后的表示数据
        """
        pass
    
    @abstractmethod
    def representation_to_protein_tensor(self, *args) -> Any:
        """
        将特定表示格式转换为 ProteinTensor。
        
        Args:
            *args: 表示数据的各个组件
            
        Returns:
            ProteinTensor: 转换后的 ProteinTensor 对象
        """
        pass
    
    @abstractmethod
    def validate_representation_data(self, *args) -> None:
        """
        验证表示数据的一致性和有效性。
        
        Args:
            *args: 表示数据的各个组件
            
        Raises:
            ValueError: 当数据不一致或无效时
        """
        pass
    
    @abstractmethod
    def save_representation_to_cif(
        self, 
        representation: Any, 
        output_path: Union[str, Path], 
        title: str = "ProtRepr Structure"
    ) -> None:
        """
        将表示数据保存为 CIF 文件。
        
        Args:
            representation: 表示实例
            output_path: 输出文件路径
            title: 结构标题
        """
        pass


# ================================
# 特定表示的原子名称处理函数
# ================================

def create_atom_name_tensor(num_atoms: int, device: torch.device) -> torch.Tensor:
    """
    创建原子名称张量（整数编码）。
    
    Args:
        num_atoms: 原子数量
        device: 目标设备
        
    Returns:
        torch.Tensor: 编码后的原子名称张量
    """
    return torch.arange(num_atoms, dtype=torch.long, device=device)


def decode_atom_names(atom_tensor: torch.Tensor, idx_to_atom_name: Dict[int, str]) -> List[str]:
    """
    将原子名称张量解码为字符串列表。
    
    Args:
        atom_tensor: 编码的原子名称张量
        idx_to_atom_name: 索引到原子名称的映射
        
    Returns:
        List[str]: 解码后的原子名称列表
    """
    names = []
    for idx in atom_tensor.cpu().numpy():
        if idx in idx_to_atom_name:
            names.append(idx_to_atom_name[idx])
        else:
            names.append(f"UNK{idx}")
    return names 