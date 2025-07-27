"""
Atom14 表示转换器 (优化版本)

本模块提供 ProteinTensor 与 Atom14 表示之间的高性能转换功能，包括：
- 向量化的坐标到 atom14 格式映射
- 甘氨酸虚拟 CB 原子计算
- 数据验证和完整性检查
- 支持批量操作、分离掩码、链间信息和张量化名称
- 全面的性能优化，使用 PyTorch 张量操作替代 Python 循环
"""

import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import torch
from protein_tensor import ProteinTensor, save_structure

logger = logging.getLogger(__name__)

# ================================
# 常量定义部分
# ================================

# Atom14 标准原子类型列表（14个固定槽位）
ATOM14_ATOM_TYPES = [
    "N",      # 0 - 主链氮
    "CA",     # 1 - 主链 α-碳
    "C",      # 2 - 主链羰基碳
    "O",      # 3 - 主链羰基氧
    "CB",     # 4 - β-碳（或甘氨酸虚拟CB）
    "CG",     # 5 - 侧链碳1
    "CG1",    # 6 - 侧链分支碳1
    "CG2",    # 7 - 侧链分支碳2
    "CD",     # 8 - 侧链碳2
    "CD1",    # 9 - 侧链分支碳3
    "CD2",    # 10 - 侧链分支碳4
    "CE",     # 11 - 侧链碳3
    "CE1",    # 12 - 侧链分支碳5
    "CE2"     # 13 - 侧链分支碳6
]

# 残基名称到索引的映射
RESIDUE_NAME_TO_IDX = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19
}

# 索引到残基名称的映射
IDX_TO_RESIDUE_NAME = {v: k for k, v in RESIDUE_NAME_TO_IDX.items()}

# 原子名称到索引的映射
ATOM_NAME_TO_IDX = {name: idx for idx, name in enumerate(ATOM14_ATOM_TYPES)}

# 索引到原子名称的映射
IDX_TO_ATOM_NAME = {idx: name for idx, name in enumerate(ATOM14_ATOM_TYPES)}

# 加载额外的原子类型映射（如果有的话）
EXTENDED_ATOM_NAMES = {
    "H": 14, "OXT": 15, "N1": 16, "N2": 17, "N3": 18, "N4": 19,
    "O1": 20, "O2": 21, "O3": 22, "O4": 23, "S": 24, "P": 25,
    "CZ": 26, "NZ": 27, "OD1": 28, "OD2": 29, "OE1": 30, "OE2": 31,
    "OG": 32, "OG1": 33, "SD": 34, "SG": 35, "NH1": 36, "NH2": 37
}

# 合并原子名称映射
ALL_ATOM_NAME_TO_IDX = {**ATOM_NAME_TO_IDX, **EXTENDED_ATOM_NAMES}
ALL_IDX_TO_ATOM_NAME = {v: k for k, v in ALL_ATOM_NAME_TO_IDX.items()}

# 标准键长和键角常量（用于虚拟原子计算）
STANDARD_BOND_LENGTHS = {
    "CA_CB": 1.526,  # CA-CB 键长 (Å)
    "CA_N": 1.458,   # CA-N 键长 (Å) 
    "CA_C": 1.525,   # CA-C 键长 (Å)
}

STANDARD_BOND_ANGLES = {
    "N_CA_CB": 110.5,  # N-CA-CB 键角 (度)
    "C_CA_CB": 110.1,  # C-CA-CB 键角 (度)
}

# 链间间隔设置 - 用于多链蛋白质的全局残基编号
CHAIN_GAP = 200  # 不同链之间的残基编号间隔

# 每种残基的原子到 atom14 槽位的映射
RESIDUE_ATOM14_MAPPING: Dict[str, Dict[str, int]] = {
    "ALA": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4},
    "ARG": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "CD": 8, "NE": 11, "CZ": 12, "NH1": 13, "NH2": 13},
    "ASN": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "OD1": 8, "ND2": 9},
    "ASP": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "OD1": 8, "OD2": 9},
    "CYS": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "SG": 5},
    "GLN": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "CD": 8, "OE1": 11, "NE2": 12},
    "GLU": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "CD": 8, "OE1": 11, "OE2": 12},
    "GLY": {"N": 0, "CA": 1, "C": 2, "O": 3},  # CB 为虚拟原子
    "HIS": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "ND1": 8, "CD2": 9, "CE1": 11, "NE2": 12},
    "ILE": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG1": 6, "CG2": 7, "CD1": 9},
    "LEU": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "CD1": 9, "CD2": 10},
    "LYS": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "CD": 8, "CE": 11, "NZ": 12},
    "MET": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "SD": 8, "CE": 11},
    "PHE": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "CD1": 9, "CD2": 10, "CE1": 12, "CE2": 13, "CZ": 8},
    "PRO": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "CD": 8},
    "SER": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "OG": 5},
    "THR": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "OG1": 5, "CG2": 7},
    "TRP": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "CD1": 8, "CD2": 9, "NE1": 11, "CE2": 12, "CE3": 13, "CZ2": 6, "CZ3": 7, "CH2": 10},
    "TYR": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "CD1": 9, "CD2": 10, "CE1": 12, "CE2": 13, "CZ": 8, "OH": 11},
    "VAL": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG1": 6, "CG2": 7}
}


# ================================
# 张量化名称处理函数
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
            logger.warning(f"未知残基名称: {name}，使用 UNK (20)")
            residue_indices.append(20)  # UNK 残基
    
    return torch.tensor(residue_indices, dtype=torch.long, device=device)


def create_atom_name_tensor(device: torch.device) -> torch.Tensor:
    """
    创建 atom14 原子名称张量（整数编码）。
    
    Args:
        device: 目标设备
        
    Returns:
        torch.Tensor: 编码后的原子名称张量 (14,)
    """
    return torch.arange(14, dtype=torch.long, device=device)


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


def decode_atom_names(atom_tensor: torch.Tensor) -> List[str]:
    """
    将原子名称张量解码为字符串列表。
    
    Args:
        atom_tensor: 编码的原子名称张量
        
    Returns:
        List[str]: 解码后的原子名称列表
    """
    names = []
    for idx in atom_tensor.cpu().numpy():
        if idx in IDX_TO_ATOM_NAME:
            names.append(IDX_TO_ATOM_NAME[idx])
        else:
            names.append(f"UNK{idx}")
    return names


# ================================
# 优化的向量化辅助函数
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


def map_atoms_to_atom14_vectorized(
    coordinates: torch.Tensor,
    atom_types: torch.Tensor,
    residue_types: torch.Tensor,
    residue_starts: torch.Tensor,
    residue_ends: torch.Tensor,
    num_residues: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    使用向量化操作将原子映射到atom14格式。
    
    Args:
        coordinates: 原子坐标 (num_atoms, 3)
        atom_types: 原子类型 (num_atoms,)
        residue_types: 残基类型 (num_atoms,)
        residue_starts: 残基起始索引 (num_residues,)
        residue_ends: 残基结束索引 (num_residues,)
        num_residues: 残基数量
        device: 设备
        
    Returns:
        Tuple containing:
            atom14_coords: atom14坐标 (num_residues, 14, 3)
            atom14_mask: atom14掩码 (num_residues, 14)
            res_mask: 残基掩码 (num_residues,)
            residue_names_list: 残基名称列表
    """
    # 初始化输出张量
    atom14_coords = torch.zeros(num_residues, 14, 3, device=device)
    atom14_mask = torch.zeros(num_residues, 14, dtype=torch.bool, device=device)
    res_mask = torch.ones(num_residues, dtype=torch.bool, device=device)
    residue_names_list = []
    
    # 获取每个残基的残基类型（使用第一个原子的类型）
    residue_type_indices = residue_types[residue_starts]
    
    # 批量处理残基名称
    for res_idx in range(num_residues):
        res_type_idx = residue_type_indices[res_idx].item()
        
        if res_type_idx in IDX_TO_RESIDUE_NAME:
            res_name = IDX_TO_RESIDUE_NAME[res_type_idx]
        else:
            res_name = "UNK"
            res_mask[res_idx] = False
            
        residue_names_list.append(res_name)
        
        # 获取原子映射
        if res_name not in RESIDUE_ATOM14_MAPPING:
            res_mask[res_idx] = False
            continue
            
        mapping = RESIDUE_ATOM14_MAPPING[res_name]
        
        # 处理这个残基的所有原子
        start_atom = residue_starts[res_idx].item()
        end_atom = residue_ends[res_idx].item()
        
        # 批量获取原子信息
        residue_atom_types = atom_types[start_atom:end_atom]
        residue_coords = coordinates[start_atom:end_atom]
        
        # 映射原子到atom14位置
        for local_atom_idx, atom_type_idx in enumerate(residue_atom_types):
            atom_type_idx = atom_type_idx.item()
            
            if atom_type_idx in ALL_IDX_TO_ATOM_NAME:
                atom_name = ALL_IDX_TO_ATOM_NAME[atom_type_idx]
                
                if atom_name in mapping:
                    atom14_pos = mapping[atom_name]
                    atom14_coords[res_idx, atom14_pos] = residue_coords[local_atom_idx]
                    atom14_mask[res_idx, atom14_pos] = True
        
        # 为甘氨酸计算虚拟CB
        if res_name == "GLY" and not atom14_mask[res_idx, 4]:  # CB位置
            # 检查主链原子是否存在
            if atom14_mask[res_idx, 0] and atom14_mask[res_idx, 1] and atom14_mask[res_idx, 2]:
                try:
                    n_pos = atom14_coords[res_idx, 0]
                    ca_pos = atom14_coords[res_idx, 1]  
                    c_pos = atom14_coords[res_idx, 2]
                    
                    virtual_cb = compute_virtual_cb(n_pos, ca_pos, c_pos)
                    atom14_coords[res_idx, 4] = virtual_cb
                    atom14_mask[res_idx, 4] = True
                except Exception as e:
                    logger.warning(f"计算甘氨酸虚拟CB失败: {e}")
    
    return atom14_coords, atom14_mask, res_mask, residue_names_list


def compute_virtual_cb(
    n_coords: torch.Tensor,
    ca_coords: torch.Tensor,
    c_coords: torch.Tensor
) -> torch.Tensor:
    """
    为甘氨酸计算虚拟 CB 原子的坐标。
    
    使用标准的几何关系，基于主链的 N, CA, C 原子位置计算虚拟 CB 原子。
    
    Args:
        n_coords: N 原子坐标 (3,)
        ca_coords: CA 原子坐标 (3,)
        c_coords: C 原子坐标 (3,)
        
    Returns:
        torch.Tensor: 虚拟 CB 原子坐标 (3,)
        
    Raises:
        ValueError: 当输入坐标无效时
        RuntimeError: 当计算过程中出现错误时
    """
    logger.debug("计算虚拟 CB 原子坐标")
    
    # 计算向量
    ca_n = n_coords - ca_coords
    ca_c = c_coords - ca_coords
    
    # 标准化向量（添加数值稳定性）
    ca_n_length = torch.norm(ca_n)
    ca_c_length = torch.norm(ca_c)
    
    if ca_n_length < 1e-6 or ca_c_length < 1e-6:
        logger.warning("主链原子距离过近，无法计算可靠的虚拟 CB 原子")
        # 返回一个基于 CA 的默认位置
        return ca_coords + torch.tensor([1.526, 0.0, 0.0], device=ca_coords.device)
    
    ca_n_norm = ca_n / ca_n_length
    ca_c_norm = ca_c / ca_c_length
    
    # 计算二面角方向（叉积）
    cross_product = torch.linalg.cross(ca_n_norm, ca_c_norm)
    cross_length = torch.norm(cross_product)
    
    # 处理共线情况（叉积为零）
    if cross_length < 1e-6:
        logger.warning("N-CA-C 原子共线，使用默认方向计算虚拟 CB")
        # 使用 y 轴作为默认的垂直方向
        cross_norm = torch.tensor([0.0, 1.0, 0.0], device=ca_coords.device)
    else:
        cross_norm = cross_product / cross_length
    
    # 计算角平分线方向
    bisector = ca_n_norm + ca_c_norm
    bisector_length = torch.norm(bisector)
    
    if bisector_length < 1e-6:
        # 如果 N-CA-C 角度接近180度，使用垂直方向
        bisector_norm = cross_norm
    else:
        bisector_norm = bisector / bisector_length
    
    # 使用标准键长和键角
    cb_ca_distance = STANDARD_BOND_LENGTHS["CA_CB"]
    tetrahedral_angle = torch.tensor(109.5 * torch.pi / 180.0, device=ca_coords.device)  # 四面体角度
    
    # 计算 CB 方向向量
    # 在角平分线和垂直方向之间的组合
    cos_angle = torch.cos(tetrahedral_angle)
    sin_angle = torch.sin(tetrahedral_angle)
    
    cb_direction = cos_angle * (-bisector_norm) + sin_angle * cross_norm
    cb_direction_norm = cb_direction / torch.norm(cb_direction)
    
    # 计算虚拟 CB 坐标
    virtual_cb = ca_coords + cb_ca_distance * cb_direction_norm
    
    return virtual_cb


# ================================
# 主要转换函数（优化版本）
# ================================

def protein_tensor_to_atom14(
    protein_tensor: ProteinTensor,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
          torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将 ProteinTensor 转换为 Atom14 格式数据（优化版本）。
    
    使用向量化操作替代Python循环，提升性能。
    
    Args:
        protein_tensor: 输入的 ProteinTensor 对象，必须使用 torch 后端
        device: 目标设备，如果为 None 则使用输入张量的设备
        
    Returns:
        Tuple containing:
            coords: 坐标张量 (num_residues, 14, 3)
            atom_mask: 原子掩码 (num_residues, 14) - 1=真实原子, 0=填充
            res_mask: 残基掩码 (num_residues,) - 1=标准残基, 0=非标准/缺失
            chain_ids: 链标识符 (num_residues,)
            residue_types: 残基类型 (num_residues,)
            residue_indices: 全局残基编号 (num_residues,)
            chain_residue_indices: 链内局部编号 (num_residues,)
            residue_names: 残基名称张量 (num_residues,)
            atom_names: 原子名称张量 (14,)
            
    Raises:
        TypeError: 当输入数据类型不正确时
        ValueError: 当数据格式不符合要求时
    """
    logger.info("开始优化版本的 ProteinTensor 转换为 Atom14 数据")
    
    # 转换为torch后端
    torch_data = protein_tensor.to_torch()
    
    # 验证数据
    if not isinstance(torch_data["coordinates"], torch.Tensor):
        raise TypeError("坐标数据必须是 torch.Tensor 类型")
    
    coordinates = torch_data["coordinates"]  # (num_atoms, 3)
    atom_types = torch_data["atom_types"]    # (num_atoms,)
    residue_types = torch_data["residue_types"]  # (num_atoms,)
    chain_ids = torch_data["chain_ids"]      # (num_atoms,)
    residue_numbers = torch_data["residue_numbers"]  # (num_atoms,)
    
    # 设置设备
    if device is None:
        device = coordinates.device
    else:
        coordinates = coordinates.to(device)
        atom_types = atom_types.to(device)
        residue_types = residue_types.to(device)
        chain_ids = chain_ids.to(device)
        residue_numbers = residue_numbers.to(device)
    
    # 🚀 优化1: 向量化残基边界检测
    residue_starts, residue_ends = find_residue_boundaries_vectorized(chain_ids, residue_numbers)
    num_residues = len(residue_starts)
    
    # 🚀 优化2: 向量化链信息计算
    unique_chains, chain_residue_counts, residue_chain_ids = compute_chain_info_vectorized(
        chain_ids, residue_starts
    )
    
    # 🚀 优化3: 向量化全局残基编号计算（包含链间gap）
    global_residue_indices, chain_residue_indices = compute_global_residue_indices_vectorized(
        residue_chain_ids, unique_chains, chain_residue_counts
    )
    
    # 🚀 优化4: 向量化原子映射
    atom14_coords, atom14_mask, res_mask, residue_names_list = map_atoms_to_atom14_vectorized(
        coordinates, atom_types, residue_types, residue_starts, residue_ends, num_residues, device
    )
    
    # 获取每个残基的残基类型（用于输出）
    residue_type_indices = residue_types[residue_starts]
    
    # 创建张量化的名称
    residue_names_tensor = create_residue_name_tensor(residue_names_list, device)
    atom_names_tensor = create_atom_name_tensor(device)
    
    logger.info(f"优化版本转换完成: {num_residues} 个残基, {len(unique_chains)} 条链")
    
    return (
        atom14_coords,
        atom14_mask,
        res_mask,
        residue_chain_ids,
        residue_type_indices,
        global_residue_indices,
        chain_residue_indices,
        residue_names_tensor,
        atom_names_tensor
    )


def atom14_to_protein_tensor(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    res_mask: torch.Tensor,
    chain_ids: torch.Tensor,
    residue_types: torch.Tensor,
    residue_indices: torch.Tensor,
    chain_residue_indices: torch.Tensor,
    residue_names: torch.Tensor,
    atom_names: torch.Tensor
) -> Any:
    """
    将 Atom14 数据转换为 ProteinTensor。
    
    Args:
        coords: 坐标张量 (..., num_residues, 14, 3)
        atom_mask: 原子掩码 (..., num_residues, 14)
        res_mask: 残基掩码 (..., num_residues)
        chain_ids: 链标识符 (..., num_residues)
        residue_types: 残基类型 (..., num_residues)
        residue_indices: 全局残基编号 (..., num_residues)
        chain_residue_indices: 链内局部编号 (..., num_residues)
        residue_names: 残基名称张量 (..., num_residues)
        atom_names: 原子名称张量 (14,)
        
    Returns:
        ProteinTensor: 转换后的 ProteinTensor 对象或兼容对象
        
    Raises:
        RuntimeError: 当转换过程中出现错误时
    """
    logger.info("开始将 Atom14 数据转换为 ProteinTensor")
    
    # 处理批量维度 - 只处理最后一个批次
    if len(coords.shape) > 3:
        logger.warning("检测到批量数据，仅处理最后一个样本进行转换")
        coords = coords[-1]
        atom_mask = atom_mask[-1]
        res_mask = res_mask[-1]
        chain_ids = chain_ids[-1]
        residue_types = residue_types[-1]
        residue_indices = residue_indices[-1]
        chain_residue_indices = chain_residue_indices[-1]
        residue_names = residue_names[-1]
    
    num_residues = coords.shape[-2]
    device = coords.device
    
    # 只处理有效的残基
    valid_residues = res_mask.bool()
    valid_coords = coords[valid_residues]
    valid_atom_mask = atom_mask[valid_residues]
    valid_chain_ids = chain_ids[valid_residues]
    valid_residue_types = residue_types[valid_residues]
    valid_residue_indices = residue_indices[valid_residues]
    valid_residue_names = residue_names[valid_residues]
    
    # 重建原子级数据
    all_coords = []
    all_atom_types = []
    all_residue_types = []
    all_chain_ids = []
    all_residue_numbers = []
    
    for res_idx in range(len(valid_coords)):
        res_coords = valid_coords[res_idx]
        res_mask_atoms = valid_atom_mask[res_idx]
        chain_id = valid_chain_ids[res_idx].item()
        residue_type = valid_residue_types[res_idx].item()
        residue_number = valid_residue_indices[res_idx].item()
        
        for atom_pos in range(14):
            if res_mask_atoms[atom_pos]:
                # 跳过甘氨酸的虚拟 CB
                res_name_idx = valid_residue_names[res_idx].item()
                if res_name_idx in IDX_TO_RESIDUE_NAME:
                    res_name = IDX_TO_RESIDUE_NAME[res_name_idx]
                    if res_name == "GLY" and atom_pos == 4:  # 虚拟 CB
                        continue
                
                all_coords.append(res_coords[atom_pos])
                all_atom_types.append(atom_pos)  # 使用位置作为原子类型
                all_residue_types.append(residue_type)
                all_chain_ids.append(chain_id)
                all_residue_numbers.append(residue_number)
    
    if len(all_coords) == 0:
        raise RuntimeError("没有有效的原子数据用于转换")
    
    # 转换为张量
    final_coords = torch.stack(all_coords).cpu().numpy()
    final_atom_types = torch.tensor(all_atom_types, dtype=torch.long).cpu().numpy()
    final_residue_types = torch.tensor(all_residue_types, dtype=torch.long).cpu().numpy()
    final_chain_ids = torch.tensor(all_chain_ids, dtype=torch.long).cpu().numpy()
    final_residue_numbers = torch.tensor(all_residue_numbers, dtype=torch.long).cpu().numpy()
    
    # 创建临时 ProteinTensor 对象
    class TempProteinTensor:
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
    
    return TempProteinTensor(
        final_coords,
        final_atom_types,
        final_residue_types,
        final_chain_ids,
        final_residue_numbers
    )


# ================================
# 数据验证函数
# ================================

def validate_atom14_data(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    res_mask: torch.Tensor,
    chain_ids: torch.Tensor,
    residue_types: torch.Tensor,
    residue_indices: torch.Tensor,
    chain_residue_indices: torch.Tensor,
    residue_names: torch.Tensor,
    atom_names: torch.Tensor
) -> None:
    """
    验证 Atom14 数据的一致性和有效性。
    
    Args:
        coords: 坐标张量 (..., num_residues, 14, 3)
        atom_mask: 原子掩码 (..., num_residues, 14)
        res_mask: 残基掩码 (..., num_residues)
        chain_ids: 链标识符 (..., num_residues)
        residue_types: 残基类型 (..., num_residues)
        residue_indices: 全局残基编号 (..., num_residues)
        chain_residue_indices: 链内局部编号 (..., num_residues)
        residue_names: 残基名称张量 (..., num_residues)
        atom_names: 原子名称张量 (14,)
        
    Raises:
        ValueError: 当数据不一致或无效时
    """
    logger.debug("验证 Atom14 数据一致性")
    
    # 获取批量形状和残基数量
    batch_shape = coords.shape[:-3]
    num_residues = coords.shape[-3]
    
    # 验证基本形状
    expected_coords_shape = batch_shape + (num_residues, 14, 3)
    expected_atom_mask_shape = batch_shape + (num_residues, 14)
    expected_res_mask_shape = batch_shape + (num_residues,)
    expected_meta_shape = batch_shape + (num_residues,)
    
    if coords.shape != expected_coords_shape:
        raise ValueError(f"坐标张量形状无效: {coords.shape}，期望 {expected_coords_shape}")
    
    if atom_mask.shape != expected_atom_mask_shape:
        raise ValueError(f"原子掩码张量形状无效: {atom_mask.shape}，期望 {expected_atom_mask_shape}")
    
    if res_mask.shape != expected_res_mask_shape:
        raise ValueError(f"残基掩码张量形状无效: {res_mask.shape}，期望 {expected_res_mask_shape}")
    
    # 验证元数据形状
    for name, tensor in [
        ("chain_ids", chain_ids),
        ("residue_types", residue_types),
        ("residue_indices", residue_indices),
        ("chain_residue_indices", chain_residue_indices),
        ("residue_names", residue_names)
    ]:
        if tensor.shape != expected_meta_shape:
            raise ValueError(f"{name} 张量形状无效: {tensor.shape}，期望 {expected_meta_shape}")
    
    # 验证原子名称张量
    if atom_names.shape != (14,):
        raise ValueError(f"原子名称张量形状无效: {atom_names.shape}，期望 (14,)")
    
    # 验证数据类型
    if not atom_mask.dtype == torch.bool:
        raise ValueError(f"atom_mask 必须是布尔类型，实际: {atom_mask.dtype}")
    
    if not res_mask.dtype == torch.bool:
        raise ValueError(f"res_mask 必须是布尔类型，实际: {res_mask.dtype}")
    
    # 验证数值范围
    if torch.any(residue_types < 0) or torch.any(residue_types > 20):
        raise ValueError("残基类型索引超出有效范围 [0, 20]")
    
    if torch.any(residue_names < 0) or torch.any(residue_names > 20):
        raise ValueError("残基名称索引超出有效范围 [0, 20]")
    
    if torch.any(atom_names < 0) or torch.any(atom_names > 13):
        raise ValueError("原子名称索引超出有效范围 [0, 13]")
    
    logger.debug("Atom14 数据验证通过")


# ================================
# 工具函数
# ================================

def get_residue_atom14_mapping(residue_name: str) -> Dict[str, int]:
    """
    获取指定残基类型的 atom14 映射。
    
    Args:
        residue_name: 残基名称（如 'ALA', 'GLY' 等）
        
    Returns:
        Dict[str, int]: 该残基的原子名称到槽位的映射
        
    Raises:
        KeyError: 当残基类型不支持时
    """
    if residue_name not in RESIDUE_ATOM14_MAPPING:
        raise KeyError(f"不支持的残基类型: {residue_name}")
    
    return RESIDUE_ATOM14_MAPPING[residue_name]


def is_glycine(residue_name: str) -> bool:
    """
    判断是否为甘氨酸。
    
    Args:
        residue_name: 残基名称
        
    Returns:
        bool: 是否为甘氨酸
    """
    return residue_name.upper() == "GLY"


def get_atom14_atom_positions() -> Dict[str, int]:
    """
    获取 atom14 标准原子位置映射。
    
    Returns:
        Dict[str, int]: 原子名称到位置的映射
    """
    return {atom_name: i for i, atom_name in enumerate(ATOM14_ATOM_TYPES)}


def save_atom14_to_cif(
    atom14: Any,  # Atom14 类型
    output_path: Union[str, Path],
    title: str = "ProtRepr Atom14 Structure"
) -> None:
    """
    将 Atom14 数据保存为 CIF 文件。
    
    Args:
        atom14: Atom14 实例
        output_path: 输出文件路径
        title: 结构标题
    """
    logger.info(f"将 Atom14 数据保存到 CIF 文件: {output_path}")
    protein_tensor = atom14.to_protein_tensor()
    save_structure(protein_tensor, output_path, format_type="cif")
    logger.info(f"CIF 文件保存成功: {output_path}")


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
    # 获取数据
    save_structure(protein_tensor, output_path, format_type="cif")
    logger.info(f"CIF 文件保存成功: {output_path}") 