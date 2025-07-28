"""
Atom37 表示转换器 (基于 AlphaFold 标准)

本模块提供 ProteinTensor 与 Atom37 表示之间的高性能转换功能，包括：
- 基于 AlphaFold 标准的 37 个重原子槽位定义
- 向量化的坐标到 atom37 格式映射
- 支持所有 20 种标准氨基酸的完整原子映射
- 数据验证和完整性检查
- 支持批量操作、分离掩码、链间信息和张量化名称
- 复用 Atom14 的优化技术，使用 PyTorch 张量操作替代 Python 循环
"""

import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import torch
from protein_tensor import ProteinTensor, save_structure

# 导入基础转换器的共同功能
from .base_converter import (
    RESIDUE_NAME_TO_IDX, IDX_TO_RESIDUE_NAME, CHAIN_GAP,
    create_residue_name_tensor, decode_residue_names,
    find_residue_boundaries_vectorized, compute_chain_info_vectorized,
    compute_global_residue_indices_vectorized, TempProteinTensor,
    save_protein_tensor_to_cif
)
from .base_converter import create_atom_name_tensor as _create_atom_name_tensor
from .base_converter import decode_atom_names as _decode_atom_names

logger = logging.getLogger(__name__)

# ================================
# AlphaFold Atom37 标准定义
# ================================

# Atom37 标准原子类型列表（37个固定槽位，按 AlphaFold 标准顺序）
ATOM37_ATOM_TYPES = [
    "N",      # 0 - 主链氮
    "CA",     # 1 - 主链 α-碳
    "C",      # 2 - 主链羰基碳
    "O",      # 3 - 主链羰基氧
    "CB",     # 4 - β-碳
    "CG",     # 5 - 侧链碳1
    "CG1",    # 6 - 侧链分支碳1（ILE, VAL）
    "CG2",    # 7 - 侧链分支碳2（ILE, THR, VAL）
    "CD",     # 8 - 侧链碳2
    "CD1",    # 9 - 侧链分支碳3（LEU, PHE, TRP, TYR）
    "CD2",    # 10 - 侧链分支碳4（LEU, PHE, HIS, TRP, TYR）
    "CE",     # 11 - 侧链碳3（LYS, MET）
    "CE1",    # 12 - 侧链分支碳5（PHE, HIS, TRP, TYR）
    "CE2",    # 13 - 侧链分支碳6（PHE, TRP, TYR）
    "CE3",    # 14 - 侧链分支碳7（TRP）
    "CZ",     # 15 - 侧链碳4（ARG, PHE, TYR）
    "CZ2",    # 16 - 侧链分支碳8（TRP）
    "CZ3",    # 17 - 侧链分支碳9（TRP）
    "CH2",    # 18 - 侧链碳5（TRP）
    "NZ",     # 19 - 侧链氮（LYS）
    "NH1",    # 20 - 侧链氮1（ARG）
    "NH2",    # 21 - 侧链氮2（ARG）
    "ND1",    # 22 - 侧链氮3（HIS）
    "ND2",    # 23 - 侧链氮4（ASN, HIS）
    "NE",     # 24 - 侧链氮5（ARG）
    "NE1",    # 25 - 侧链氮6（TRP）
    "NE2",    # 26 - 侧链氮7（GLN, HIS）
    "OD1",    # 27 - 侧链氧1（ASP, ASN）
    "OD2",    # 28 - 侧链氧2（ASP）
    "OE1",    # 29 - 侧链氧3（GLU, GLN）
    "OE2",    # 30 - 侧链氧4（GLU）
    "OG",     # 31 - 侧链氧5（SER）
    "OG1",    # 32 - 侧链氧6（THR）
    "OH",     # 33 - 侧链氧7（TYR）
    "SD",     # 34 - 侧链硫1（MET）
    "SG",     # 35 - 侧链硫2（CYS）
    "OXT"     # 36 - C端额外氧（可选）
]

# 原子名称到索引的映射
ATOM_NAME_TO_IDX = {name: idx for idx, name in enumerate(ATOM37_ATOM_TYPES)}

# 索引到原子名称的映射
IDX_TO_ATOM_NAME = {idx: name for idx, name in enumerate(ATOM37_ATOM_TYPES)}

# ================================
# AlphaFold 标准 Atom37 原子映射
# ================================

# 每种残基的原子到 atom37 槽位的映射（基于 AlphaFold 标准）
RESIDUE_ATOM37_MAPPING: Dict[str, Dict[str, int]] = {
    "ALA": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4},
    
    "ARG": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "CD": 8, 
            "NE": 24, "CZ": 15, "NH1": 20, "NH2": 21},
    
    "ASN": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, 
            "OD1": 27, "ND2": 23},
    
    "ASP": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, 
            "OD1": 27, "OD2": 28},
    
    "CYS": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "SG": 35},
    
    "GLN": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "CD": 8,
            "OE1": 29, "NE2": 26},
    
    "GLU": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "CD": 8,
            "OE1": 29, "OE2": 30},
    
    "GLY": {"N": 0, "CA": 1, "C": 2, "O": 3},  # 甘氨酸没有CB
    
    "HIS": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5,
            "ND1": 22, "CD2": 10, "CE1": 12, "NE2": 26},
    
    "ILE": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG1": 6, "CG2": 7, "CD1": 9},
    
    "LEU": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "CD1": 9, "CD2": 10},
    
    "LYS": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "CD": 8, 
            "CE": 11, "NZ": 19},
    
    "MET": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "SD": 34, "CE": 11},
    
    "PHE": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5,
            "CD1": 9, "CD2": 10, "CE1": 12, "CE2": 13, "CZ": 15},
    
    "PRO": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "CD": 8},
    
    "SER": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "OG": 31},
    
    "THR": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "OG1": 32, "CG2": 7},
    
    "TRP": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5,
            "CD1": 9, "CD2": 10, "NE1": 25, "CE2": 13, "CE3": 14,
            "CZ2": 16, "CZ3": 17, "CH2": 18},
    
    "TYR": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5,
            "CD1": 9, "CD2": 10, "CE1": 12, "CE2": 13, "CZ": 15, "OH": 33},
    
    "VAL": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG1": 6, "CG2": 7}
}


# ================================
# Atom37 专用包装函数
# ================================

def create_atom_name_tensor(device: torch.device) -> torch.Tensor:
    """
    创建 atom37 原子名称张量（整数编码）。
    
    Args:
        device: 目标设备
        
    Returns:
        torch.Tensor: 编码后的原子名称张量 (37,)
    """
    return _create_atom_name_tensor(37, device)


def decode_atom_names(atom_tensor: torch.Tensor) -> List[str]:
    """
    将原子名称张量解码为字符串列表。
    
    Args:
        atom_tensor: 编码的原子名称张量
        
    Returns:
        List[str]: 解码后的原子名称列表
    """
    return _decode_atom_names(atom_tensor, IDX_TO_ATOM_NAME)


# ================================
# Atom37 专用的向量化辅助函数
# ================================


def map_atoms_to_atom37_vectorized(
    coordinates: torch.Tensor,
    atom_types: torch.Tensor,
    residue_types: torch.Tensor,
    residue_starts: torch.Tensor,
    residue_ends: torch.Tensor,
    num_residues: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    使用向量化操作将原子映射到atom37格式。
    
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
            atom37_coords: atom37坐标 (num_residues, 37, 3)
            atom37_mask: atom37掩码 (num_residues, 37)
            res_mask: 残基掩码 (num_residues,)
            residue_names_list: 残基名称列表
    """
    # 初始化输出张量
    atom37_coords = torch.zeros(num_residues, 37, 3, device=device)
    atom37_mask = torch.zeros(num_residues, 37, dtype=torch.bool, device=device)
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
        if res_name not in RESIDUE_ATOM37_MAPPING:
            res_mask[res_idx] = False
            continue
            
        mapping = RESIDUE_ATOM37_MAPPING[res_name]
        
        # 处理这个残基的所有原子
        start_atom = residue_starts[res_idx].item()
        end_atom = residue_ends[res_idx].item()
        
        # 批量获取原子信息
        residue_atom_types = atom_types[start_atom:end_atom]
        residue_coords = coordinates[start_atom:end_atom]
        
        # 映射原子到atom37位置
        for local_atom_idx, atom_type_idx in enumerate(residue_atom_types):
            atom_type_idx = atom_type_idx.item()
            
            if atom_type_idx in IDX_TO_ATOM_NAME:
                atom_name = IDX_TO_ATOM_NAME[atom_type_idx]
                
                if atom_name in mapping:
                    atom37_pos = mapping[atom_name]
                    atom37_coords[res_idx, atom37_pos] = residue_coords[local_atom_idx]
                    atom37_mask[res_idx, atom37_pos] = True
    
    return atom37_coords, atom37_mask, res_mask, residue_names_list


# ================================
# 主要转换函数
# ================================

def protein_tensor_to_atom37(
    protein_tensor: ProteinTensor,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
          torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将 ProteinTensor 转换为 Atom37 格式数据。
    
    Args:
        protein_tensor: 输入的 ProteinTensor 对象，必须使用 torch 后端
        device: 目标设备，如果为 None 则使用输入张量的设备
        
    Returns:
        Tuple containing:
            coords: 坐标张量 (num_residues, 37, 3)
            atom_mask: 原子掩码 (num_residues, 37) - 1=真实原子, 0=填充
            res_mask: 残基掩码 (num_residues,) - 1=标准残基, 0=非标准/缺失
            chain_ids: 链标识符 (num_residues,)
            residue_types: 残基类型 (num_residues,)
            residue_indices: 全局残基编号 (num_residues,)
            chain_residue_indices: 链内局部编号 (num_residues,)
            residue_names: 残基名称张量 (num_residues,)
            atom_names: 原子名称张量 (37,)
            
    Raises:
        TypeError: 当输入数据类型不正确时
        ValueError: 当数据格式不符合要求时
    """
    logger.info("开始 ProteinTensor 转换为 Atom37 数据")
    
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
    
    # 🚀 优化3: 向量化全局残基编号计算
    global_residue_indices, chain_residue_indices = compute_global_residue_indices_vectorized(
        residue_chain_ids, unique_chains, chain_residue_counts
    )
    
    # 🚀 优化4: 向量化原子映射
    atom37_coords, atom37_mask, res_mask, residue_names_list = map_atoms_to_atom37_vectorized(
        coordinates, atom_types, residue_types, residue_starts, residue_ends, num_residues, device
    )
    
    # 获取每个残基的残基类型（用于输出）
    residue_type_indices = residue_types[residue_starts]
    
    # 创建张量化的名称
    residue_names_tensor = create_residue_name_tensor(residue_names_list, device)
    atom_names_tensor = create_atom_name_tensor(device)
    
    logger.info(f"Atom37 转换完成: {num_residues} 个残基, {len(unique_chains)} 条链")
    
    return (
        atom37_coords,
        atom37_mask,
        res_mask,
        residue_chain_ids,
        residue_type_indices,
        global_residue_indices,
        chain_residue_indices,
        residue_names_tensor,
        atom_names_tensor
    )


def atom37_to_protein_tensor(
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
    将 Atom37 数据转换为 ProteinTensor。
    
    Args:
        coords: 坐标张量 (..., num_residues, 37, 3)
        atom_mask: 原子掩码 (..., num_residues, 37)
        res_mask: 残基掩码 (..., num_residues)
        chain_ids: 链标识符 (..., num_residues)
        residue_types: 残基类型 (..., num_residues)
        residue_indices: 全局残基编号 (..., num_residues)
        chain_residue_indices: 链内局部编号 (..., num_residues)
        residue_names: 残基名称张量 (..., num_residues)
        atom_names: 原子名称张量 (37,)
        
    Returns:
        ProteinTensor: 转换后的 ProteinTensor 对象或兼容对象
        
    Raises:
        RuntimeError: 当转换过程中出现错误时
    """
    logger.info("开始将 Atom37 数据转换为 ProteinTensor")
    
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
    
    num_residues = coords.shape[-3]
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
        
        for atom_pos in range(37):
            if res_mask_atoms[atom_pos]:
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
    
    # 使用基础转换器的 TempProteinTensor 类
    
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

def validate_atom37_data(
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
    验证 Atom37 数据的一致性和有效性。
    
    Args:
        coords: 坐标张量 (..., num_residues, 37, 3)
        atom_mask: 原子掩码 (..., num_residues, 37)
        res_mask: 残基掩码 (..., num_residues)
        chain_ids: 链标识符 (..., num_residues)
        residue_types: 残基类型 (..., num_residues)
        residue_indices: 全局残基编号 (..., num_residues)
        chain_residue_indices: 链内局部编号 (..., num_residues)
        residue_names: 残基名称张量 (..., num_residues)
        atom_names: 原子名称张量 (37,)
        
    Raises:
        ValueError: 当数据不一致或无效时
    """
    logger.debug("验证 Atom37 数据一致性")
    
    # 获取批量形状和残基数量
    batch_shape = coords.shape[:-3]
    num_residues = coords.shape[-3]
    
    # 验证基本形状
    expected_coords_shape = batch_shape + (num_residues, 37, 3)
    expected_atom_mask_shape = batch_shape + (num_residues, 37)
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
    if atom_names.shape != (37,):
        raise ValueError(f"原子名称张量形状无效: {atom_names.shape}，期望 (37,)")
    
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
    
    if torch.any(atom_names < 0) or torch.any(atom_names > 36):
        raise ValueError("原子名称索引超出有效范围 [0, 36]")
    
    logger.debug("Atom37 数据验证通过")


# ================================
# 工具函数
# ================================

def get_residue_atom37_mapping(residue_name: str) -> Dict[str, int]:
    """
    获取指定残基类型的 atom37 映射。
    
    Args:
        residue_name: 残基名称（如 'ALA', 'GLY' 等）
        
    Returns:
        Dict[str, int]: 该残基的原子名称到槽位的映射
        
    Raises:
        KeyError: 当残基类型不支持时
    """
    if residue_name not in RESIDUE_ATOM37_MAPPING:
        raise KeyError(f"不支持的残基类型: {residue_name}")
    
    return RESIDUE_ATOM37_MAPPING[residue_name]


def get_atom37_atom_positions() -> Dict[str, int]:
    """
    获取 atom37 标准原子位置映射。
    
    Returns:
        Dict[str, int]: 原子名称到位置的映射
    """
    return {atom_name: i for i, atom_name in enumerate(ATOM37_ATOM_TYPES)}


def save_atom37_to_cif(
    atom37: Any,  # Atom37 类型
    output_path: Union[str, Path],
    title: str = "ProtRepr Atom37 Structure"
) -> None:
    """
    将 Atom37 数据保存为 CIF 文件。
    
    Args:
        atom37: Atom37 实例
        output_path: 输出文件路径
        title: 结构标题
    """
    logger.info(f"将 Atom37 数据保存到 CIF 文件: {output_path}")
    protein_tensor = atom37.to_protein_tensor()
    save_structure(protein_tensor, output_path, format_type="cif")
    logger.info(f"CIF 文件保存成功: {output_path}")


