"""
Atom37 转换工具模块

本模块提供 Atom37 数据类与 ProteinTensor 之间的转换工具函数。
这些函数被 Atom37 数据类的类方法调用，提供底层的转换逻辑。

核心功能：
- protein_tensor_to_atom37: 将 ProteinTensor 转换为 Atom37 数据
- atom37_to_protein_tensor: 将 Atom37 数据转换回 ProteinTensor
- validate_atom37_data: 验证 Atom37 数据的有效性
- 提供 atom37 标准的原子类型映射和常量定义
- 支持所有20种标准氨基酸的重原子表示
"""

import logging
from typing import Tuple, Dict, List, Optional, Any

import torch
from protein_tensor import ProteinTensor

logger = logging.getLogger(__name__)

# atom37 标准原子类型到槽位的映射表（按标准顺序）
ATOM37_ATOM_TYPES: List[str] = [
    # 主链原子 (所有残基共有，0-3)
    "N", "CA", "C", "O",
    # 通用侧链原子 (4-10)  
    "CB", "CG", "CG1", "CG2", "CD", "CD1", "CD2",
    # 深层侧链原子 (11-20)
    "CE", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2", "NZ", "NH1",
    # 特殊原子类型 (21-36)
    "NH2", "ND1", "ND2", "NE", "NE1", "NE2", "OE1", "OE2", "OG", "OG1",
    "OH", "SD", "SG", "OD1", "OD2", "OXT", "OT1"
    # TODO: 验证并完善完整的37个原子类型定义
]

# 氨基酸类型到 atom37 槽位的映射
RESIDUE_ATOM37_MAPPING: Dict[str, Dict[str, int]] = {
    "ALA": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4},
    "GLY": {"N": 0, "CA": 1, "C": 2, "O": 3},  # 甘氨酸没有CB
    "SER": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "OG": 18},
    "THR": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "OG1": 19, "CG2": 7},
    # TODO: 添加其他16种标准氨基酸的完整映射
}


def protein_tensor_to_atom37(
    protein_tensor: ProteinTensor,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
           torch.Tensor, List[str], List[str]]:
    """
    将 ProteinTensor 转换为 Atom37 表示所需的所有数据。
    
    Args:
        protein_tensor: 输入的 ProteinTensor 对象，必须使用 torch 后端
        device: 目标设备，如果为 None 则使用输入张量的设备
        
    Returns:
        Tuple包含:
            - coords: 形状为 (num_residues, 37, 3) 的坐标张量
            - mask: 形状为 (num_residues, 37) 的掩码张量
            - chain_ids: 形状为 (num_residues,) 的链标识符张量
            - residue_types: 形状为 (num_residues,) 的残基类型张量
            - residue_indices: 形状为 (num_residues,) 的残基位置张量
            - residue_names: 残基名称列表
            - atom_names: atom37 原子名称列表
            
    Raises:
        ValueError: 当 protein_tensor 未使用 torch 后端时
        TypeError: 当坐标数据不是 torch.Tensor 类型时
        RuntimeError: 当转换过程中出现错误时
    """
    logger.info("开始将 ProteinTensor 转换为 Atom37 数据")
    
    # TODO: 实现完整的转换逻辑
    # 1. 验证输入参数的有效性
    # 2. 提取原子坐标和元数据
    # 3. 根据 RESIDUE_ATOM37_MAPPING 映射原子位置
    # 4. 处理缺失原子的填充（用零向量）
    # 5. 生成掩码张量标识真实原子
    # 6. 提取链信息和残基信息
    # 7. 处理设备转移
    
    raise NotImplementedError("ProteinTensor 到 Atom37 转换功能尚未实现")


def atom37_to_protein_tensor(
    coords: torch.Tensor,
    mask: torch.Tensor,
    chain_ids: torch.Tensor,
    residue_types: torch.Tensor,
    residue_indices: torch.Tensor,
    residue_names: List[str],
    atom_names: List[str]
) -> ProteinTensor:
    """
    将 Atom37 数据转换为 ProteinTensor。
    
    Args:
        coords: 形状为 (num_residues, 37, 3) 的坐标张量
        mask: 形状为 (num_residues, 37) 的掩码张量
        chain_ids: 形状为 (num_residues,) 的链标识符张量
        residue_types: 形状为 (num_residues,) 的残基类型张量
        residue_indices: 形状为 (num_residues,) 的残基位置张量
        residue_names: 残基名称列表
        atom_names: atom37 原子名称列表
        
    Returns:
        ProteinTensor: 转换后的 ProteinTensor 对象，使用 torch 后端
        
    Raises:
        RuntimeError: 当转换过程中出现错误时
    """
    logger.info("开始将 Atom37 数据转换为 ProteinTensor")
    
    # TODO: 实现完整的转换逻辑
    # 1. 根据掩码提取真实原子坐标
    # 2. 重建原子类型和残基信息
    # 3. 构造 ProteinTensor 所需的数据结构
    # 4. 确保使用 torch 后端
    # 5. 保持原始的原子顺序和命名
    # 6. 过滤掉填充的零向量位置
    
    raise NotImplementedError("Atom37 到 ProteinTensor 转换功能尚未实现")


def validate_atom37_data(
    coords: torch.Tensor,
    mask: torch.Tensor,
    chain_ids: torch.Tensor,
    residue_types: torch.Tensor,
    residue_indices: torch.Tensor,
    residue_names: List[str],
    atom_names: List[str]
) -> None:
    """
    验证 Atom37 数据的一致性和有效性。
    
    Args:
        coords: 坐标张量
        mask: 掩码张量
        chain_ids: 链标识符张量
        residue_types: 残基类型张量
        residue_indices: 残基位置张量
        residue_names: 残基名称列表
        atom_names: 原子名称列表
        
    Raises:
        ValueError: 当数据不一致或无效时
    """
    logger.debug("验证 Atom37 数据一致性")
    
    # TODO: 实现数据验证
    # 1. 验证张量形状（coords: (N, 37, 3), mask: (N, 37)）
    # 2. 验证数据类型
    # 3. 验证坐标范围的合理性
    # 4. 验证掩码的逻辑一致性
    # 5. 验证元数据的一致性
    # 6. 验证原子名称和残基名称的有效性
    # 7. 验证 atom_names 长度为 37
    # 8. 验证 residue_names 与残基数量匹配
    
    pass  # 临时跳过验证


def get_atom37_atom_positions() -> Dict[str, int]:
    """
    获取 atom37 标准中原子类型到槽位的映射表。
    
    Returns:
        Dict[str, int]: 原子类型名称到槽位索引的映射
    """
    return {atom_type: idx for idx, atom_type in enumerate(ATOM37_ATOM_TYPES)}


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


def check_atom_completeness(
    residue_name: str, 
    atom_names: List[str]
) -> Tuple[List[str], List[str]]:
    """
    检查指定残基的原子完整性。
    
    Args:
        residue_name: 残基名称
        atom_names: 实际存在的原子名称列表
        
    Returns:
        Tuple[List[str], List[str]]: 
            - missing_atoms: 缺失的原子列表
            - extra_atoms: 多余的原子列表
    """
    if residue_name not in RESIDUE_ATOM37_MAPPING:
        return [], list(atom_names)  # 未知残基类型
    
    expected_atoms = set(RESIDUE_ATOM37_MAPPING[residue_name].keys())
    actual_atoms = set(atom_names)
    
    missing_atoms = list(expected_atoms - actual_atoms)
    extra_atoms = list(actual_atoms - expected_atoms)
    
    return missing_atoms, extra_atoms


def compute_residue_center_of_mass(
    coords: torch.Tensor,
    mask: torch.Tensor,
    residue_idx: int
) -> torch.Tensor:
    """
    计算指定残基的质心坐标。
    
    Args:
        coords: 形状为 (num_residues, 37, 3) 的坐标张量
        mask: 形状为 (num_residues, 37) 的掩码张量
        residue_idx: 残基索引
        
    Returns:
        torch.Tensor: 形状为 (3,) 的质心坐标
        
    Notes:
        只考虑掩码标识为True的真实原子
    """
    residue_coords = coords[residue_idx]  # (37, 3)
    residue_mask = mask[residue_idx]      # (37,)
    
    # TODO: 实现质心计算
    # 1. 根据掩码过滤真实原子
    # 2. 计算坐标的平均值
    # 3. 处理没有真实原子的情况
    
    raise NotImplementedError("残基质心计算功能尚未实现")


def get_backbone_atom_indices() -> List[int]:
    """
    获取主链原子在 atom37 中的索引。
    
    Returns:
        List[int]: 主链原子索引列表 [N, CA, C, O]
    """
    backbone_atoms = ["N", "CA", "C", "O"]
    atom_positions = get_atom37_atom_positions()
    return [atom_positions[atom] for atom in backbone_atoms if atom in atom_positions]


def get_sidechain_atom_indices(residue_name: str) -> List[int]:
    """
    获取指定残基侧链原子在 atom37 中的索引。
    
    Args:
        residue_name: 残基名称
        
    Returns:
        List[int]: 侧链原子索引列表
    """
    if residue_name not in RESIDUE_ATOM37_MAPPING:
        return []
    
    backbone_atoms = {"N", "CA", "C", "O"}
    residue_mapping = RESIDUE_ATOM37_MAPPING[residue_name]
    
    sidechain_indices = []
    for atom_name, atom_idx in residue_mapping.items():
        if atom_name not in backbone_atoms:
            sidechain_indices.append(atom_idx)
    
    return sorted(sidechain_indices) 