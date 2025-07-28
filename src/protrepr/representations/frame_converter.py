"""
Frame 表示转换器

本模块提供 ProteinTensor 与 Frame 表示之间的高性能转换功能，包括：
- 从主链原子计算刚体变换（旋转矩阵和平移向量）
- 从刚体变换重建主链原子坐标
- 数据验证和完整性检查
- 支持批量操作、分离掩码、链间信息和张量化名称
- 全面的性能优化，使用 PyTorch 张量操作替代 Python 循环
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

logger = logging.getLogger(__name__)

# ================================
# Frame 专用常量定义
# ================================

# 主链原子名称到索引的映射
BACKBONE_ATOM_NAME_TO_IDX = {
    "N": 0, "CA": 1, "C": 2, "O": 3
}

# 索引到主链原子名称的映射
IDX_TO_BACKBONE_ATOM_NAME = {v: k for k, v in BACKBONE_ATOM_NAME_TO_IDX.items()}

# ================================
# Frame 专用的向量化辅助函数
# ================================


def extract_backbone_atoms_vectorized(
    coordinates: torch.Tensor,
    atom_types: torch.Tensor,
    residue_starts: torch.Tensor,
    residue_ends: torch.Tensor,
    num_residues: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用向量化操作提取主链原子坐标。
    
    Args:
        coordinates: 原子坐标 (num_atoms, 3)
        atom_types: 原子类型 (num_atoms,)
        residue_starts: 残基起始索引 (num_residues,)
        residue_ends: 残基结束索引 (num_residues,)
        num_residues: 残基数量
        device: 设备
        
    Returns:
        Tuple containing:
            backbone_coords: 主链原子坐标 (num_residues, 4, 3) - N, CA, C, O
            backbone_mask: 主链原子掩码 (num_residues, 4) - 标识哪些原子存在
    """
    # 初始化输出张量
    backbone_coords = torch.zeros(num_residues, 4, 3, device=device)
    backbone_mask = torch.zeros(num_residues, 4, dtype=torch.bool, device=device)
    
    # 批量处理每个残基
    for res_idx in range(num_residues):
        start_atom = residue_starts[res_idx].item()
        end_atom = residue_ends[res_idx].item()
        
        # 获取这个残基的原子信息
        residue_atom_types = atom_types[start_atom:end_atom]
        residue_coords = coordinates[start_atom:end_atom]
        
        # 为该残基查找主链原子，使用更严格的匹配
        backbone_found = {0: False, 1: False, 2: False, 3: False}  # N, CA, C, O
        
        for local_atom_idx, atom_type_idx in enumerate(residue_atom_types):
            atom_type_int = int(atom_type_idx.item())
            
            # 只接受标准主链原子类型 (N=0, CA=1, C=2, O=3)
            if atom_type_int in backbone_found and not backbone_found[atom_type_int]:
                backbone_coords[res_idx, atom_type_int] = residue_coords[local_atom_idx]
                backbone_mask[res_idx, atom_type_int] = True
                backbone_found[atom_type_int] = True
    
    return backbone_coords, backbone_mask


# ================================
# 主要转换函数
# ================================

def protein_tensor_to_frame(
    protein_tensor: ProteinTensor,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
          torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将 ProteinTensor 转换为 Frame 格式数据。
    
    Args:
        protein_tensor: 输入的 ProteinTensor 对象，必须使用 torch 后端
        device: 目标设备，如果为 None 则使用输入张量的设备
        
    Returns:
        Tuple containing:
            translations: 平移向量 (num_residues, 3) - CA 原子坐标
            rotations: 旋转矩阵 (num_residues, 3, 3) - 局部坐标系
            res_mask: 残基掩码 (num_residues,) - 1=标准残基, 0=非标准/缺失
            chain_ids: 链标识符 (num_residues,)
            residue_types: 残基类型 (num_residues,)
            residue_indices: 全局残基编号 (num_residues,)
            chain_residue_indices: 链内局部编号 (num_residues,)
            residue_names: 残基名称张量 (num_residues,)
            
    Raises:
        TypeError: 当输入数据类型不正确时
        ValueError: 当数据格式不符合要求时
    """
    logger.info("开始将 ProteinTensor 转换为 Frame 数据")
    
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
    
    # 🚀 优化4: 向量化主链原子提取
    backbone_coords, backbone_mask = extract_backbone_atoms_vectorized(
        coordinates, atom_types, residue_starts, residue_ends, num_residues, device
    )
    
    # 获取每个残基的残基类型（用于输出）
    residue_type_indices = residue_types[residue_starts]
    
    # 创建残基名称列表和张量，同时过滤有效残基
    residue_names_list = []
    res_mask = torch.zeros(num_residues, dtype=torch.bool, device=device)
    
    # 统计数据用于调试
    total_residues = num_residues
    valid_residues = 0
    missing_backbone_count = 0
    
    for res_idx in range(num_residues):
        res_type_idx = residue_type_indices[res_idx].item()
        
        # 检查是否有足够的主链原子 (至少需要 N, CA, C)
        has_n = backbone_mask[res_idx, 0]   # N
        has_ca = backbone_mask[res_idx, 1]  # CA  
        has_c = backbone_mask[res_idx, 2]   # C
        has_backbone = has_n and has_ca and has_c
        
        # 如果有主链原子但不完整，记录警告
        if not has_backbone and (has_n or has_ca or has_c):
            missing_atoms = []
            if not has_n: missing_atoms.append("N")
            if not has_ca: missing_atoms.append("CA")
            if not has_c: missing_atoms.append("C")
            logger.debug(f"残基 {res_idx} 缺少主链原子: {missing_atoms}")
            missing_backbone_count += 1
        
        if res_type_idx in IDX_TO_RESIDUE_NAME and has_backbone:
            res_name = IDX_TO_RESIDUE_NAME[res_type_idx]
            res_mask[res_idx] = True
            valid_residues += 1
        else:
            res_name = "UNK"
            res_mask[res_idx] = False
            
        residue_names_list.append(res_name)
    
    # 输出统计信息
    logger.info(f"残基过滤统计: {valid_residues}/{total_residues} 有效残基, {missing_backbone_count} 个残基缺少主链原子")
    
    # 计算刚体变换
    from ..utils.geometry import compute_rigid_transforms_from_backbone
    
    # 提取主链原子坐标（N, CA, C）
    n_coords = backbone_coords[:, 0, :]   # N 原子
    ca_coords = backbone_coords[:, 1, :]  # CA 原子
    c_coords = backbone_coords[:, 2, :]   # C 原子
    
    # 过滤：只对有完整主链原子的残基计算刚体变换
    valid_indices = torch.where(res_mask)[0]
    
    if len(valid_indices) == 0:
        logger.error("没有找到任何有完整主链原子的残基")
        raise ValueError("没有找到有效的残基用于Frame转换")
    
    # 只对有效残基计算刚体变换
    valid_n_coords = n_coords[res_mask]
    valid_ca_coords = ca_coords[res_mask]
    valid_c_coords = c_coords[res_mask]
    
    logger.info(f"对 {len(valid_indices)}/{num_residues} 个有效残基计算刚体变换")
    
    # 计算刚体变换
    valid_translations, valid_rotations = compute_rigid_transforms_from_backbone(
        valid_n_coords, valid_ca_coords, valid_c_coords
    )
    
    # 将结果填回完整的张量中
    translations = torch.zeros(num_residues, 3, device=device)
    rotations = torch.eye(3, device=device).unsqueeze(0).repeat(num_residues, 1, 1)
    
    translations[res_mask] = valid_translations
    rotations[res_mask] = valid_rotations
    
    # 创建张量化的残基名称
    residue_names_tensor = create_residue_name_tensor(residue_names_list, device)
    
    logger.info(f"Frame 转换完成: {num_residues} 个残基, {len(unique_chains)} 条链")
    
    return (
        translations,
        rotations,
        res_mask,
        residue_chain_ids,
        residue_type_indices,
        global_residue_indices,
        chain_residue_indices,
        residue_names_tensor
    )


def frame_to_protein_tensor(
    translations: torch.Tensor,
    rotations: torch.Tensor,
    res_mask: torch.Tensor,
    chain_ids: torch.Tensor,
    residue_types: torch.Tensor,
    residue_indices: torch.Tensor,
    chain_residue_indices: torch.Tensor,
    residue_names: torch.Tensor
) -> Any:
    """
    将 Frame 数据转换为 ProteinTensor。
    
    Args:
        translations: 平移向量 (..., num_residues, 3)
        rotations: 旋转矩阵 (..., num_residues, 3, 3)
        res_mask: 残基掩码 (..., num_residues)
        chain_ids: 链标识符 (..., num_residues)
        residue_types: 残基类型 (..., num_residues)
        residue_indices: 全局残基编号 (..., num_residues)
        chain_residue_indices: 链内局部编号 (..., num_residues)
        residue_names: 残基名称张量 (..., num_residues)
        
    Returns:
        ProteinTensor: 转换后的 ProteinTensor 对象或兼容对象
        
    Raises:
        RuntimeError: 当转换过程中出现错误时
    """
    logger.info("开始将 Frame 数据转换为 ProteinTensor")
    
    # 处理批量维度 - 只处理最后一个批次
    if len(translations.shape) > 2:
        logger.warning("检测到批量数据，仅处理最后一个样本进行转换")
        translations = translations[-1]
        rotations = rotations[-1]
        res_mask = res_mask[-1]
        chain_ids = chain_ids[-1]
        residue_types = residue_types[-1]
        residue_indices = residue_indices[-1]
        chain_residue_indices = chain_residue_indices[-1]
        residue_names = residue_names[-1]
    
    num_residues = translations.shape[-2]
    device = translations.device
    
    # 只处理有效的残基
    valid_residues = res_mask.bool()
    valid_translations = translations[valid_residues]
    valid_rotations = rotations[valid_residues]
    valid_chain_ids = chain_ids[valid_residues]
    valid_residue_types = residue_types[valid_residues]
    valid_residue_indices = residue_indices[valid_residues]
    valid_residue_names = residue_names[valid_residues]
    
    # 从刚体变换重建主链坐标
    from ..utils.geometry import reconstruct_backbone_from_rigid_transforms
    n_coords, ca_coords, c_coords, o_coords = reconstruct_backbone_from_rigid_transforms(
        valid_translations, valid_rotations
    )
    
    # 组合为单个张量 (num_valid_residues, 4, 3)
    backbone_coords = torch.stack([n_coords, ca_coords, c_coords, o_coords], dim=1)
    
    # 重建原子级数据
    all_coords = []
    all_atom_types = []
    all_residue_types = []
    all_chain_ids = []
    all_residue_numbers = []
    
    for res_idx in range(len(valid_translations)):
        res_backbone_coords = backbone_coords[res_idx]  # (4, 3)
        chain_id = valid_chain_ids[res_idx].item()
        residue_type = valid_residue_types[res_idx].item()
        residue_number = valid_residue_indices[res_idx].item()
        
        # 添加主链原子：N, CA, C, O
        for atom_pos in range(4):
            all_coords.append(res_backbone_coords[atom_pos])
            all_atom_types.append(atom_pos)  # 使用位置作为原子类型编号
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

def validate_frame_data(
    translations: torch.Tensor,
    rotations: torch.Tensor,
    res_mask: torch.Tensor,
    chain_ids: torch.Tensor,
    residue_types: torch.Tensor,
    residue_indices: torch.Tensor,
    chain_residue_indices: torch.Tensor,
    residue_names: torch.Tensor
) -> None:
    """
    验证 Frame 数据的一致性和有效性。
    
    Args:
        translations: 平移向量 (..., num_residues, 3)
        rotations: 旋转矩阵 (..., num_residues, 3, 3)
        res_mask: 残基掩码 (..., num_residues)
        chain_ids: 链标识符 (..., num_residues)
        residue_types: 残基类型 (..., num_residues)
        residue_indices: 全局残基编号 (..., num_residues)
        chain_residue_indices: 链内局部编号 (..., num_residues)
        residue_names: 残基名称张量 (..., num_residues)
        
    Raises:
        ValueError: 当数据不一致或无效时
    """
    logger.debug("验证 Frame 数据一致性")
    
    # 获取批量形状和残基数量
    batch_shape = translations.shape[:-2]
    num_residues = translations.shape[-2]
    
    # 验证基本形状
    expected_translations_shape = batch_shape + (num_residues, 3)
    expected_rotations_shape = batch_shape + (num_residues, 3, 3)
    expected_res_mask_shape = batch_shape + (num_residues,)
    expected_meta_shape = batch_shape + (num_residues,)
    
    if translations.shape != expected_translations_shape:
        raise ValueError(f"平移向量张量形状无效: {translations.shape}，期望 {expected_translations_shape}")
    
    if rotations.shape != expected_rotations_shape:
        raise ValueError(f"旋转矩阵张量形状无效: {rotations.shape}，期望 {expected_rotations_shape}")
    
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
    
    # 验证数据类型
    if not res_mask.dtype == torch.bool:
        raise ValueError(f"res_mask 必须是布尔类型，实际: {res_mask.dtype}")
    
    # 验证数值范围
    if torch.any(residue_types < 0) or torch.any(residue_types > 20):
        raise ValueError("残基类型索引超出有效范围 [0, 20]")
    
    if torch.any(residue_names < 0) or torch.any(residue_names > 20):
        raise ValueError("残基名称索引超出有效范围 [0, 20]")
    
    # 验证旋转矩阵的有效性
    from ..utils.geometry import validate_rotation_matrix
    validate_rotation_matrix(rotations, eps=1e-4)
    
    logger.debug("Frame 数据验证通过")


# ================================
# 工具函数
# ================================

def save_frame_to_cif(
    frame: Any,  # Frame 类型
    output_path: Union[str, Path],
    title: str = "ProtRepr Frame Structure"
) -> None:
    """
    将 Frame 数据保存为 CIF 文件。
    
    Args:
        frame: Frame 实例
        output_path: 输出文件路径
        title: 结构标题
    """
    logger.info(f"将 Frame 数据保存到 CIF 文件: {output_path}")
    protein_tensor = frame.to_protein_tensor()
    save_structure(protein_tensor, output_path, format_type="cif")
    logger.info(f"CIF 文件保存成功: {output_path}")


