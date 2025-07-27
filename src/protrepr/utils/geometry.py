"""
几何计算工具模块

本模块提供蛋白质结构分析中常用的几何计算功能，包括：
- 向量操作：标准化、点积、叉积
- 角度计算：键角、二面角
- 距离计算：原子间距离、残基间距离
- 坐标变换：旋转、平移、刚体变换

所有计算都基于 PyTorch 张量，支持 GPU 加速和自动微分，
确保与深度学习工作流的无缝集成。
"""

import logging
from typing import Tuple, Optional

import torch
import math

logger = logging.getLogger(__name__)

# 标准氨基酸主链几何参数（来自晶体结构统计）
STANDARD_BACKBONE_GEOMETRY = {
    # 键长 (Å)
    "CA_N_BOND_LENGTH": 1.458,    # CA-N 键长
    "CA_C_BOND_LENGTH": 1.525,    # CA-C 键长  
    "C_O_BOND_LENGTH": 1.229,     # C=O 键长
    "N_CA_BOND_LENGTH": 1.458,    # N-CA 键长
    
    # 键角 (度)
    "N_CA_C_ANGLE": 111.2,        # N-CA-C 键角
    "CA_C_O_ANGLE": 120.8,        # CA-C-O 键角
    "C_N_CA_ANGLE": 121.7,        # C-N-CA 键角 (trans peptide)
    
    # 二面角 (度) - 反式肽键的理想值
    "OMEGA_ANGLE": 180.0,         # omega 二面角 (CA-C-N-CA)
}

# 数值计算常量
NUMERICAL_CONSTANTS = {
    "EPSILON": 1e-8,              # 数值稳定性常数
    "ANGLE_TOLERANCE": 1e-4,      # 角度计算容差
    "DETERMINANT_TOLERANCE": 1e-4, # 行列式验证容差
}


def normalize_vectors(
    vectors: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    标准化向量到单位长度。
    
    Args:
        vectors: 形状为 (..., 3) 的向量张量
        dim: 标准化的维度，默认为最后一维
        eps: 数值稳定性的小常数
        
    Returns:
        torch.Tensor: 标准化后的单位向量
        
    Raises:
        ValueError: 当输入向量包含零向量时
    """
    logger.debug("标准化向量")
    
    # 计算向量的模长
    norms = torch.norm(vectors, dim=dim, keepdim=True)
    
    # 检查是否有零向量
    if torch.any(norms < eps):
        logger.warning("检测到接近零的向量，可能导致数值不稳定")
    
    # 标准化，添加 eps 避免除零
    normalized = vectors / (norms + eps)
    
    return normalized


def cross_product(
    vectors1: torch.Tensor,
    vectors2: torch.Tensor
) -> torch.Tensor:
    """
    计算两组向量的叉积。
    
    Args:
        vectors1: 形状为 (..., 3) 的第一组向量
        vectors2: 形状为 (..., 3) 的第二组向量
        
    Returns:
        torch.Tensor: 叉积结果，形状与输入相同
        
    Raises:
        ValueError: 当输入向量形状不匹配或不是3维向量时
    """
    logger.debug("计算向量叉积")
    
    # 验证输入形状
    if vectors1.shape != vectors2.shape:
        raise ValueError(f"向量形状不匹配: {vectors1.shape} vs {vectors2.shape}")
    
    if vectors1.shape[-1] != 3:
        raise ValueError(f"输入必须是3维向量，但得到形状: {vectors1.shape}")
    
    # 使用 PyTorch 的叉积函数
    cross = torch.cross(vectors1, vectors2, dim=-1)
    
    return cross


def dot_product(
    vectors1: torch.Tensor,
    vectors2: torch.Tensor
) -> torch.Tensor:
    """
    计算两组向量的点积。
    
    Args:
        vectors1: 形状为 (..., 3) 的第一组向量
        vectors2: 形状为 (..., 3) 的第二组向量
        
    Returns:
        torch.Tensor: 点积结果，形状为 (...)
        
    Raises:
        ValueError: 当输入向量形状不匹配时
    """
    logger.debug("计算向量点积")
    
    # 验证输入形状
    if vectors1.shape != vectors2.shape:
        raise ValueError(f"向量形状不匹配: {vectors1.shape} vs {vectors2.shape}")
    
    # 计算点积
    dot_prod = torch.sum(vectors1 * vectors2, dim=-1)
    
    return dot_prod


def gram_schmidt_orthogonalization(
    v1: torch.Tensor,
    v2: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    执行 Gram-Schmidt 正交化过程构建旋转矩阵。
    
    基于两个输入向量构建标准正交基，生成 3x3 旋转矩阵。
    这是构建残基局部坐标系的核心算法。
    
    Args:
        v1: 形状为 (..., 3) 的第一个向量 (通常是 CA->C)
        v2: 形状为 (..., 3) 的第二个向量 (通常是 CA->N)
        eps: 数值稳定性的小常数
        
    Returns:
        torch.Tensor: 形状为 (..., 3, 3) 的旋转矩阵
        
    Raises:
        ValueError: 当输入向量形状不匹配时
        RuntimeError: 当向量共线导致正交化失败时
        
    Notes:
        正交化步骤：
        1. e1 = normalize(v1)
        2. u2 = v2 - (v2·e1)e1  # 去除 v2 在 e1 方向的分量
        3. e2 = normalize(u2)
        4. e3 = e1 × e2         # 叉积确保右手坐标系
    """
    logger.debug("执行 Gram-Schmidt 正交化")
    
    # 验证输入形状
    if v1.shape != v2.shape:
        raise ValueError(f"向量形状不匹配: {v1.shape} vs {v2.shape}")
    
    if v1.shape[-1] != 3:
        raise ValueError(f"输入必须是3维向量，但得到形状: {v1.shape}")
    
    # 1. 标准化第一个向量作为第一个基向量 (x轴)
    e1 = normalize_vectors(v1, eps=eps)
    
    # 2. 将第二个向量正交化到第一个向量
    # u2 = v2 - proj_e1(v2) = v2 - (v2·e1)e1
    proj_v2_on_e1 = dot_product(v2, e1).unsqueeze(-1) * e1
    u2 = v2 - proj_v2_on_e1
    
    # 检查正交化是否成功
    u2_norm = torch.norm(u2, dim=-1)
    if torch.any(u2_norm < eps):
        logger.warning("检测到共线向量，可能导致正交化失败")
    
    # 3. 标准化得到第二个基向量
    e2 = normalize_vectors(u2, eps=eps)
    
    # 4. 通过叉积计算第三个基向量，确保右手坐标系
    e3 = cross_product(e1, e2)
    e3 = normalize_vectors(e3, eps=eps)  # 确保单位长度
    
    # 5. 构建旋转矩阵：[e1, e2, e3] 作为列向量
    # 形状：(..., 3, 3)
    rotation_matrix = torch.stack([e1, e2, e3], dim=-1)
    
    return rotation_matrix


def compute_rigid_transforms_from_backbone(
    n_coords: torch.Tensor,
    ca_coords: torch.Tensor,
    c_coords: torch.Tensor,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于主链原子坐标计算刚体变换。
    
    使用 N, CA, C 原子的坐标通过 Gram-Schmidt 正交化过程构建标准的
    局部坐标系。这是 AlphaFold 和许多 SE(3)-equivariant 网络的标准做法。
    
    Args:
        n_coords: 形状为 (..., 3) 的 N 原子坐标
        ca_coords: 形状为 (..., 3) 的 CA 原子坐标
        c_coords: 形状为 (..., 3) 的 C 原子坐标
        eps: 数值稳定性的小常数
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - translations: 形状为 (..., 3) 的平移向量 (CA 坐标)
            - rotations: 形状为 (..., 3, 3) 的旋转矩阵
            
    Raises:
        ValueError: 当输入坐标形状不匹配时
        RuntimeError: 当正交化过程失败时
        
    Notes:
        局部坐标系的构建规则：
        - x 轴：从 CA 指向 C 的标准化向量
        - 临时轴：从 CA 指向 N 的向量
        - y 轴：通过正交化得到，在 CA-N, CA-C 平面内
        - z 轴：通过 x 和 y 向量的叉积确定，确保右手坐标系
    """
    logger.debug("计算基于主链原子的刚体变换")
    
    # 验证输入坐标的形状一致性
    if not (n_coords.shape == ca_coords.shape == c_coords.shape):
        raise ValueError(f"坐标形状不匹配: N={n_coords.shape}, CA={ca_coords.shape}, C={c_coords.shape}")
    
    if n_coords.shape[-1] != 3:
        raise ValueError(f"坐标必须是3维，但得到形状: {n_coords.shape}")
    
    # 1. 平移向量：CA 原子坐标
    translations = ca_coords.clone()
    
    # 2. 计算方向向量
    ca_to_c = c_coords - ca_coords  # CA -> C 方向 (x轴)
    ca_to_n = n_coords - ca_coords  # CA -> N 方向 (用于正交化)
    
    # 检查向量长度
    ca_to_c_norm = torch.norm(ca_to_c, dim=-1)
    ca_to_n_norm = torch.norm(ca_to_n, dim=-1)
    
    if torch.any(ca_to_c_norm < eps) or torch.any(ca_to_n_norm < eps):
        logger.warning("检测到异常短的主链键长，可能导致数值不稳定")
    
    # 3. 使用 Gram-Schmidt 正交化构建旋转矩阵
    rotations = gram_schmidt_orthogonalization(ca_to_c, ca_to_n, eps=eps)
    
    # 4. 验证旋转矩阵的有效性
    validate_rotation_matrix(rotations, eps=1e-4)
    
    return translations, rotations


def validate_rotation_matrix(
    rotations: torch.Tensor,
    eps: float = 1e-4
) -> None:
    """
    验证旋转矩阵的有效性。
    
    Args:
        rotations: 形状为 (..., 3, 3) 的旋转矩阵张量
        eps: 验证容差
        
    Raises:
        ValueError: 当旋转矩阵无效时
    """
    logger.debug("验证旋转矩阵的有效性")
    
    # 验证行列式接近 1
    det = torch.det(rotations)
    det_error = torch.abs(det - 1.0)
    
    if torch.any(det_error > eps):
        max_error = torch.max(det_error).item()
        logger.warning(f"检测到旋转矩阵行列式偏差: 最大误差 = {max_error:.8e}")
        
        if max_error > 100 * eps:  # 严重错误
            raise ValueError(f"旋转矩阵行列式严重偏离1: 最大误差 = {max_error:.8e}")
    
    # 验证正交性：R @ R.T ≈ I
    batch_shape = rotations.shape[:-2]
    identity = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
    if batch_shape:
        identity = identity.expand(*batch_shape, 3, 3)
    
    orthogonality_check = torch.bmm(
        rotations.view(-1, 3, 3),
        rotations.transpose(-1, -2).view(-1, 3, 3)
    ).view(*batch_shape, 3, 3)
    
    orthogonality_error = torch.max(torch.abs(orthogonality_check - identity))
    
    if orthogonality_error > eps:
        logger.warning(f"检测到旋转矩阵正交性偏差: 最大误差 = {orthogonality_error:.8e}")
        
        if orthogonality_error > 100 * eps:  # 严重错误
            raise ValueError(f"旋转矩阵正交性严重偏差: 最大误差 = {orthogonality_error:.8e}")


def reconstruct_backbone_from_rigid_transforms(
    translations: torch.Tensor,
    rotations: torch.Tensor,
    residue_names: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    使用刚体变换重建主链原子坐标。
    
    基于旋转矩阵和平移向量，使用标准的主链几何参数重建
    N, CA, C, O 原子的坐标。
    
    Args:
        translations: 形状为 (..., 3) 的平移向量 (CA 坐标)
        rotations: 形状为 (..., 3, 3) 的旋转矩阵
        residue_names: 可选的残基名称，暂时未使用 (默认丙氨酸)
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - n_coords: N 原子坐标, shape: (..., 3)
            - ca_coords: CA 原子坐标, shape: (..., 3)  
            - c_coords: C 原子坐标, shape: (..., 3)
            - o_coords: O 原子坐标, shape: (..., 3)
            
    Notes:
        使用标准的主链几何参数：
        - CA-N 键长: 1.458 Å
        - CA-C 键长: 1.525 Å  
        - C=O 键长: 1.229 Å
        - N-CA-C 键角: 111.2°
        - CA-C-O 键角: 120.8°
    """
    logger.debug("使用刚体变换重建主链原子坐标")
    
    # 验证输入形状
    if translations.shape[:-1] != rotations.shape[:-2]:
        raise ValueError(f"平移和旋转的批次形状不匹配: {translations.shape} vs {rotations.shape}")
    
    batch_shape = translations.shape[:-1]
    device = translations.device
    dtype = translations.dtype
    
    # 标准主链几何参数
    ca_n_length = STANDARD_BACKBONE_GEOMETRY["CA_N_BOND_LENGTH"]
    ca_c_length = STANDARD_BACKBONE_GEOMETRY["CA_C_BOND_LENGTH"] 
    c_o_length = STANDARD_BACKBONE_GEOMETRY["C_O_BOND_LENGTH"]
    n_ca_c_angle = math.radians(STANDARD_BACKBONE_GEOMETRY["N_CA_C_ANGLE"])
    ca_c_o_angle = math.radians(STANDARD_BACKBONE_GEOMETRY["CA_C_O_ANGLE"])
    
    # 1. CA 原子坐标就是平移向量
    ca_coords = translations
    
    # 2. 在局部坐标系中定义标准原子位置
    # 局部坐标系：x轴沿CA->C，y轴在CA-N-C平面内，z轴垂直平面
    
    # C 原子：沿x轴正方向
    c_local = torch.tensor([ca_c_length, 0.0, 0.0], device=device, dtype=dtype)
    c_local = c_local.expand(*batch_shape, 3)
    
    # N 原子：在x-y平面内，考虑键角
    # N-CA-C 键角，N在x轴负方向，与C形成指定角度
    n_x = -ca_n_length * math.cos(math.pi - n_ca_c_angle)  # x分量 (负值，因为N在C的对面)  
    n_y = ca_n_length * math.sin(math.pi - n_ca_c_angle)   # y分量
    n_local = torch.tensor([n_x, n_y, 0.0], device=device, dtype=dtype)
    n_local = n_local.expand(*batch_shape, 3)
    
    # O 原子：沿CA-C方向延伸，然后偏转CA-C-O键角
    # 在x-z平面内，因为O通常偏出CA-N-C平面
    o_x = ca_c_length + c_o_length * math.cos(math.pi - ca_c_o_angle)
    o_z = c_o_length * math.sin(math.pi - ca_c_o_angle)
    o_local = torch.tensor([o_x, 0.0, o_z], device=device, dtype=dtype)
    o_local = o_local.expand(*batch_shape, 3)
    
    # 3. 应用刚体变换：将局部坐标转换为全局坐标
    # global_coord = rotation @ local_coord + translation
    
    # 为了批处理，需要reshape
    flat_rotations = rotations.view(-1, 3, 3)
    flat_translations = translations.view(-1, 3)
    
    # N 原子
    n_local_flat = n_local.view(-1, 3, 1)  # 添加维度用于矩阵乘法
    n_global_flat = torch.bmm(flat_rotations, n_local_flat).squeeze(-1) + flat_translations
    n_coords = n_global_flat.view(*batch_shape, 3)
    
    # C 原子
    c_local_flat = c_local.view(-1, 3, 1)
    c_global_flat = torch.bmm(flat_rotations, c_local_flat).squeeze(-1) + flat_translations
    c_coords = c_global_flat.view(*batch_shape, 3)
    
    # O 原子
    o_local_flat = o_local.view(-1, 3, 1)
    o_global_flat = torch.bmm(flat_rotations, o_local_flat).squeeze(-1) + flat_translations
    o_coords = o_global_flat.view(*batch_shape, 3)
    
    return n_coords, ca_coords, c_coords, o_coords 


def compute_distances(
    coords1: torch.Tensor,
    coords2: torch.Tensor
) -> torch.Tensor:
    """
    计算两组坐标之间的欧几里得距离。
    
    Args:
        coords1: 形状为 (..., 3) 的第一组坐标
        coords2: 形状为 (..., 3) 的第二组坐标
        
    Returns:
        torch.Tensor: 距离标量，形状为 (...)
        
    Raises:
        ValueError: 当输入坐标形状不匹配时
    """
    logger.debug("计算坐标间距离")
    
    # TODO: 实现距离计算
    # 使用公式：distance = sqrt(sum((coords1 - coords2)^2))
    
    raise NotImplementedError("距离计算功能尚未实现")


def compute_bond_angles(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    coords3: torch.Tensor
) -> torch.Tensor:
    """
    计算三个原子形成的键角。
    
    计算以 coords2 为顶点，coords1-coords2-coords3 形成的键角。
    
    Args:
        coords1: 形状为 (..., 3) 的第一个原子坐标
        coords2: 形状为 (..., 3) 的顶点原子坐标
        coords3: 形状为 (..., 3) 的第三个原子坐标
        
    Returns:
        torch.Tensor: 键角（弧度），形状为 (...)
        
    Raises:
        ValueError: 当输入坐标形状不匹配时
        RuntimeError: 当计算过程中出现数值问题时
    """
    logger.debug("计算键角")
    
    # TODO: 实现键角计算
    # 1. 计算向量 v1 = coords1 - coords2
    # 2. 计算向量 v2 = coords3 - coords2
    # 3. 使用点积公式计算夹角：angle = arccos(v1·v2 / (|v1||v2|))
    
    raise NotImplementedError("键角计算功能尚未实现")


def compute_dihedral_angles(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    coords3: torch.Tensor,
    coords4: torch.Tensor
) -> torch.Tensor:
    """
    计算四个原子形成的二面角。
    
    计算由四个连续原子 coords1-coords2-coords3-coords4 定义的二面角。
    这在分析蛋白质的 phi/psi 角度时非常重要。
    
    Args:
        coords1: 形状为 (..., 3) 的第一个原子坐标
        coords2: 形状为 (..., 3) 的第二个原子坐标
        coords3: 形状为 (..., 3) 的第三个原子坐标
        coords4: 形状为 (..., 3) 的第四个原子坐标
        
    Returns:
        torch.Tensor: 二面角（弧度），形状为 (...)，范围 [-π, π]
        
    Raises:
        ValueError: 当输入坐标形状不匹配时
        RuntimeError: 当计算过程中出现数值问题时
    """
    logger.debug("计算二面角")
    
    # TODO: 实现二面角计算
    # 1. 计算三个向量：v1 = coords2 - coords1, v2 = coords3 - coords2, v3 = coords4 - coords3
    # 2. 计算两个法向量：n1 = v1 × v2, n2 = v2 × v3
    # 3. 使用公式计算二面角：angle = atan2((v2·(n1×n2))/|v2|, n1·n2)
    
    raise NotImplementedError("二面角计算功能尚未实现")


def rodrigues_rotation(
    vectors: torch.Tensor,
    axes: torch.Tensor,
    angles: torch.Tensor
) -> torch.Tensor:
    """
    使用罗德里格斯公式执行向量绕轴旋转。
    
    这是一个高效的 3D 旋转算法，常用于蛋白质结构的几何变换。
    
    Args:
        vectors: 形状为 (..., 3) 的输入向量
        axes: 形状为 (..., 3) 的旋转轴单位向量
        angles: 形状为 (...,) 的旋转角度（弧度）
        
    Returns:
        torch.Tensor: 旋转后的向量，形状与输入相同
        
    Raises:
        ValueError: 当输入张量形状不匹配时
        RuntimeError: 当旋转轴不是单位向量时
    """
    logger.debug("执行罗德里格斯旋转")
    
    # TODO: 实现罗德里格斯旋转公式
    # v_rot = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
    # 其中 v 是输入向量，k 是旋转轴，θ 是旋转角度
    
    raise NotImplementedError("罗德里格斯旋转功能尚未实现")


def compute_rmsd(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    计算两组坐标之间的均方根偏差 (RMSD)。
    
    RMSD 是衡量蛋白质结构相似性的重要指标。
    
    Args:
        coords1: 形状为 (..., N, 3) 的第一组坐标
        coords2: 形状为 (..., N, 3) 的第二组坐标
        mask: 可选的掩码，形状为 (..., N)，标识参与计算的原子
        
    Returns:
        torch.Tensor: RMSD 值，形状为 (...)
        
    Raises:
        ValueError: 当输入坐标形状不匹配时
    """
    logger.debug("计算RMSD")
    
    # TODO: 实现RMSD计算
    # 1. 计算坐标差异的平方
    # 2. 应用掩码（如果提供）
    # 3. 计算均值的平方根
    
    raise NotImplementedError("RMSD计算功能尚未实现")


def kabsch_alignment(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 Kabsch 算法计算最优的刚体变换对齐两组坐标。
    
    Kabsch 算法是蛋白质结构对比中的经典方法，用于找到最小化 RMSD 的
    最优旋转和平移变换。
    
    Args:
        coords1: 形状为 (..., N, 3) 的参考坐标
        coords2: 形状为 (..., N, 3) 的目标坐标
        mask: 可选的掩码，形状为 (..., N)，标识参与对齐的原子
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - rotation: 形状为 (..., 3, 3) 的最优旋转矩阵
            - translation: 形状为 (..., 3) 的最优平移向量
            
    Raises:
        ValueError: 当输入坐标形状不匹配时
        RuntimeError: 当 SVD 分解失败时
    """
    logger.debug("执行Kabsch对齐")
    
    # TODO: 实现Kabsch算法
    # 1. 中心化坐标（去除质心）
    # 2. 计算协方差矩阵
    # 3. SVD分解获得最优旋转矩阵
    # 4. 计算相应的平移向量
    
    raise NotImplementedError("Kabsch对齐功能尚未实现") 