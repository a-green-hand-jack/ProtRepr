"""
Frame 蛋白质表示数据。

本模块定义了 Frame 数据类，用于表示基于刚体坐标系的蛋白质表示法。该类封装了蛋白质结构的
frame 表示相关的所有数据和方法，支持与 ProteinTensor 的双向转换。

核心特性：
- 使用 @dataclass 装饰器，支持 frame.translations 风格的属性访问
- 存储每个残基的局部坐标系（旋转矩阵和平移向量）
- 存储完整的蛋白质元数据（链ID、残基类型、残基编号等）
- 双向转换：ProteinTensor ↔ Frame
- PyTorch 原生支持，GPU 加速计算
- 天然支持批量操作，所有张量支持任意 batch 维度
- 完整的验证和设备管理功能
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import torch
from protein_tensor import ProteinTensor
from ..representations.frame_converter import (
    protein_tensor_to_frame,
    frame_to_protein_tensor,
    validate_frame_data,
    save_frame_to_cif,
)

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """
    Frame 蛋白质表示数据类。

    该类封装了基于刚体坐标系的蛋白质结构表示，使用每个残基的旋转矩阵和平移向量
    来代替原子坐标表示。支持与 ProteinTensor 的双向转换，天然支持批量操作。

    Attributes:
        translations: 形状为 (..., num_residues, 3) 的平移向量张量（CA 原子坐标）
        rotations: 形状为 (..., num_residues, 3, 3) 的旋转矩阵张量（局部坐标系）
        res_mask: 形状为 (..., num_residues) 的布尔掩码张量
                  标识真实残基：1=标准氨基酸残基，0=非标准或缺失残基
        chain_ids: 形状为 (..., num_residues) 的链标识符张量
        residue_types: 形状为 (..., num_residues) 的残基类型张量（氨基酸编号）
        residue_indices: 形状为 (..., num_residues) 的残基在蛋白质中的全局编号
                        支持链间 gap，如 A链:1-100, B链:200-300
        chain_residue_indices: 形状为 (..., num_residues) 的残基在各自链中的局部编号
        residue_names: 形状为 (..., num_residues) 的残基名称张量（整数编码）

    Properties:
        device: 张量所在的设备
        batch_shape: 批量维度的形状
        num_residues: 残基数量
        num_chains: 链的数量

    Methods:
        from_protein_tensor: 从 ProteinTensor 创建 Frame 实例的类方法
        to_protein_tensor: 将 Frame 实例转换回 ProteinTensor 的实例方法
        to_device: 将所有张量移动到指定设备
        validate: 验证数据一致性
        get_chain_residues: 获取指定链的残基范围
        get_backbone_coords: 重建主链原子坐标
        get_local_coordinates: 获取标准局部坐标系中的原子位置
    """

    # 刚体变换数据
    translations: torch.Tensor  # (..., num_residues, 3) - CA 原子坐标
    rotations: torch.Tensor  # (..., num_residues, 3, 3) - 局部坐标系旋转矩阵
    res_mask: torch.Tensor  # (..., num_residues) - 1=标准残基, 0=非标准/缺失

    # 蛋白质元数据
    chain_ids: torch.Tensor  # (..., num_residues) - 链标识符
    residue_types: torch.Tensor  # (..., num_residues) - 残基类型编号
    residue_indices: torch.Tensor  # (..., num_residues) - 全局残基编号（支持链间gap）
    chain_residue_indices: torch.Tensor  # (..., num_residues) - 链内局部编号

    # 名称映射（张量化）
    residue_names: torch.Tensor  # (..., num_residues) - 残基名称编码

    # 可选的额外元数据
    b_factors: Optional[torch.Tensor] = None  # (..., num_residues) - B因子

    def __post_init__(self) -> None:
        """
        数据类初始化后的验证和处理。

        Raises:
            ValueError: 当数据形状或类型不符合要求时
        """
        logger.debug("初始化 Frame 数据类")

        # 获取批量形状和残基数量
        # translations 形状: (..., num_residues, 3)
        # 最后2个维度是: (num_residues, 3)
        batch_shape = self.translations.shape[:-2]  # 去掉最后两个维度
        num_residues = self.translations.shape[-2]  # 倒数第二个维度是 num_residues

        # 验证平移向量形状
        expected_translations_shape = batch_shape + (num_residues, 3)
        if self.translations.shape != expected_translations_shape:
            raise ValueError(
                f"平移向量张量形状无效: {self.translations.shape}，期望 {expected_translations_shape}"
            )

        # 验证旋转矩阵形状
        expected_rotations_shape = batch_shape + (num_residues, 3, 3)
        if self.rotations.shape != expected_rotations_shape:
            raise ValueError(
                f"旋转矩阵张量形状无效: {self.rotations.shape}，期望 {expected_rotations_shape}"
            )

        # 验证残基掩码形状
        expected_res_mask_shape = batch_shape + (num_residues,)
        if self.res_mask.shape != expected_res_mask_shape:
            raise ValueError(
                f"残基掩码张量形状无效: {self.res_mask.shape}，期望 {expected_res_mask_shape}"
            )

        # 验证元数据张量形状
        expected_meta_shape = batch_shape + (num_residues,)

        if self.chain_ids.shape != expected_meta_shape:
            raise ValueError(
                f"链ID张量形状无效: {self.chain_ids.shape}，期望 {expected_meta_shape}"
            )

        if self.residue_types.shape != expected_meta_shape:
            raise ValueError(
                f"残基类型张量形状无效: {self.residue_types.shape}，期望 {expected_meta_shape}"
            )

        if self.residue_indices.shape != expected_meta_shape:
            raise ValueError(
                f"残基索引张量形状无效: {self.residue_indices.shape}，期望 {expected_meta_shape}"
            )

        if self.chain_residue_indices.shape != expected_meta_shape:
            raise ValueError(
                f"链内残基索引张量形状无效: {self.chain_residue_indices.shape}，期望 {expected_meta_shape}"
            )

        if self.residue_names.shape != expected_meta_shape:
            raise ValueError(
                f"残基名称张量形状无效: {self.residue_names.shape}，期望 {expected_meta_shape}"
            )

        # 验证可选属性
        if self.b_factors is not None and self.b_factors.shape != expected_meta_shape:
            raise ValueError(
                f"B因子张量形状无效: {self.b_factors.shape}，期望 {expected_meta_shape}"
            )

    @classmethod
    def from_protein_tensor(
        cls, protein_tensor: ProteinTensor, device: Optional[torch.device] = None
    ) -> "Frame":
        """
        从 ProteinTensor 创建 Frame 实例。

        Args:
            protein_tensor: 输入的 ProteinTensor 对象，必须使用 torch 后端
            device: 目标设备，如果为 None 则使用输入张量的设备

        Returns:
            Frame: 新创建的 Frame 实例

        Raises:
            ValueError: 当 protein_tensor 未使用 torch 后端时
            TypeError: 当坐标数据不是 torch.Tensor 类型时
            RuntimeError: 当转换过程中出现错误时

        Example:
            >>> from protein_tensor import load_structure
            >>> protein_pt = load_structure("protein.pdb", backend='torch')
            >>> frame = Frame.from_protein_tensor(protein_pt)
            >>> print(f"Translations shape: {frame.translations.shape}")
            >>> print(f"Rotations shape: {frame.rotations.shape}")
        """
        logger.info("从 ProteinTensor 创建 Frame 实例")

        result = protein_tensor_to_frame(protein_tensor, device)
        (
            translations,
            rotations,
            res_mask,
            chain_ids,
            residue_types,
            residue_indices,
            chain_residue_indices,
            residue_names,
        ) = result

        return cls(
            translations=translations,
            rotations=rotations,
            res_mask=res_mask,
            chain_ids=chain_ids,
            residue_types=residue_types,
            residue_indices=residue_indices,
            chain_residue_indices=chain_residue_indices,
            residue_names=residue_names,
        )

    def to_protein_tensor(self) -> ProteinTensor:
        """
        将 Frame 实例转换为 ProteinTensor。

        Returns:
            ProteinTensor: 转换后的 ProteinTensor 对象，使用 torch 后端

        Raises:
            RuntimeError: 当转换过程中出现错误时

        Example:
            >>> frame = Frame.from_protein_tensor(protein_pt)
            >>> reconstructed_pt = frame.to_protein_tensor()
        """
        logger.info("将 Frame 转换为 ProteinTensor")

        return frame_to_protein_tensor(
            self.translations,
            self.rotations,
            self.res_mask,
            self.chain_ids,
            self.residue_types,
            self.residue_indices,
            self.chain_residue_indices,
            self.residue_names,
        )

    def to_device(self, device: torch.device) -> "Frame":
        """
        将所有张量移动到指定设备。

        Args:
            device: 目标设备

        Returns:
            Frame: 移动到新设备后的 Frame 实例
        """
        logger.debug(f"将 Frame 数据移动到设备: {device}")

        # 移动所有张量属性
        translations = self.translations.to(device)
        rotations = self.rotations.to(device)
        res_mask = self.res_mask.to(device)
        chain_ids = self.chain_ids.to(device)
        residue_types = self.residue_types.to(device)
        residue_indices = self.residue_indices.to(device)
        chain_residue_indices = self.chain_residue_indices.to(device)
        residue_names = self.residue_names.to(device)

        # 移动可选张量
        b_factors = self.b_factors.to(device) if self.b_factors is not None else None

        return Frame(
            translations=translations,
            rotations=rotations,
            res_mask=res_mask,
            chain_ids=chain_ids,
            residue_types=residue_types,
            residue_indices=residue_indices,
            chain_residue_indices=chain_residue_indices,
            residue_names=residue_names,
            b_factors=b_factors,
        )

    def validate(self) -> None:
        """
        验证 Frame 数据的一致性和有效性。

        Raises:
            ValueError: 当数据不一致或无效时
        """
        logger.debug("验证 Frame 数据一致性")

        validate_frame_data(
            self.translations,
            self.rotations,
            self.res_mask,
            self.chain_ids,
            self.residue_types,
            self.residue_indices,
            self.chain_residue_indices,
            self.residue_names,
        )

    @property
    def device(self) -> torch.device:
        """获取张量所在的设备。"""
        return self.translations.device

    @property
    def batch_shape(self) -> torch.Size:
        """获取批量维度的形状。"""
        return self.translations.shape[:-2]

    @property
    def num_residues(self) -> int:
        """获取残基数量。"""
        return self.translations.shape[-2]

    @property
    def num_chains(self) -> int:
        """获取链的数量。"""
        # 对于批量数据，返回最后一个 batch 的链数量
        if len(self.batch_shape) == 0:
            return len(torch.unique(self.chain_ids))
        else:
            # 对于批量数据，取最后一个样本
            last_sample_chain_ids = self.chain_ids[..., -1, :].flatten()
            return len(torch.unique(last_sample_chain_ids))

    def get_chain_residues(self, chain_id: Union[str, int]) -> torch.Tensor:
        """
        获取指定链的残基索引。

        Args:
            chain_id: 链标识符

        Returns:
            torch.Tensor: 属于该链的残基索引张量
        """
        if isinstance(chain_id, str):
            # 如果传入字符串，尝试转换为数字
            try:
                chain_id = int(chain_id)
            except ValueError:
                raise ValueError(f"无法将链ID '{chain_id}' 转换为数字")

        chain_mask = self.chain_ids == chain_id

        # 对于批量数据，返回所有批次中匹配的残基
        if len(self.batch_shape) == 0:
            residue_indices = torch.where(chain_mask)[0]
        else:
            # 对于批量数据，返回布尔掩码
            residue_indices = chain_mask

        return residue_indices

    def get_backbone_coords(self) -> torch.Tensor:
        """
        从 Frame 数据重建主链原子（N, CA, C, O）的坐标。

        Returns:
            torch.Tensor: 形状为 (..., num_residues, 4, 3) 的主链原子坐标
        """
        from ..utils.geometry import reconstruct_backbone_from_rigid_transforms

        # 重建主链坐标
        backbone_coords_tuple = reconstruct_backbone_from_rigid_transforms(
            self.translations, self.rotations
        )
        backbone_coords = torch.stack(backbone_coords_tuple, dim=-1)
        return backbone_coords

    def get_local_coordinates(self) -> Dict[str, torch.Tensor]:
        """
        获取标准局部坐标系中的原子位置。

        Returns:
            Dict[str, torch.Tensor]: 包含各原子在局部坐标系中标准位置的字典
        """
        from ..utils.geometry import STANDARD_BACKBONE_GEOMETRY
        import math

        device = self.device
        batch_shape = self.batch_shape
        num_residues = self.num_residues

        # 标准主链几何参数
        ca_n_length = STANDARD_BACKBONE_GEOMETRY["CA_N_BOND_LENGTH"]
        ca_c_length = STANDARD_BACKBONE_GEOMETRY["CA_C_BOND_LENGTH"]
        c_o_length = STANDARD_BACKBONE_GEOMETRY["C_O_BOND_LENGTH"]
        n_ca_c_angle = math.radians(STANDARD_BACKBONE_GEOMETRY["N_CA_C_ANGLE"])
        ca_c_o_angle = math.radians(STANDARD_BACKBONE_GEOMETRY["CA_C_O_ANGLE"])

        # CA 原子在局部坐标系原点
        ca_local = torch.zeros(*batch_shape, num_residues, 3, device=device)

        # C 原子在 x 轴正方向
        c_local = torch.zeros(*batch_shape, num_residues, 3, device=device)
        c_local[..., 0] = ca_c_length

        # N 原子在 x-y 平面内，考虑键角
        n_x = -ca_n_length * math.cos(math.pi - n_ca_c_angle)
        n_y = ca_n_length * math.sin(math.pi - n_ca_c_angle)
        n_local = torch.zeros(*batch_shape, num_residues, 3, device=device)
        n_local[..., 0] = n_x
        n_local[..., 1] = n_y

        # O 原子：从 C 原子出发，考虑 CA-C-O 键角
        o_direction_x = math.cos(math.pi - ca_c_o_angle)  # 相对于 CA-C 方向
        o_direction_y = math.sin(math.pi - ca_c_o_angle)  # 垂直分量

        o_local = torch.zeros(*batch_shape, num_residues, 3, device=device)
        o_local[..., 0] = ca_c_length + c_o_length * o_direction_x
        o_local[..., 1] = c_o_length * o_direction_y

        return {"N": n_local, "CA": ca_local, "C": c_local, "O": o_local}

    def save(self, filepath: Union[str, Path], save_as_instance: bool = True) -> None:
        """
        保存 Frame 数据到文件。

        Args:
            filepath: 保存路径，推荐使用 .pt 扩展名
            save_as_instance: 如果为 True，保存完整的 Frame 实例；
                            如果为 False，保存为字典格式
        """
        filepath = Path(filepath)

        if save_as_instance:
            # 直接保存 Frame 实例
            torch.save(self, filepath)
            logger.info(f"Frame 实例已保存到: {filepath}")
        else:
            # 保存为字典格式
            data_dict = {
                "translations": self.translations,
                "rotations": self.rotations,
                "res_mask": self.res_mask,
                "chain_ids": self.chain_ids,
                "residue_types": self.residue_types,
                "residue_indices": self.residue_indices,
                "chain_residue_indices": self.chain_residue_indices,
                "residue_names": self.residue_names,
                "metadata": {
                    "format": "frame_dict",
                    "version": "1.0",
                    "num_residues": self.num_residues,
                    "num_chains": self.num_chains,
                    "device": str(self.device),
                },
            }
            torch.save(data_dict, filepath)
            logger.info(f"Frame 字典已保存到: {filepath}")

    @classmethod
    def load(
        cls, filepath: Union[str, Path], map_location: Optional[str] = None
    ) -> "Frame":
        """
        从文件加载 Frame 数据。

        Args:
            filepath: 文件路径
            map_location: 设备映射位置，如 'cpu', 'cuda' 等

        Returns:
            Frame: 加载的 Frame 实例

        Raises:
            ValueError: 如果文件格式不正确
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        # 加载数据
        data = torch.load(filepath, map_location=map_location, weights_only=False)

        # 判断是实例还是字典
        if isinstance(data, cls):
            # 直接是 Frame 实例
            logger.info(f"从 {filepath} 加载 Frame 实例")
            return data
        elif isinstance(data, dict):
            # 是字典格式，需要重构实例
            if "metadata" in data and data["metadata"].get("format") == "frame_dict":
                # 标准的 Frame 字典格式
                logger.info(f"从 {filepath} 加载 Frame 字典并重构实例")
                return cls(
                    translations=data["translations"],
                    rotations=data["rotations"],
                    res_mask=data["res_mask"],
                    chain_ids=data["chain_ids"],
                    residue_types=data["residue_types"],
                    residue_indices=data["residue_indices"],
                    chain_residue_indices=data["chain_residue_indices"],
                    residue_names=data["residue_names"],
                )
            else:
                # 尝试从通用字典格式重构
                logger.warning(f"从 {filepath} 加载的字典格式不标准，尝试重构")
                return cls(
                    translations=data["translations"],
                    rotations=data["rotations"],
                    res_mask=data["res_mask"],
                    chain_ids=data["chain_ids"],
                    residue_types=data["residue_types"],
                    residue_indices=data["residue_indices"],
                    chain_residue_indices=data["chain_residue_indices"],
                    residue_names=data["residue_names"],
                )
        else:
            raise ValueError(f"无法识别的文件格式: {type(data)}")

    def to_cif(self, output_path: str) -> None:
        """
        将 Frame 数据转换并保存为 CIF 文件。

        Args:
            output_path: 输出 CIF 文件路径
        """
        save_frame_to_cif(self, output_path)
        logger.info(f"Frame 数据已转换并保存为 CIF 文件: {output_path}")
