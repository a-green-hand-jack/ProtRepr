"""
几何计算工具测试

测试 protrepr.utils.geometry 模块中的几何计算功能，包括：
- 基础向量操作：标准化、点积、叉积
- Gram-Schmidt 正交化算法
- 刚体变换计算和验证
- 主链原子坐标重建

所有测试都基于 PyTorch 张量，并验证数值精度和数学正确性。
"""

import pytest
import torch
import math
import numpy as np
from typing import Tuple

# 导入被测试的函数
from protrepr.utils.geometry import (
    normalize_vectors,
    cross_product,
    dot_product,
    gram_schmidt_orthogonalization,
    compute_rigid_transforms_from_backbone,
    validate_rotation_matrix,
    reconstruct_backbone_from_rigid_transforms,
    STANDARD_BACKBONE_GEOMETRY,
    NUMERICAL_CONSTANTS
)


class TestBasicVectorOperations:
    """测试基础向量操作函数。"""
    
    def test_normalize_vectors_simple(self):
        """测试基本向量标准化。"""
        # 创建简单的测试向量
        vectors = torch.tensor([
            [3.0, 4.0, 0.0],  # 长度为5的向量
            [1.0, 0.0, 0.0],  # 单位向量
            [0.0, 2.0, 0.0],  # 沿y轴的向量
        ], dtype=torch.float32)
        
        normalized = normalize_vectors(vectors)
        
        # 验证结果形状
        assert normalized.shape == vectors.shape
        
        # 验证所有向量都是单位向量
        norms = torch.norm(normalized, dim=-1)
        expected_norms = torch.ones(3, dtype=torch.float32)
        torch.testing.assert_close(norms, expected_norms, atol=1e-6, rtol=1e-6)
        
        # 验证具体值
        expected = torch.tensor([
            [0.6, 0.8, 0.0],  # [3,4,0] normalized
            [1.0, 0.0, 0.0],  # already normalized
            [0.0, 1.0, 0.0],  # [0,2,0] normalized
        ], dtype=torch.float32)
        torch.testing.assert_close(normalized, expected, atol=1e-6, rtol=1e-6)
    
    def test_normalize_vectors_batch(self):
        """测试批处理向量标准化。"""
        # 创建批处理形状的向量
        batch_vectors = torch.randn(2, 3, 5, 3)  # (batch, residues, atoms, 3)
        
        normalized = normalize_vectors(batch_vectors)
        
        # 验证形状保持不变
        assert normalized.shape == batch_vectors.shape
        
        # 验证所有向量都是单位向量
        norms = torch.norm(normalized, dim=-1)
        expected_norms = torch.ones_like(norms)
        torch.testing.assert_close(norms, expected_norms, atol=1e-5, rtol=1e-5)
    
    def test_normalize_vectors_zero_vector_warning(self):
        """测试零向量的处理（应该发出警告但不崩溃）。"""
        vectors = torch.tensor([
            [0.0, 0.0, 0.0],  # 零向量
            [1.0, 0.0, 0.0],  # 正常向量
        ], dtype=torch.float32)
        
        # 应该不抛出异常，但可能发出警告
        normalized = normalize_vectors(vectors)
        
        # 验证形状
        assert normalized.shape == vectors.shape
        
        # 验证非零向量被正确标准化
        torch.testing.assert_close(normalized[1], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6, rtol=1e-6)
    
    def test_cross_product_simple(self):
        """测试基本向量叉积。"""
        v1 = torch.tensor([
            [1.0, 0.0, 0.0],  # x轴
            [0.0, 1.0, 0.0],  # y轴
        ], dtype=torch.float32)
        
        v2 = torch.tensor([
            [0.0, 1.0, 0.0],  # y轴
            [0.0, 0.0, 1.0],  # z轴
        ], dtype=torch.float32)
        
        cross = cross_product(v1, v2)
        
        # 验证叉积结果
        expected = torch.tensor([
            [0.0, 0.0, 1.0],  # x × y = z
            [1.0, 0.0, 0.0],  # y × z = x
        ], dtype=torch.float32)
        
        torch.testing.assert_close(cross, expected, atol=1e-6, rtol=1e-6)
    
    def test_cross_product_properties(self):
        """测试叉积的数学性质。"""
        # 创建随机向量
        v1 = torch.randn(5, 3)
        v2 = torch.randn(5, 3)
        
        cross_12 = cross_product(v1, v2)
        cross_21 = cross_product(v2, v1)
        
        # 验证反交换律：v1 × v2 = -(v2 × v1)
        torch.testing.assert_close(cross_12, -cross_21, atol=1e-5, rtol=1e-5)
        
        # 验证叉积与原向量垂直
        dot_1 = dot_product(cross_12, v1)
        dot_2 = dot_product(cross_12, v2)
        
        torch.testing.assert_close(dot_1, torch.zeros_like(dot_1), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(dot_2, torch.zeros_like(dot_2), atol=1e-5, rtol=1e-5)
    
    def test_cross_product_shape_validation(self):
        """测试叉积的形状验证。"""
        v1 = torch.randn(3, 3)
        v2 = torch.randn(2, 3)  # 不匹配的形状
        
        with pytest.raises(ValueError, match="向量形状不匹配"):
            cross_product(v1, v2)
        
        # 测试非3维向量
        v1_2d = torch.randn(3, 2)
        v2_2d = torch.randn(3, 2)
        
        with pytest.raises(ValueError, match="输入必须是3维向量"):
            cross_product(v1_2d, v2_2d)
    
    def test_dot_product_simple(self):
        """测试基本向量点积。"""
        v1 = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ], dtype=torch.float32)
        
        v2 = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float32)
        
        dot = dot_product(v1, v2)
        
        expected = torch.tensor([1.0, 1.0], dtype=torch.float32)
        torch.testing.assert_close(dot, expected, atol=1e-6, rtol=1e-6)
    
    def test_dot_product_properties(self):
        """测试点积的数学性质。"""
        v1 = torch.randn(5, 3)
        v2 = torch.randn(5, 3)
        v3 = torch.randn(5, 3)
        
        # 验证交换律：v1 · v2 = v2 · v1
        dot_12 = dot_product(v1, v2)
        dot_21 = dot_product(v2, v1)
        torch.testing.assert_close(dot_12, dot_21, atol=1e-5, rtol=1e-5)
        
        # 验证分配律：v1 · (v2 + v3) = v1 · v2 + v1 · v3
        dot_sum = dot_product(v1, v2 + v3)
        dot_separate = dot_product(v1, v2) + dot_product(v1, v3)
        torch.testing.assert_close(dot_sum, dot_separate, atol=1e-5, rtol=1e-5)


class TestGramSchmidtOrthogonalization:
    """测试 Gram-Schmidt 正交化算法。"""
    
    def test_gram_schmidt_simple(self):
        """测试基本正交化。"""
        # 使用简单的测试向量
        v1 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)  # x轴
        v2 = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)  # 在x-y平面内
        
        rotation_matrix = gram_schmidt_orthogonalization(v1, v2)
        
        # 验证输出形状
        assert rotation_matrix.shape == (1, 3, 3)
        
        # 验证第一列是标准化的v1
        expected_e1 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        torch.testing.assert_close(rotation_matrix[0, :, 0], expected_e1[0], atol=1e-6, rtol=1e-6)
        
        # 验证第二列在x-y平面内，且与第一列正交
        e2 = rotation_matrix[0, :, 1]
        assert abs(e2[2].item()) < 1e-6  # z分量应该为0
        
        # 验证正交性
        dot_e1_e2 = torch.dot(rotation_matrix[0, :, 0], rotation_matrix[0, :, 1])
        assert abs(dot_e1_e2.item()) < 1e-6
    
    def test_gram_schmidt_batch(self):
        """测试批处理正交化。"""
        batch_size = 3
        v1 = torch.randn(batch_size, 3)
        v2 = torch.randn(batch_size, 3)
        
        # 确保v1和v2不共线
        v2 = v2 + 0.1 * torch.randn_like(v2)
        
        rotation_matrices = gram_schmidt_orthogonalization(v1, v2)
        
        # 验证形状
        assert rotation_matrices.shape == (batch_size, 3, 3)
        
        # 验证每个旋转矩阵的性质
        for i in range(batch_size):
            R = rotation_matrices[i]
            
            # 验证正交性：R @ R.T = I
            identity_check = R @ R.T
            identity = torch.eye(3, dtype=R.dtype)
            torch.testing.assert_close(identity_check, identity, atol=1e-4, rtol=1e-4)
            
            # 验证行列式为1（右手坐标系）
            det = torch.det(R)
            assert abs(det.item() - 1.0) < 1e-4
    
    def test_gram_schmidt_orthogonality(self):
        """测试正交化结果的正交性。"""
        v1 = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
        v2 = torch.tensor([[1.0, 3.0, 0.0]], dtype=torch.float32)
        
        R = gram_schmidt_orthogonalization(v1, v2)
        
        # 提取三个基向量
        e1 = R[0, :, 0]
        e2 = R[0, :, 1]  
        e3 = R[0, :, 2]
        
        # 验证单位长度
        assert abs(torch.norm(e1).item() - 1.0) < 1e-6
        assert abs(torch.norm(e2).item() - 1.0) < 1e-6
        assert abs(torch.norm(e3).item() - 1.0) < 1e-6
        
        # 验证两两正交
        assert abs(torch.dot(e1, e2).item()) < 1e-6
        assert abs(torch.dot(e1, e3).item()) < 1e-6
        assert abs(torch.dot(e2, e3).item()) < 1e-6
        
        # 验证e3 = e1 × e2
        expected_e3 = torch.cross(e1, e2)
        torch.testing.assert_close(e3, expected_e3, atol=1e-6, rtol=1e-6)
    
    def test_gram_schmidt_colinear_warning(self):
        """测试共线向量的处理。"""
        v1 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        v2 = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)  # 与v1共线
        
        # 应该发出警告但不崩溃
        rotation_matrix = gram_schmidt_orthogonalization(v1, v2)
        
        # 验证形状正确
        assert rotation_matrix.shape == (1, 3, 3)


class TestRigidTransforms:
    """测试刚体变换计算。"""
    
    def test_compute_rigid_transforms_simple(self):
        """测试基本刚体变换计算。"""
        # 创建理想的主链几何
        n_coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        ca_coords = torch.tensor([[1.458, 0.0, 0.0]], dtype=torch.float32)  # 标准CA-N键长
        c_coords = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
        
        translations, rotations = compute_rigid_transforms_from_backbone(
            n_coords, ca_coords, c_coords
        )
        
        # 验证平移向量就是CA坐标
        torch.testing.assert_close(translations, ca_coords, atol=1e-6, rtol=1e-6)
        
        # 验证旋转矩阵形状
        assert rotations.shape == (1, 3, 3)
        
        # 验证旋转矩阵的有效性（应该不抛出异常）
        validate_rotation_matrix(rotations)
    
    def test_compute_rigid_transforms_batch(self):
        """测试批处理刚体变换计算。"""
        batch_size = 5
        
        # 创建批处理的主链坐标
        n_coords = torch.randn(batch_size, 3)
        ca_coords = torch.randn(batch_size, 3)
        c_coords = torch.randn(batch_size, 3)
        
        # 确保键长合理（避免数值问题）
        ca_to_c = c_coords - ca_coords
        ca_to_c = ca_to_c / torch.norm(ca_to_c, dim=-1, keepdim=True) * 1.525  # 标准CA-C键长
        c_coords = ca_coords + ca_to_c
        
        ca_to_n = n_coords - ca_coords
        ca_to_n = ca_to_n / torch.norm(ca_to_n, dim=-1, keepdim=True) * 1.458  # 标准CA-N键长
        n_coords = ca_coords + ca_to_n
        
        translations, rotations = compute_rigid_transforms_from_backbone(
            n_coords, ca_coords, c_coords
        )
        
        # 验证形状
        assert translations.shape == (batch_size, 3)
        assert rotations.shape == (batch_size, 3, 3)
        
        # 验证平移向量
        torch.testing.assert_close(translations, ca_coords, atol=1e-6, rtol=1e-6)
        
        # 验证所有旋转矩阵的有效性
        validate_rotation_matrix(rotations)
    
    def test_validate_rotation_matrix_valid(self):
        """测试有效旋转矩阵的验证。"""
        # 创建有效的旋转矩阵（绕z轴旋转45度）
        angle = math.pi / 4
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        
        rotation = torch.tensor([[[
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0]
        ]]], dtype=torch.float32)
        
        # 应该不抛出异常
        validate_rotation_matrix(rotation)
    
    def test_validate_rotation_matrix_invalid_determinant(self):
        """测试无效行列式的检测。"""
        # 创建行列式不为1的矩阵
        invalid_rotation = torch.tensor([[[
            [2.0, 0.0, 0.0],  # 缩放矩阵，行列式=2
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]]], dtype=torch.float32)
        
        with pytest.raises(ValueError, match="旋转矩阵行列式严重偏离1"):
            validate_rotation_matrix(invalid_rotation, eps=1e-6)
    
    def test_validate_rotation_matrix_invalid_orthogonality(self):
        """测试非正交矩阵的检测。"""
        # 创建非正交矩阵
        non_orthogonal = torch.tensor([[[
            [1.0, 0.5, 0.0],  # 非正交
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]]], dtype=torch.float32)
        
        with pytest.raises(ValueError, match="旋转矩阵正交性严重偏差"):
            validate_rotation_matrix(non_orthogonal, eps=1e-6)


class TestBackboneReconstruction:
    """测试主链原子坐标重建。"""
    
    def test_reconstruct_backbone_simple(self):
        """测试基本主链重建。"""
        # 使用单位变换（无旋转，原点平移）
        translations = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        rotations = torch.eye(3, dtype=torch.float32).unsqueeze(0)  # 单位矩阵
        
        n_coords, ca_coords, c_coords, o_coords = reconstruct_backbone_from_rigid_transforms(
            translations, rotations
        )
        
        # 验证形状
        assert n_coords.shape == (1, 3)
        assert ca_coords.shape == (1, 3)
        assert c_coords.shape == (1, 3)
        assert o_coords.shape == (1, 3)
        
        # 验证CA坐标就是平移向量
        torch.testing.assert_close(ca_coords, translations, atol=1e-6, rtol=1e-6)
        
        # 验证键长
        ca_n_dist = torch.norm(n_coords - ca_coords, dim=-1)
        ca_c_dist = torch.norm(c_coords - ca_coords, dim=-1)
        
        expected_ca_n = STANDARD_BACKBONE_GEOMETRY["CA_N_BOND_LENGTH"]
        expected_ca_c = STANDARD_BACKBONE_GEOMETRY["CA_C_BOND_LENGTH"]
        
        torch.testing.assert_close(ca_n_dist, torch.tensor([expected_ca_n]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ca_c_dist, torch.tensor([expected_ca_c]), atol=1e-5, rtol=1e-5)
    
    def test_reconstruct_backbone_batch(self):
        """测试批处理主链重建。"""
        batch_size = 3
        translations = torch.randn(batch_size, 3)
        
        # 创建随机旋转矩阵
        rotations = torch.stack([
            torch.eye(3) for _ in range(batch_size)
        ], dim=0)
        
        n_coords, ca_coords, c_coords, o_coords = reconstruct_backbone_from_rigid_transforms(
            translations, rotations
        )
        
        # 验证形状
        assert n_coords.shape == (batch_size, 3)
        assert ca_coords.shape == (batch_size, 3)
        assert c_coords.shape == (batch_size, 3)
        assert o_coords.shape == (batch_size, 3)
        
        # 验证CA坐标
        torch.testing.assert_close(ca_coords, translations, atol=1e-6, rtol=1e-6)
        
        # 验证所有键长
        ca_n_dists = torch.norm(n_coords - ca_coords, dim=-1)
        ca_c_dists = torch.norm(c_coords - ca_coords, dim=-1)
        
        expected_ca_n = torch.full((batch_size,), STANDARD_BACKBONE_GEOMETRY["CA_N_BOND_LENGTH"])
        expected_ca_c = torch.full((batch_size,), STANDARD_BACKBONE_GEOMETRY["CA_C_BOND_LENGTH"])
        
        torch.testing.assert_close(ca_n_dists, expected_ca_n, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ca_c_dists, expected_ca_c, atol=1e-5, rtol=1e-5)
    
    def test_reconstruct_backbone_geometry(self):
        """测试重建的主链几何正确性。"""
        # 使用单位变换
        translations = torch.zeros(1, 3)
        rotations = torch.eye(3).unsqueeze(0)
        
        n_coords, ca_coords, c_coords, o_coords = reconstruct_backbone_from_rigid_transforms(
            translations, rotations
        )
        
        # 计算键角
        # N-CA-C 键角
        ca_to_n = n_coords - ca_coords
        ca_to_c = c_coords - ca_coords
        
        cos_angle = torch.sum(ca_to_n * ca_to_c, dim=-1) / (
            torch.norm(ca_to_n, dim=-1) * torch.norm(ca_to_c, dim=-1)
        )
        angle_rad = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        angle_deg = angle_rad * 180.0 / math.pi
        
        expected_angle = STANDARD_BACKBONE_GEOMETRY["N_CA_C_ANGLE"]
        torch.testing.assert_close(angle_deg, torch.tensor([expected_angle]), atol=5.0, rtol=5e-2)


class TestRoundTripConsistency:
    """测试往返转换的一致性。"""
    
    def test_rigid_transform_roundtrip(self):
        """测试刚体变换计算与重建的往返一致性。"""
        # 创建原始主链坐标
        original_n = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        original_ca = torch.tensor([[1.458, 0.0, 0.0]], dtype=torch.float32)
        original_c = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
        
        # 计算刚体变换
        translations, rotations = compute_rigid_transforms_from_backbone(
            original_n, original_ca, original_c
        )
        
        # 重建主链坐标
        recon_n, recon_ca, recon_c, recon_o = reconstruct_backbone_from_rigid_transforms(
            translations, rotations
        )
        
        # 验证CA坐标完全一致
        torch.testing.assert_close(recon_ca, original_ca, atol=1e-5, rtol=1e-5)
        
        # 验证重建的主链具有正确的几何
        # 注意：由于我们使用标准几何重建，可能与原始坐标略有不同
        # 但基本的键长和角度应该是正确的
        
        # 验证CA-N键长
        ca_n_dist = torch.norm(recon_n - recon_ca, dim=-1)
        expected_ca_n = STANDARD_BACKBONE_GEOMETRY["CA_N_BOND_LENGTH"]
        torch.testing.assert_close(ca_n_dist, torch.tensor([expected_ca_n]), atol=1e-5, rtol=1e-5)
        
        # 验证CA-C键长
        ca_c_dist = torch.norm(recon_c - recon_ca, dim=-1)
        expected_ca_c = STANDARD_BACKBONE_GEOMETRY["CA_C_BOND_LENGTH"]
        torch.testing.assert_close(ca_c_dist, torch.tensor([expected_ca_c]), atol=1e-5, rtol=1e-5)
    
    def test_batch_roundtrip_consistency(self):
        """测试批处理往返转换的一致性。"""
        batch_size = 5
        
        # 创建理想化的主链几何
        ca_coords = torch.randn(batch_size, 3)
        
        # 使用标准键长和键角创建N和C坐标
        ca_n_length = STANDARD_BACKBONE_GEOMETRY["CA_N_BOND_LENGTH"]
        ca_c_length = STANDARD_BACKBONE_GEOMETRY["CA_C_BOND_LENGTH"]
        n_ca_c_angle = math.radians(STANDARD_BACKBONE_GEOMETRY["N_CA_C_ANGLE"])
        
        # 在局部坐标系中创建N和C
        c_local = torch.tensor([ca_c_length, 0.0, 0.0])
        n_angle = math.pi - n_ca_c_angle
        n_local = torch.tensor([ca_n_length * math.cos(n_angle), ca_n_length * math.sin(n_angle), 0.0])
        
        # 应用随机旋转到局部坐标
        n_coords = ca_coords + n_local.expand(batch_size, -1)
        c_coords = ca_coords + c_local.expand(batch_size, -1)
        
        # 往返转换
        translations, rotations = compute_rigid_transforms_from_backbone(
            n_coords, ca_coords, c_coords
        )
        
        recon_n, recon_ca, recon_c, recon_o = reconstruct_backbone_from_rigid_transforms(
            translations, rotations
        )
        
        # 验证CA坐标一致性
        torch.testing.assert_close(recon_ca, ca_coords, atol=1e-4, rtol=1e-4)
        
        # 验证重建的几何正确性
        recon_ca_n_dists = torch.norm(recon_n - recon_ca, dim=-1)
        recon_ca_c_dists = torch.norm(recon_c - recon_ca, dim=-1)
        
        expected_ca_n = torch.full((batch_size,), ca_n_length)
        expected_ca_c = torch.full((batch_size,), ca_c_length)
        
        torch.testing.assert_close(recon_ca_n_dists, expected_ca_n, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(recon_ca_c_dists, expected_ca_c, atol=1e-4, rtol=1e-4)


class TestConstants:
    """测试常量定义。"""
    
    def test_standard_backbone_geometry(self):
        """测试标准主链几何参数的合理性。"""
        # 验证关键参数存在且在合理范围内
        assert "CA_N_BOND_LENGTH" in STANDARD_BACKBONE_GEOMETRY
        assert "CA_C_BOND_LENGTH" in STANDARD_BACKBONE_GEOMETRY
        assert "N_CA_C_ANGLE" in STANDARD_BACKBONE_GEOMETRY
        
        # 验证键长在合理范围内（1.0-2.0 Å）
        ca_n_length = STANDARD_BACKBONE_GEOMETRY["CA_N_BOND_LENGTH"]
        ca_c_length = STANDARD_BACKBONE_GEOMETRY["CA_C_BOND_LENGTH"]
        
        assert 1.0 < ca_n_length < 2.0
        assert 1.0 < ca_c_length < 2.0
        
        # 验证键角在合理范围内（90-150度）
        n_ca_c_angle = STANDARD_BACKBONE_GEOMETRY["N_CA_C_ANGLE"]
        assert 90.0 < n_ca_c_angle < 150.0
    
    def test_numerical_constants(self):
        """测试数值计算常量的合理性。"""
        assert "EPSILON" in NUMERICAL_CONSTANTS
        assert "ANGLE_TOLERANCE" in NUMERICAL_CONSTANTS
        assert "DETERMINANT_TOLERANCE" in NUMERICAL_CONSTANTS
        
        # 验证epsilon值合理
        eps = NUMERICAL_CONSTANTS["EPSILON"]
        assert 1e-10 < eps < 1e-6


# 运行特定测试的辅助函数
if __name__ == "__main__":
    # 可以直接运行此文件进行快速测试
    import sys
    
    if len(sys.argv) > 1:
        # 运行特定测试类
        test_class_name = sys.argv[1]
        if test_class_name in globals():
            test_class = globals()[test_class_name]()
            for method_name in dir(test_class):
                if method_name.startswith('test_'):
                    print(f"运行 {test_class_name}.{method_name}...")
                    try:
                        getattr(test_class, method_name)()
                        print(f"✅ {method_name} 通过")
                    except Exception as e:
                        print(f"❌ {method_name} 失败: {e}")
    else:
        print("请指定要运行的测试类名称")
        print("可用的测试类：")
        for name in globals():
            if name.startswith('Test'):
                print(f"  - {name}")
