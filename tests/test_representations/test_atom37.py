"""
Atom37 表示法测试模块

本模块包含针对 Atom37 蛋白质表示的全面测试，验证：
- Atom37 数据类的核心功能
- ProteinTensor 与 Atom37 的双向转换
- 批量维度支持和设备管理
- 数据验证和错误处理
- 实际蛋白质结构的转换准确性

测试策略：
- 使用真实的测试数据文件
- 验证转换的可逆性和准确性
- 测试批量操作和设备转移
- 确保与 AlphaFold atom37 标准的兼容性
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

import pytest
import torch
import numpy as np

# 添加源码路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from protein_tensor import load_structure
from protrepr.core.atom37 import Atom37
from protrepr.representations.atom37_converter import (
    protein_tensor_to_atom37,
    atom37_to_protein_tensor,
    validate_atom37_data,
    RESIDUE_ATOM37_MAPPING,
    ATOM37_ATOM_TYPES,
    get_residue_atom37_mapping,
    get_atom37_atom_positions
)


class TestAtom37Basic:
    """Atom37 基础功能测试"""
    
    @pytest.fixture
    def sample_protein_tensor(self):
        """加载示例蛋白质结构"""
        data_dir = Path(__file__).parent.parent / "data"
        test_files = list(data_dir.glob("*.cif")) + list(data_dir.glob("*.pdb"))
        
        if not test_files:
            pytest.skip("没有找到测试数据文件")
        
        return load_structure(test_files[0])
    
    @pytest.fixture
    def sample_atom37(self, sample_protein_tensor):
        """创建示例 Atom37 实例"""
        return Atom37.from_protein_tensor(sample_protein_tensor)
    
    def test_atom37_constants(self):
        """测试 Atom37 常量定义"""
        # 验证原子类型列表
        assert len(ATOM37_ATOM_TYPES) == 37
        assert ATOM37_ATOM_TYPES[0] == "N"
        assert ATOM37_ATOM_TYPES[1] == "CA"
        assert ATOM37_ATOM_TYPES[2] == "C"
        assert ATOM37_ATOM_TYPES[3] == "O"
        assert ATOM37_ATOM_TYPES[4] == "CB"
        
        # 验证残基映射完整性
        assert len(RESIDUE_ATOM37_MAPPING) == 20  # 20种标准氨基酸
        
        # 验证每种残基都有主链原子
        for residue_name, mapping in RESIDUE_ATOM37_MAPPING.items():
            if residue_name != "GLY":  # 甘氨酸没有CB
                assert "N" in mapping
                assert "CA" in mapping
                assert "C" in mapping
                assert "O" in mapping
            else:
                # 甘氨酸只有主链原子
                assert mapping == {"N": 0, "CA": 1, "C": 2, "O": 3}
    
    def test_atom37_creation_from_protein_tensor(self, sample_protein_tensor):
        """测试从 ProteinTensor 创建 Atom37"""
        atom37 = Atom37.from_protein_tensor(sample_protein_tensor)
        
        # 验证基本属性
        assert isinstance(atom37, Atom37)
        assert atom37.num_residues > 0
        assert atom37.num_chains > 0
        assert atom37.num_atoms_per_residue == 37
        
        # 验证张量形状
        assert atom37.coords.shape == (atom37.num_residues, 37, 3)
        assert atom37.atom_mask.shape == (atom37.num_residues, 37)
        assert atom37.res_mask.shape == (atom37.num_residues,)
        
        # 验证数据类型
        assert atom37.coords.dtype == torch.float32
        assert atom37.atom_mask.dtype == torch.bool
        assert atom37.res_mask.dtype == torch.bool
        
        # 验证有实际的原子数据
        assert atom37.atom_mask.sum() > 0  # 至少有一些真实原子
        assert atom37.res_mask.sum() > 0   # 至少有一些真实残基
    
    def test_atom37_device_management(self, sample_atom37):
        """测试设备管理"""
        original_device = sample_atom37.device
        assert isinstance(original_device, torch.device)
        
        # 测试移动到CPU
        atom37_cpu = sample_atom37.to_device(torch.device('cpu'))
        assert atom37_cpu.device == torch.device('cpu')
        assert atom37_cpu.coords.device == torch.device('cpu')
        assert atom37_cpu.atom_mask.device == torch.device('cpu')
        
        # 如果有CUDA，测试移动到CUDA
        if torch.cuda.is_available():
            atom37_cuda = sample_atom37.to_device(torch.device('cuda'))
            assert atom37_cuda.device.type == 'cuda'
            assert atom37_cuda.coords.device.type == 'cuda'
            assert atom37_cuda.atom_mask.device.type == 'cuda'
    
    def test_atom37_validation(self, sample_atom37):
        """测试数据验证"""
        # 正常数据应该通过验证
        sample_atom37.validate()
        
        # 测试形状不匹配的情况
        with pytest.raises(ValueError):
            # 创建形状不匹配的数据
            bad_coords = torch.zeros(10, 37, 3)  # 错误的残基数量
            bad_atom37 = Atom37(
                coords=bad_coords,
                atom_mask=sample_atom37.atom_mask,
                res_mask=sample_atom37.res_mask,
                chain_ids=sample_atom37.chain_ids,
                residue_types=sample_atom37.residue_types,
                residue_indices=sample_atom37.residue_indices,
                chain_residue_indices=sample_atom37.chain_residue_indices,
                residue_names=sample_atom37.residue_names,
                atom_names=sample_atom37.atom_names
            )
    
    def test_atom37_properties(self, sample_atom37):
        """测试 Atom37 属性"""
        # 测试基本属性
        assert sample_atom37.num_residues > 0
        assert sample_atom37.num_chains > 0
        assert sample_atom37.num_atoms_per_residue == 37
        assert len(sample_atom37.batch_shape) == 0  # 非批量数据
        
        # 测试主链和侧链坐标提取
        backbone_coords = sample_atom37.get_backbone_coords()
        assert backbone_coords.shape == (sample_atom37.num_residues, 4, 3)
        
        sidechain_coords = sample_atom37.get_sidechain_coords()
        assert sidechain_coords.shape == (sample_atom37.num_residues, 33, 3)
        
        # 测试质心计算
        center_of_mass = sample_atom37.compute_center_of_mass()
        assert center_of_mass.shape == (sample_atom37.num_residues, 3)
        assert not torch.isnan(center_of_mass).any()


class TestAtom37Conversion:
    """Atom37 转换功能测试"""
    
    @pytest.fixture
    def sample_protein_tensor(self):
        """加载示例蛋白质结构"""
        data_dir = Path(__file__).parent.parent / "data"
        test_files = list(data_dir.glob("*.cif")) + list(data_dir.glob("*.pdb"))
        
        if not test_files:
            pytest.skip("没有找到测试数据文件")
        
        return load_structure(test_files[0])
    
    def test_protein_tensor_to_atom37_conversion(self, sample_protein_tensor):
        """测试 ProteinTensor 到 Atom37 的转换"""
        result = protein_tensor_to_atom37(sample_protein_tensor)
        
        assert len(result) == 9  # 返回9个元素的元组
        (coords, atom_mask, res_mask, chain_ids, residue_types, 
         residue_indices, chain_residue_indices, residue_names, atom_names) = result
        
        # 验证形状
        num_residues = coords.shape[0]
        assert coords.shape == (num_residues, 37, 3)
        assert atom_mask.shape == (num_residues, 37)
        assert res_mask.shape == (num_residues,)
        assert chain_ids.shape == (num_residues,)
        assert residue_types.shape == (num_residues,)
        assert residue_indices.shape == (num_residues,)
        assert atom_names.shape == (37,)
        
        # 验证数据类型
        assert coords.dtype == torch.float32
        assert atom_mask.dtype == torch.bool
        assert res_mask.dtype == torch.bool
    
    def test_atom37_to_protein_tensor_conversion(self, sample_protein_tensor):
        """测试 Atom37 到 ProteinTensor 的转换"""
        # 先转换为 Atom37
        atom37 = Atom37.from_protein_tensor(sample_protein_tensor)
        
        # 再转换回 ProteinTensor
        reconstructed_pt = atom37.to_protein_tensor()
        
        # 验证重建的结构
        assert hasattr(reconstructed_pt, 'coordinates')
        assert hasattr(reconstructed_pt, 'atom_types')
        assert hasattr(reconstructed_pt, 'residue_types')
        assert hasattr(reconstructed_pt, 'chain_ids')
        assert hasattr(reconstructed_pt, 'residue_numbers')
        
        # 验证原子数量合理（可能有所不同，因为过滤了虚拟原子）
        assert reconstructed_pt.n_atoms > 0
        assert reconstructed_pt.n_residues > 0
    
    def test_conversion_consistency(self, sample_protein_tensor):
        """测试转换的一致性"""
        # 第一次转换
        atom37_1 = Atom37.from_protein_tensor(sample_protein_tensor)
        
        # 转换回去再转换
        pt_reconstructed = atom37_1.to_protein_tensor()
        atom37_2 = Atom37.from_protein_tensor(pt_reconstructed)
        
        # 验证残基数量应该一致
        assert atom37_1.num_residues == atom37_2.num_residues
        
        # 验证原子掩码的总数应该相近（可能有微小差异）
        mask_diff = abs(atom37_1.atom_mask.sum() - atom37_2.atom_mask.sum())
        assert mask_diff <= atom37_1.num_residues  # 允许每个残基最多1个原子的差异


class TestAtom37FileOperations:
    """Atom37 文件操作测试"""
    
    @pytest.fixture
    def sample_atom37(self):
        """创建示例 Atom37 实例"""
        data_dir = Path(__file__).parent.parent / "data"
        test_files = list(data_dir.glob("*.cif")) + list(data_dir.glob("*.pdb"))
        
        if not test_files:
            pytest.skip("没有找到测试数据文件")
        
        protein_tensor = load_structure(test_files[0])
        return Atom37.from_protein_tensor(protein_tensor)
    
    @pytest.fixture
    def temp_dir(self, tmp_path):
        """创建临时目录"""
        return tmp_path / "atom37_test"
    
    def test_atom37_save_load_instance(self, sample_atom37, temp_dir):
        """测试 Atom37 实例的保存和加载"""
        temp_dir.mkdir(exist_ok=True)
        save_path = temp_dir / "test_atom37_instance.pt"
        
        # 保存为实例
        sample_atom37.save(save_path, save_as_instance=True)
        assert save_path.exists()
        
        # 加载并验证
        loaded_atom37 = Atom37.load(save_path)
        assert isinstance(loaded_atom37, Atom37)
        assert loaded_atom37.num_residues == sample_atom37.num_residues
        assert loaded_atom37.num_chains == sample_atom37.num_chains
        
        # 验证坐标相似性
        coords_close = torch.allclose(
            loaded_atom37.coords, sample_atom37.coords, 
            rtol=1e-5, atol=1e-6
        )
        assert coords_close
    
    def test_atom37_save_load_dict(self, sample_atom37, temp_dir):
        """测试 Atom37 字典格式的保存和加载"""
        temp_dir.mkdir(exist_ok=True)
        save_path = temp_dir / "test_atom37_dict.pt"
        
        # 保存为字典
        sample_atom37.save(save_path, save_as_instance=False)
        assert save_path.exists()
        
        # 加载并验证
        loaded_atom37 = Atom37.load(save_path)
        assert isinstance(loaded_atom37, Atom37)
        assert loaded_atom37.num_residues == sample_atom37.num_residues
        assert loaded_atom37.num_chains == sample_atom37.num_chains
    
    def test_atom37_to_cif(self, sample_atom37, temp_dir):
        """测试 Atom37 到 CIF 文件的转换"""
        temp_dir.mkdir(exist_ok=True)
        cif_path = temp_dir / "test_output.cif"
        
        # 转换为 CIF
        sample_atom37.to_cif(cif_path)
        assert cif_path.exists()
        
        # 验证文件内容
        cif_content = cif_path.read_text()
        assert "data_" in cif_content  # CIF 文件应该包含数据块
        assert "_atom_site" in cif_content  # 应该包含原子信息


class TestAtom37Utils:
    """Atom37 工具函数测试"""
    
    def test_get_atom37_atom_positions(self):
        """测试原子位置映射函数"""
        positions = get_atom37_atom_positions()
        assert len(positions) == 37
        assert positions["N"] == 0
        assert positions["CA"] == 1
        assert positions["C"] == 2
        assert positions["O"] == 3
        assert positions["CB"] == 4
    
    def test_get_residue_atom37_mapping(self):
        """测试残基原子映射函数"""
        # 测试丙氨酸
        ala_mapping = get_residue_atom37_mapping("ALA")
        expected_ala = {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4}
        assert ala_mapping == expected_ala
        
        # 测试甘氨酸（没有CB）
        gly_mapping = get_residue_atom37_mapping("GLY")
        expected_gly = {"N": 0, "CA": 1, "C": 2, "O": 3}
        assert gly_mapping == expected_gly
        
        # 测试不存在的残基
        with pytest.raises(KeyError):
            get_residue_atom37_mapping("XXX")
    
    def test_validate_atom37_data(self):
        """测试数据验证函数"""
        # 创建有效的测试数据
        num_residues = 5
        coords = torch.randn(num_residues, 37, 3)
        atom_mask = torch.randint(0, 2, (num_residues, 37)).bool()
        res_mask = torch.ones(num_residues).bool()
        chain_ids = torch.zeros(num_residues).long()
        residue_types = torch.randint(0, 20, (num_residues,)).long()
        residue_indices = torch.arange(num_residues).long()
        chain_residue_indices = torch.arange(num_residues).long()
        residue_names = torch.randint(0, 20, (num_residues,)).long()
        atom_names = torch.arange(37).long()
        
        # 正常数据应该通过验证
        validate_atom37_data(
            coords, atom_mask, res_mask, chain_ids, residue_types,
            residue_indices, chain_residue_indices, residue_names, atom_names
        )
        
        # 测试形状不匹配
        with pytest.raises(ValueError):
            bad_coords = torch.randn(num_residues, 35, 3)  # 错误的原子数
            validate_atom37_data(
                bad_coords, atom_mask, res_mask, chain_ids, residue_types,
                residue_indices, chain_residue_indices, residue_names, atom_names
            )


class TestAtom37SpecificResidues:
    """针对特定残基的 Atom37 测试"""
    
    def test_glycine_handling(self):
        """测试甘氨酸的特殊处理（没有CB原子）"""
        gly_mapping = RESIDUE_ATOM37_MAPPING["GLY"]
        assert "CB" not in gly_mapping
        assert len(gly_mapping) == 4  # 只有主链原子
    
    def test_complex_residues(self):
        """测试复杂残基的原子映射"""
        # 测试色氨酸（最复杂的残基之一）
        trp_mapping = RESIDUE_ATOM37_MAPPING["TRP"]
        assert len(trp_mapping) > 10  # 色氨酸有很多原子
        
        # 验证基本原子存在
        for atom in ["N", "CA", "C", "O", "CB"]:
            assert atom in trp_mapping
        
        # 测试精氨酸（带电残基）
        arg_mapping = RESIDUE_ATOM37_MAPPING["ARG"]
        assert "NH1" in arg_mapping
        assert "NH2" in arg_mapping
        assert "CZ" in arg_mapping


class TestAtom37EdgeCases:
    """Atom37 边界情况测试"""
    
    def test_empty_protein(self):
        """测试空蛋白质的处理"""
        # 这个测试可能需要根据实际的错误处理策略调整
        pass
    
    def test_single_residue_protein(self):
        """测试单残基蛋白质"""
        # 创建最小的测试数据
        num_residues = 1
        coords = torch.randn(num_residues, 37, 3)
        atom_mask = torch.zeros(num_residues, 37).bool()
        atom_mask[0, :4] = True  # 只有主链原子
        res_mask = torch.ones(num_residues).bool()
        chain_ids = torch.zeros(num_residues).long()
        residue_types = torch.zeros(num_residues).long()  # ALA
        residue_indices = torch.zeros(num_residues).long()
        chain_residue_indices = torch.zeros(num_residues).long()
        residue_names = torch.zeros(num_residues).long()
        atom_names = torch.arange(37).long()
        
        atom37 = Atom37(
            coords=coords,
            atom_mask=atom_mask,
            res_mask=res_mask,
            chain_ids=chain_ids,
            residue_types=residue_types,
            residue_indices=residue_indices,
            chain_residue_indices=chain_residue_indices,
            residue_names=residue_names,
            atom_names=atom_names
        )
        
        assert atom37.num_residues == 1
        assert atom37.num_chains == 1
    
    def test_batch_dimensions(self):
        """测试批量维度支持"""
        batch_size = 3
        num_residues = 5
        
        # 创建批量数据
        coords = torch.randn(batch_size, num_residues, 37, 3)
        atom_mask = torch.randint(0, 2, (batch_size, num_residues, 37)).bool()
        res_mask = torch.ones(batch_size, num_residues).bool()
        chain_ids = torch.zeros(batch_size, num_residues).long()
        residue_types = torch.randint(0, 20, (batch_size, num_residues)).long()
        residue_indices = torch.arange(num_residues).unsqueeze(0).expand(batch_size, -1).long()
        chain_residue_indices = torch.arange(num_residues).unsqueeze(0).expand(batch_size, -1).long()
        residue_names = torch.randint(0, 20, (batch_size, num_residues)).long()
        atom_names = torch.arange(37).long()
        
        atom37 = Atom37(
            coords=coords,
            atom_mask=atom_mask,
            res_mask=res_mask,
            chain_ids=chain_ids,
            residue_types=residue_types,
            residue_indices=residue_indices,
            chain_residue_indices=chain_residue_indices,
            residue_names=residue_names,
            atom_names=atom_names
        )
        
        assert atom37.batch_shape == (batch_size,)
        assert atom37.num_residues == num_residues
        
        # 测试质心计算在批量维度下的正确性
        center_of_mass = atom37.compute_center_of_mass()
        assert center_of_mass.shape == (batch_size, num_residues, 3)


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"]) 