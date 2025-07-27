"""
Atom14 数据类和转换器的测试

本模块包含 Atom14 数据类及其相关转换函数的全面测试，包括：
- 数据类创建和验证
- ProteinTensor 双向转换
- 设备管理和批量操作
- 虚拟原子计算
- CIF 文件读写功能
"""

import tempfile
from pathlib import Path
from typing import Dict, Any

import torch
import pytest

from protrepr.core.atom14 import Atom14
from protrepr.representations.atom14_converter import (
    protein_tensor_to_atom14,
    atom14_to_protein_tensor, 
    compute_virtual_cb,
    validate_atom14_data,
    save_atom14_to_cif,
    save_protein_tensor_to_cif,
    ATOM14_ATOM_TYPES,
    RESIDUE_ATOM14_MAPPING
)


class TestAtom14DataClass:
    """测试 Atom14 数据类的基本功能。"""
    
    def test_atom14_creation_from_data(self, sample_atom14_data: Dict[str, Any]):
        """测试从数据字典创建 Atom14 实例。"""
        atom14 = Atom14(
            coords=sample_atom14_data["coords"],
            atom_mask=sample_atom14_data["atom_mask"],
            res_mask=sample_atom14_data["res_mask"],
            chain_ids=sample_atom14_data["chain_ids"],
            residue_types=sample_atom14_data["residue_types"],
            residue_indices=sample_atom14_data["residue_indices"],
            chain_residue_indices=sample_atom14_data["chain_residue_indices"],
            residue_names=sample_atom14_data["residue_names"],
            atom_names=sample_atom14_data["atom_names"]
        )
        
        assert atom14.coords.shape == (10, 14, 3)
        assert atom14.atom_mask.shape == (10, 14)
        assert atom14.res_mask.shape == (10,)
        assert atom14.num_residues == 10
        assert atom14.residue_names.shape == (10,)
        assert atom14.atom_names.shape == (14,)
    
    def test_atom14_from_protein_tensor(self, mock_protein_tensor):
        """测试从 ProteinTensor 创建 Atom14 实例。"""
        atom14 = Atom14.from_protein_tensor(mock_protein_tensor)
        
        # 验证基本属性
        assert isinstance(atom14, Atom14)
        assert atom14.coords.dim() == 3
        assert atom14.atom_mask.dim() == 2
        assert atom14.res_mask.dim() == 1
        assert atom14.coords.shape[-1] == 3
        assert atom14.coords.shape[-2] == 14
        
    def test_atom14_to_protein_tensor(self, sample_atom14_data: Dict[str, Any]):
        """测试将 Atom14 转换为 ProteinTensor。"""
        atom14 = Atom14(
            coords=sample_atom14_data["coords"],
            atom_mask=sample_atom14_data["atom_mask"],
            res_mask=sample_atom14_data["res_mask"],
            chain_ids=sample_atom14_data["chain_ids"],
            residue_types=sample_atom14_data["residue_types"],
            residue_indices=sample_atom14_data["residue_indices"],
            chain_residue_indices=sample_atom14_data["chain_residue_indices"],
            residue_names=sample_atom14_data["residue_names"],
            atom_names=sample_atom14_data["atom_names"]
        )
        
        protein_tensor = atom14.to_protein_tensor()
        
        # 验证转换结果
        assert hasattr(protein_tensor, 'coordinates')
        assert hasattr(protein_tensor, 'atom_types')
        assert hasattr(protein_tensor, 'residue_types')
        assert hasattr(protein_tensor, 'chain_ids')
        assert hasattr(protein_tensor, 'residue_numbers')

    def test_atom14_device_management(self, sample_atom14_data: Dict[str, Any]):
        """测试 Atom14 的设备管理功能。"""
        # 创建 CPU 上的 Atom14
        cpu_data = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                   for k, v in sample_atom14_data.items()}
        atom14_cpu = Atom14(
            coords=cpu_data["coords"],
            atom_mask=cpu_data["atom_mask"],
            res_mask=cpu_data["res_mask"],
            chain_ids=cpu_data["chain_ids"],
            residue_types=cpu_data["residue_types"],
            residue_indices=cpu_data["residue_indices"],
            chain_residue_indices=cpu_data["chain_residue_indices"],
            residue_names=cpu_data["residue_names"],
            atom_names=cpu_data["atom_names"]
        )
        
        assert atom14_cpu.device == torch.device("cpu")
        
        # 如果有 GPU，测试设备转换
        if torch.cuda.is_available():
            gpu_device = torch.device("cuda")
            atom14_gpu = atom14_cpu.to_device(gpu_device)
            assert atom14_gpu.device == gpu_device
            assert atom14_gpu.coords.device == gpu_device
            assert atom14_gpu.atom_mask.device == gpu_device
            assert atom14_gpu.res_mask.device == gpu_device


class TestAtom14ConverterFunctions:
    """测试 Atom14 转换器函数。"""
    
    def test_protein_tensor_to_atom14(self, mock_protein_tensor):
        """测试 ProteinTensor 到 Atom14 的转换。"""
        result = protein_tensor_to_atom14(mock_protein_tensor)
        
        # 验证返回值数量（新API应该返回9个值）
        assert len(result) == 9
        
        coords, atom_mask, res_mask, chain_ids, residue_types, residue_indices, chain_residue_indices, residue_names, atom_names = result
        
        # 验证形状
        assert coords.dim() == 3
        assert coords.shape[-1] == 3
        assert coords.shape[-2] == 14
        
        assert atom_mask.dim() == 2
        assert atom_mask.shape[-1] == 14
        assert atom_mask.dtype == torch.bool
        
        assert res_mask.dim() == 1
        assert res_mask.dtype == torch.bool
        
        # 验证张量化名称
        assert isinstance(residue_names, torch.Tensor)
        assert isinstance(atom_names, torch.Tensor)
        assert atom_names.shape == (14,)
    
    def test_atom14_to_protein_tensor(self, sample_atom14_data: Dict[str, Any]):
        """测试 Atom14 到 ProteinTensor 的转换。"""
        result = atom14_to_protein_tensor(
            sample_atom14_data["coords"],
            sample_atom14_data["atom_mask"],
            sample_atom14_data["res_mask"],
            sample_atom14_data["chain_ids"],
            sample_atom14_data["residue_types"],
            sample_atom14_data["residue_indices"],
            sample_atom14_data["chain_residue_indices"],
            sample_atom14_data["residue_names"],
            sample_atom14_data["atom_names"]
        )
        
        # 验证返回的对象具有必要的属性
        assert hasattr(result, 'coordinates')
        assert hasattr(result, 'atom_types')
        assert hasattr(result, 'residue_types')
        assert hasattr(result, 'chain_ids')
        assert hasattr(result, 'residue_numbers')
    
    def test_compute_virtual_cb(self):
        """测试甘氨酸虚拟 CB 原子计算。"""
        # 创建简单的主链原子坐标（单个残基）
        n_coords = torch.tensor([0.0, 0.0, 0.0])
        ca_coords = torch.tensor([1.0, 0.0, 0.0])
        c_coords = torch.tensor([1.0, 1.0, 0.0])
        
        virtual_cb = compute_virtual_cb(n_coords, ca_coords, c_coords)
        
        # 验证返回值是合理的
        assert virtual_cb.shape == (3,)
        assert not torch.isnan(virtual_cb).any()
        assert not torch.isinf(virtual_cb).any()
        
        # 验证与 CA 的距离在合理范围内（约1.5 Å）
        cb_ca_distance = torch.norm(virtual_cb - ca_coords)
        assert 1.0 < cb_ca_distance < 2.0
    
    def test_validate_atom14_data(self, sample_atom14_data: Dict[str, Any]):
        """测试 Atom14 数据验证函数。"""
        # 正常数据应该通过验证
        validate_atom14_data(
            sample_atom14_data["coords"],
            sample_atom14_data["atom_mask"],
            sample_atom14_data["res_mask"],
            sample_atom14_data["chain_ids"],
            sample_atom14_data["residue_types"],
            sample_atom14_data["residue_indices"],
            sample_atom14_data["chain_residue_indices"],
            sample_atom14_data["residue_names"],
            sample_atom14_data["atom_names"]
        )
        
        # 测试形状不匹配的情况
        with pytest.raises(ValueError):
            wrong_coords = torch.randn(5, 10, 3)  # 错误的原子数量
            validate_atom14_data(
                wrong_coords,
                sample_atom14_data["atom_mask"],
                sample_atom14_data["res_mask"],
                sample_atom14_data["chain_ids"],
                sample_atom14_data["residue_types"],
                sample_atom14_data["residue_indices"],
                sample_atom14_data["chain_residue_indices"],
                sample_atom14_data["residue_names"],
                sample_atom14_data["atom_names"]
            )


class TestAtom14Integration:
    """测试 Atom14 的集成功能。"""
    
    def test_roundtrip_conversion(self, mock_protein_tensor):
        """测试 ProteinTensor -> Atom14 -> ProteinTensor 的往返转换。"""
        # 第一步：ProteinTensor -> Atom14
        atom14 = Atom14.from_protein_tensor(mock_protein_tensor)
        
        # 第二步：Atom14 -> ProteinTensor
        reconstructed_pt = atom14.to_protein_tensor()
        
        # 验证往返转换的基本一致性
        assert hasattr(reconstructed_pt, 'coordinates')
        assert hasattr(reconstructed_pt, 'atom_types')
        assert reconstructed_pt.n_atoms > 0
        assert reconstructed_pt.n_residues > 0
    
    def test_batch_processing(self, sample_atom14_data: Dict[str, Any]):
        """测试批量处理功能。"""
        # 创建两个 Atom14 实例
        atom14_1 = Atom14(
            coords=sample_atom14_data["coords"],
            atom_mask=sample_atom14_data["atom_mask"],
            res_mask=sample_atom14_data["res_mask"],
            chain_ids=sample_atom14_data["chain_ids"],
            residue_types=sample_atom14_data["residue_types"],
            residue_indices=sample_atom14_data["residue_indices"],
            chain_residue_indices=sample_atom14_data["chain_residue_indices"],
            residue_names=sample_atom14_data["residue_names"],
            atom_names=sample_atom14_data["atom_names"]
        )
        
        # 创建第二个实例，使用不同的坐标
        coords_2 = torch.randn_like(sample_atom14_data["coords"])
        atom14_2 = Atom14(
            coords=coords_2,
            atom_mask=sample_atom14_data["atom_mask"],
            res_mask=sample_atom14_data["res_mask"],
            chain_ids=sample_atom14_data["chain_ids"],
            residue_types=sample_atom14_data["residue_types"],
            residue_indices=sample_atom14_data["residue_indices"],
            chain_residue_indices=sample_atom14_data["chain_residue_indices"],
            residue_names=sample_atom14_data["residue_names"],
            atom_names=sample_atom14_data["atom_names"]
        )
        
        # 验证它们的坐标确实不同
        assert not torch.allclose(atom14_1.coords, atom14_2.coords)
        
        # 验证其他属性相同
        assert torch.equal(atom14_1.atom_mask, atom14_2.atom_mask)
        assert torch.equal(atom14_1.res_mask, atom14_2.res_mask)
        assert torch.equal(atom14_1.chain_ids, atom14_2.chain_ids)


class TestAtom14CIFRoundTrip:
    """测试 Atom14 CIF 文件往返功能。"""
    
    def test_save_atom14_to_cif(self, sample_atom14_data: Dict[str, Any]):
        """测试将 Atom14 数据保存为 CIF 文件。"""
        atom14 = Atom14(
            coords=sample_atom14_data["coords"],
            atom_mask=sample_atom14_data["atom_mask"],
            res_mask=sample_atom14_data["res_mask"],
            chain_ids=sample_atom14_data["chain_ids"],
            residue_types=sample_atom14_data["residue_types"],
            residue_indices=sample_atom14_data["residue_indices"],
            chain_residue_indices=sample_atom14_data["chain_residue_indices"],
            residue_names=sample_atom14_data["residue_names"],
            atom_names=sample_atom14_data["atom_names"]
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_atom14.cif"
            
            # 保存 CIF 文件
            save_atom14_to_cif(atom14, str(output_path))
            
            # 验证文件已创建
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            
            # 验证文件内容基本格式
            with open(output_path, 'r') as f:
                content = f.read()
                assert "data_rebuilt_from_tensor" in content
                assert "_atom_site.Cartn_x" in content
                assert "ATOM" in content
    
    def test_save_protein_tensor_to_cif(self, mock_protein_tensor):
        """测试将 ProteinTensor 保存为 CIF 文件。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_protein_tensor.cif"
            
            # 保存 CIF 文件
            save_protein_tensor_to_cif(mock_protein_tensor, str(output_path))
            
            # 验证文件已创建
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            
            # 验证文件内容基本格式
            with open(output_path, 'r') as f:
                content = f.read()
                assert "data_rebuilt_from_tensor" in content
                assert "_atom_site.Cartn_x" in content
                assert "ATOM" in content
    
    def test_full_cif_roundtrip(self, mock_protein_tensor):
        """测试完整的 CIF 往返转换：ProteinTensor -> Atom14 -> CIF -> 验证。"""
        with tempfile.TemporaryDirectory() as temp_output_dir:
            temp_output_dir = Path(temp_output_dir)
            
            # 1. ProteinTensor -> Atom14
            atom14 = Atom14.from_protein_tensor(mock_protein_tensor)
            
            # 2. 保存 Atom14 到 CIF
            atom14_cif_path = temp_output_dir / "roundtrip_atom14.cif"
            save_atom14_to_cif(atom14, str(atom14_cif_path))
            
            # 3. Atom14 -> ProteinTensor  
            reconstructed_pt = atom14.to_protein_tensor()
            
            # 4. 保存重建的 ProteinTensor 到 CIF
            reconstructed_cif_path = temp_output_dir / "roundtrip_reconstructed.cif"
            save_protein_tensor_to_cif(reconstructed_pt, str(reconstructed_cif_path))
            
            # 验证两个文件都已创建
            assert atom14_cif_path.exists()
            assert reconstructed_cif_path.exists()
            
            # 验证文件大小合理
            assert atom14_cif_path.stat().st_size > 100
            assert reconstructed_cif_path.stat().st_size > 100
            
            # 验证基本的 CIF 格式
            for cif_path in [atom14_cif_path, reconstructed_cif_path]:
                with open(cif_path, 'r') as f:
                    content = f.read()
                    assert "data_" in content
                    assert "_atom_site" in content
                    assert "ATOM" in content 