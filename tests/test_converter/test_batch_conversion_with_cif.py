"""
测试批量转换功能 - 使用真实的 CIF 文件

使用工作区中的 9ct8.cif 文件测试批量 PDB 到 Atom14 转换功能，
包括单文件转换、批量转换、不同输出格式和错误处理。
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import torch
import numpy as np

from protrepr.batch_processing import (
    BatchPDBToAtom14Converter,
    convert_single_worker,
    save_statistics
)
from protrepr.core.atom14 import Atom14
from protein_tensor import load_structure


class TestBatchConversionWithCIF:
    """测试批量转换功能，使用真实的 CIF 文件。"""
    
    @pytest.fixture
    def cif_file_path(self) -> Path:
        """获取 9ct8.cif 文件路径。"""
        # 假设 9ct8.cif 在项目根目录
        root_dir = Path(__file__).parents[2]  # 上两级目录到项目根
        cif_path = root_dir / "9ct8.cif"
        
        if not cif_path.exists():
            pytest.skip(f"测试文件不存在: {cif_path}")
        
        return cif_path
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试输出。"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def converter(self) -> BatchPDBToAtom14Converter:
        """创建批量转换器实例。"""
        return BatchPDBToAtom14Converter(
            n_workers=1,  # 单进程避免并发问题
            preserve_structure=True,
            device="cpu",
            output_format="npz"
        )
    
    def test_converter_initialization(self, converter: BatchPDBToAtom14Converter):
        """测试转换器初始化。"""
        assert converter.n_workers == 1
        assert converter.preserve_structure is True
        assert converter.device == torch.device("cpu")
        assert converter.output_format == "npz"
    
    def test_find_structure_files_single_file(self, converter: BatchPDBToAtom14Converter, cif_file_path: Path):
        """测试查找单个结构文件。"""
        files = converter.find_structure_files(cif_file_path)
        assert len(files) == 1
        assert files[0] == cif_file_path
    
    def test_find_structure_files_directory(self, converter: BatchPDBToAtom14Converter, cif_file_path: Path):
        """测试在目录中查找结构文件。"""
        # 使用包含 CIF 文件的目录
        parent_dir = cif_file_path.parent
        files = converter.find_structure_files(parent_dir, recursive=False)
        
        # 应该找到至少一个 CIF 文件
        cif_files = [f for f in files if f.suffix.lower() in {'.cif', '.mmcif'}]
        assert len(cif_files) >= 1
        assert cif_file_path in cif_files
    
    def test_convert_single_file_npz(self, converter: BatchPDBToAtom14Converter, 
                                    cif_file_path: Path, temp_dir: Path):
        """测试单文件转换为 NPZ 格式。"""
        output_file = temp_dir / "test_output.npz"
        
        result = converter.convert_single_file(cif_file_path, output_file)
        
        # 验证转换成功
        assert result['success'] is True
        assert result['error'] is None
        assert result['processing_time'] > 0
        assert result['num_residues'] > 0
        assert result['num_atoms'] > 0
        assert result['num_chains'] > 0
        
        # 验证输出文件存在
        assert output_file.exists()
        
        # 验证 NPZ 文件内容
        data = np.load(output_file)
        expected_keys = {
            'coords', 'atom_mask', 'res_mask', 'chain_ids', 'residue_types',
            'residue_indices', 'chain_residue_indices', 'residue_names', 
            'atom_names', 'num_residues', 'num_chains'
        }
        assert set(data.keys()) == expected_keys
        
        # 验证数据形状
        num_residues = int(data['num_residues'])
        assert data['coords'].shape == (num_residues, 14, 3)
        assert data['atom_mask'].shape == (num_residues, 14)
        assert data['res_mask'].shape == (num_residues,)
    
    def test_convert_single_file_pt(self, temp_dir: Path, cif_file_path: Path):
        """测试单文件转换为 PyTorch 格式。"""
        converter = BatchPDBToAtom14Converter(
            n_workers=1,
            output_format="pt"
        )
        output_file = temp_dir / "test_output.pt"
        
        result = converter.convert_single_file(cif_file_path, output_file)
        
        # 验证转换成功
        assert result['success'] is True
        assert output_file.exists()
        
        # 验证 PT 文件内容
        data = torch.load(output_file, map_location='cpu')
        expected_keys = {
            'coords', 'atom_mask', 'res_mask', 'chain_ids', 'residue_types',
            'residue_indices', 'chain_residue_indices', 'residue_names', 
            'atom_names', 'metadata'
        }
        assert set(data.keys()) == expected_keys
        
        # 验证元数据
        metadata = data['metadata']
        assert metadata['format'] == 'atom14'
        assert metadata['version'] == '1.0'
        assert metadata['num_residues'] > 0
        assert metadata['num_chains'] > 0
    
    def test_batch_conversion_single_file(self, converter: BatchPDBToAtom14Converter,
                                         cif_file_path: Path, temp_dir: Path):
        """测试批量转换单个文件。"""
        statistics = converter.convert_batch(
            input_path=cif_file_path,
            output_dir=temp_dir,
            recursive=True
        )
        
        # 验证统计信息
        assert statistics['total'] == 1
        assert statistics['success'] == 1
        assert statistics['failed'] == 0
        assert len(statistics['failed_files']) == 0
        assert len(statistics['results']) == 1
        
        # 验证输出文件
        expected_output = temp_dir / f"{cif_file_path.stem}.npz"
        assert expected_output.exists()
        
        # 验证结果详情
        result = statistics['results'][0]
        assert result['success'] is True
        assert result['num_residues'] > 0
        assert result['num_atoms'] > 0
        assert result['num_chains'] > 0
    
    def test_convert_single_worker_function(self, cif_file_path: Path, temp_dir: Path):
        """测试单工作进程函数。"""
        output_file = temp_dir / "worker_test.npz"
        
        result = convert_single_worker(
            input_file=str(cif_file_path),
            output_file=str(output_file),
            output_format="npz",
            device="cpu"
        )
        
        assert result['success'] is True
        assert output_file.exists()
        assert result['num_residues'] > 0
    
    def test_roundtrip_conversion_validation(self, converter: BatchPDBToAtom14Converter,
                                           cif_file_path: Path, temp_dir: Path):
        """测试完整的往返转换验证。"""
        # 1. 原始 CIF -> Atom14
        original_protein = load_structure(cif_file_path)
        original_atom14 = Atom14.from_protein_tensor(original_protein)
        
        # 2. 通过批量转换器保存
        output_file = temp_dir / "roundtrip_test.npz"
        result = converter.convert_single_file(cif_file_path, output_file)
        assert result['success'] is True
        
        # 3. 从 NPZ 重新加载
        data = np.load(output_file)
        
        # 重建 Atom14 对象
        reloaded_atom14 = Atom14(
            coords=torch.from_numpy(data['coords']),
            atom_mask=torch.from_numpy(data['atom_mask']),
            res_mask=torch.from_numpy(data['res_mask']),
            chain_ids=torch.from_numpy(data['chain_ids']),
            residue_types=torch.from_numpy(data['residue_types']),
            residue_indices=torch.from_numpy(data['residue_indices']),
            chain_residue_indices=torch.from_numpy(data['chain_residue_indices']),
            residue_names=torch.from_numpy(data['residue_names']),
            atom_names=torch.from_numpy(data['atom_names'])
        )
        
        # 4. 验证数据一致性
        assert original_atom14.num_residues == reloaded_atom14.num_residues
        assert original_atom14.num_chains == reloaded_atom14.num_chains
        
        # 验证坐标（允许小的数值误差）
        torch.testing.assert_close(
            original_atom14.coords, 
            reloaded_atom14.coords, 
            rtol=1e-5, 
            atol=1e-6
        )
        
        # 验证掩码
        torch.testing.assert_close(
            original_atom14.atom_mask, 
            reloaded_atom14.atom_mask
        )
        torch.testing.assert_close(
            original_atom14.res_mask, 
            reloaded_atom14.res_mask
        )
    
    def test_save_statistics_function(self, temp_dir: Path):
        """测试统计信息保存功能。"""
        # 创建示例统计数据
        test_statistics = {
            'total': 1,
            'success': 1,
            'failed': 0,
            'failed_files': [],
            'results': [{
                'input_file': 'test.cif',
                'output_file': 'test.npz',
                'success': True,
                'error': None,
                'processing_time': 1.23,
                'num_residues': 100,
                'num_atoms': 800,
                'num_chains': 2
            }]
        }
        
        stats_file = temp_dir / "test_stats.json"
        save_statistics(test_statistics, stats_file)
        
        assert stats_file.exists()
        
        # 验证可以正确加载 JSON
        import json
        with open(stats_file, 'r', encoding='utf-8') as f:
            loaded_stats = json.load(f)
        
        assert loaded_stats['total'] == 1
        assert loaded_stats['success'] == 1
        assert len(loaded_stats['results']) == 1
    
    def test_conversion_error_handling(self, converter: BatchPDBToAtom14Converter, temp_dir: Path):
        """测试转换错误处理。"""
        # 使用不存在的文件
        fake_file = temp_dir / "nonexistent.pdb"
        output_file = temp_dir / "error_test.npz"
        
        result = converter.convert_single_file(fake_file, output_file)
        
        assert result['success'] is False
        assert result['error'] is not None
        assert result['processing_time'] >= 0
        assert result['num_residues'] == 0
        assert result['num_atoms'] == 0
        assert result['num_chains'] == 0
        assert not output_file.exists()
    
    def test_batch_conversion_with_mixed_results(self, converter: BatchPDBToAtom14Converter,
                                                cif_file_path: Path, temp_dir: Path):
        """测试混合结果的批量转换（成功和失败的文件）。"""
        # 创建一个包含真实文件和假文件的临时目录
        test_input_dir = temp_dir / "input"
        test_input_dir.mkdir()
        
        # 复制真实的 CIF 文件
        real_file = test_input_dir / "real.cif"
        shutil.copy2(cif_file_path, real_file)
        
        # 创建一个假的 PDB 文件（内容无效）
        fake_file = test_input_dir / "fake.pdb"
        fake_file.write_text("INVALID PDB CONTENT")
        
        # 执行批量转换
        test_output_dir = temp_dir / "output"
        statistics = converter.convert_batch(
            input_path=test_input_dir,
            output_dir=test_output_dir,
            recursive=True
        )
        
        # 验证统计信息
        assert statistics['total'] == 2
        assert statistics['success'] == 1  # 只有真实文件成功
        assert statistics['failed'] == 1   # 假文件失败
        assert len(statistics['failed_files']) == 1
        assert len(statistics['results']) == 2
        
        # 验证成功的输出文件存在
        successful_output = test_output_dir / "real.npz"
        assert successful_output.exists()
        
        # 验证失败的输出文件不存在
        failed_output = test_output_dir / "fake.npz"
        assert not failed_output.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 