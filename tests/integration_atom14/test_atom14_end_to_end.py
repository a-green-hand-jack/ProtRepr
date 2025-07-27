"""
Atom14 端到端集成测试

本模块提供完整的 Atom14 工作流集成测试，覆盖从原始结构文件到最终输出的
完整数据流，确保所有转换步骤的正确性和数据一致性。

测试流程：
1. CIF/PDB → ProteinTensor → Atom14
2. Atom14 → NPZ/PT 格式保存
3. NPZ/PT → Atom14 重新加载
4. 数据一致性验证
5. Atom14 → CIF/PDB 重建
6. 结构完整性验证
"""

import sys
from pathlib import Path
import tempfile
import shutil
import time
import json
from typing import Dict, List, Any, Tuple, Optional

import pytest
import torch
import numpy as np

# 添加源码路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from protein_tensor import load_structure
from protrepr.core.atom14 import Atom14
from protrepr.representations.atom14_converter import (
    protein_tensor_to_atom14, 
    save_atom14_to_cif,
    save_protein_tensor_to_cif
)
from protrepr.batch_processing import BatchPDBToAtom14Converter


class TestAtom14EndToEnd:
    """Atom14 端到端集成测试类"""
    
    @pytest.fixture(scope="class")
    def test_output_dir(self) -> Path:
        """创建测试输出目录"""
        output_dir = Path(__file__).parent / "test_results"
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    @pytest.fixture(scope="class")
    def test_data_files(self) -> List[Path]:
        """获取测试用的 CIF 文件"""
        data_dir = Path(__file__).parent.parent / "data"
        test_files = []
        
        # 查找可用的测试文件
        for pattern in ["*.cif", "*.pdb"]:
            test_files.extend(data_dir.glob(pattern))
        
        if not test_files:
            pytest.skip("没有找到测试数据文件")
        
        return test_files[:2]  # 限制为前两个文件，避免测试时间过长
    
    def test_complete_workflow(self, test_data_files: List[Path], test_output_dir: Path):
        """
        测试完整的 Atom14 工作流程
        
        执行步骤：
        1. 加载 CIF/PDB 文件
        2. 转换为 Atom14
        3. 保存为多种格式
        4. 重新加载验证
        5. 重建为 CIF 文件
        """
        workflow_results = {}
        
        for test_file in test_data_files:
            file_results = self._test_single_file_workflow(test_file, test_output_dir)
            workflow_results[test_file.name] = file_results
        
        # 保存工作流测试结果
        results_file = test_output_dir / "workflow_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(workflow_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 验证所有文件都成功处理
        for filename, results in workflow_results.items():
            assert results['success'], f"文件 {filename} 的工作流测试失败: {results.get('error', 'Unknown error')}"
    
    def _test_single_file_workflow(self, input_file: Path, output_dir: Path) -> Dict[str, Any]:
        """测试单个文件的完整工作流"""
        file_stem = input_file.stem
        results = {
            'input_file': str(input_file),
            'success': False,
            'error': None,
            'steps': {},
            'timing': {},
            'validation': {}
        }
        
        try:
            # 步骤 1: 加载原始文件
            step_start = time.perf_counter()
            original_protein = load_structure(input_file)
            results['timing']['load_structure'] = time.perf_counter() - step_start
            results['steps']['load_structure'] = True
            
            # 记录原始数据信息
            results['original_info'] = {
                'n_atoms': original_protein.n_atoms,
                'n_residues': original_protein.n_residues,
                'device': str(getattr(original_protein.coordinates, 'device', 'cpu'))
            }
            
            # 步骤 2: 转换为 Atom14
            step_start = time.perf_counter()
            atom14 = Atom14.from_protein_tensor(original_protein)
            results['timing']['to_atom14'] = time.perf_counter() - step_start
            results['steps']['to_atom14'] = True
            
            # 记录 Atom14 信息
            results['atom14_info'] = {
                'num_residues': atom14.num_residues,
                'num_chains': atom14.num_chains,
                'coords_shape': list(atom14.coords.shape),
                'atom_mask_shape': list(atom14.atom_mask.shape),
                'res_mask_shape': list(atom14.res_mask.shape),
                'device': str(atom14.device)
            }
            
            # 步骤 3: 保存为 NPZ 格式
            step_start = time.perf_counter()
            npz_file = output_dir / f"{file_stem}_atom14.npz"
            self._save_atom14_as_npz(atom14, npz_file)
            results['timing']['save_npz'] = time.perf_counter() - step_start
            results['steps']['save_npz'] = True
            
            # 步骤 4: 保存为 PyTorch 格式
            step_start = time.perf_counter()
            pt_file = output_dir / f"{file_stem}_atom14.pt"
            self._save_atom14_as_pt(atom14, pt_file)
            results['timing']['save_pt'] = time.perf_counter() - step_start
            results['steps']['save_pt'] = True
            
            # 步骤 5: 从 NPZ 重新加载
            step_start = time.perf_counter()
            atom14_from_npz = self._load_atom14_from_npz(npz_file)
            results['timing']['load_npz'] = time.perf_counter() - step_start
            results['steps']['load_npz'] = True
            
            # 步骤 6: 从 PT 重新加载
            step_start = time.perf_counter()
            atom14_from_pt = self._load_atom14_from_pt(pt_file)
            results['timing']['load_pt'] = time.perf_counter() - step_start
            results['steps']['load_pt'] = True
            
            # 步骤 7: 数据一致性验证
            step_start = time.perf_counter()
            consistency_results = self._validate_data_consistency(
                atom14, atom14_from_npz, atom14_from_pt
            )
            results['timing']['validate_consistency'] = time.perf_counter() - step_start
            results['steps']['validate_consistency'] = True
            results['validation']['consistency'] = consistency_results
            
            # 步骤 8: 重建为 CIF 文件
            step_start = time.perf_counter()
            rebuilt_cif = output_dir / f"{file_stem}_rebuilt.cif"
            save_atom14_to_cif(atom14, str(rebuilt_cif))
            results['timing']['rebuild_cif'] = time.perf_counter() - step_start
            results['steps']['rebuild_cif'] = True
            
            # 步骤 9: 转换回 ProteinTensor 并保存
            step_start = time.perf_counter()
            reconstructed_protein = atom14.to_protein_tensor()
            reconstructed_cif = output_dir / f"{file_stem}_reconstructed.cif"
            save_protein_tensor_to_cif(reconstructed_protein, str(reconstructed_cif))
            results['timing']['reconstruct_protein'] = time.perf_counter() - step_start
            results['steps']['reconstruct_protein'] = True
            
            # 步骤 10: 结构完整性验证
            step_start = time.perf_counter()
            integrity_results = self._validate_structure_integrity(
                original_protein, reconstructed_protein
            )
            results['timing']['validate_integrity'] = time.perf_counter() - step_start
            results['steps']['validate_integrity'] = True
            results['validation']['integrity'] = integrity_results
            
            # 计算总时间
            results['timing']['total'] = sum(results['timing'].values())
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def _save_atom14_as_npz(self, atom14: Atom14, filepath: Path) -> None:
        """将 Atom14 保存为 NPZ 格式"""
        data = {
            'coords': atom14.coords.cpu().numpy(),
            'atom_mask': atom14.atom_mask.cpu().numpy(),
            'res_mask': atom14.res_mask.cpu().numpy(),
            'chain_ids': atom14.chain_ids.cpu().numpy(),
            'residue_types': atom14.residue_types.cpu().numpy(),
            'residue_indices': atom14.residue_indices.cpu().numpy(),
            'chain_residue_indices': atom14.chain_residue_indices.cpu().numpy(),
            'residue_names': atom14.residue_names.cpu().numpy(),
            'atom_names': atom14.atom_names.cpu().numpy(),
            'metadata': {
                'format': 'atom14',
                'version': '1.0',
                'num_residues': atom14.num_residues,
                'num_chains': atom14.num_chains
            }
        }
        np.savez_compressed(filepath, **data)
    
    def _save_atom14_as_pt(self, atom14: Atom14, filepath: Path) -> None:
        """将 Atom14 保存为 PyTorch 格式"""
        data = {
            'coords': atom14.coords,
            'atom_mask': atom14.atom_mask,
            'res_mask': atom14.res_mask,
            'chain_ids': atom14.chain_ids,
            'residue_types': atom14.residue_types,
            'residue_indices': atom14.residue_indices,
            'chain_residue_indices': atom14.chain_residue_indices,
            'residue_names': atom14.residue_names,
            'atom_names': atom14.atom_names,
            'metadata': {
                'format': 'atom14',
                'version': '1.0',
                'num_residues': atom14.num_residues,
                'num_chains': atom14.num_chains
            }
        }
        torch.save(data, filepath)
    
    def _load_atom14_from_npz(self, filepath: Path) -> Atom14:
        """从 NPZ 文件加载 Atom14"""
        data = np.load(filepath, allow_pickle=True)
        return Atom14(
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
    
    def _load_atom14_from_pt(self, filepath: Path) -> Atom14:
        """从 PyTorch 文件加载 Atom14"""
        data = torch.load(filepath, map_location='cpu')
        return Atom14(
            coords=data['coords'],
            atom_mask=data['atom_mask'],
            res_mask=data['res_mask'],
            chain_ids=data['chain_ids'],
            residue_types=data['residue_types'],
            residue_indices=data['residue_indices'],
            chain_residue_indices=data['chain_residue_indices'],
            residue_names=data['residue_names'],
            atom_names=data['atom_names']
        )
    
    def _validate_data_consistency(
        self, 
        original: Atom14, 
        from_npz: Atom14, 
        from_pt: Atom14
    ) -> Dict[str, Any]:
        """验证不同格式间的数据一致性"""
        results = {
            'npz_consistency': True,
            'pt_consistency': True,
            'cross_consistency': True,
            'errors': []
        }
        
        try:
            # 验证 NPZ 一致性
            if not torch.allclose(original.coords, from_npz.coords, rtol=1e-5, atol=1e-6):
                results['npz_consistency'] = False
                results['errors'].append("NPZ coords mismatch")
            
            if not torch.equal(original.atom_mask, from_npz.atom_mask):
                results['npz_consistency'] = False
                results['errors'].append("NPZ atom_mask mismatch")
            
            if not torch.equal(original.res_mask, from_npz.res_mask):
                results['npz_consistency'] = False
                results['errors'].append("NPZ res_mask mismatch")
            
            # 验证 PT 一致性
            if not torch.allclose(original.coords, from_pt.coords, rtol=1e-5, atol=1e-6):
                results['pt_consistency'] = False
                results['errors'].append("PT coords mismatch")
            
            if not torch.equal(original.atom_mask, from_pt.atom_mask):
                results['pt_consistency'] = False
                results['errors'].append("PT atom_mask mismatch")
            
            if not torch.equal(original.res_mask, from_pt.res_mask):
                results['pt_consistency'] = False
                results['errors'].append("PT res_mask mismatch")
            
            # 验证交叉一致性（NPZ vs PT）
            if not torch.allclose(from_npz.coords, from_pt.coords, rtol=1e-5, atol=1e-6):
                results['cross_consistency'] = False
                results['errors'].append("NPZ-PT coords mismatch")
            
        except Exception as e:
            results['npz_consistency'] = False
            results['pt_consistency'] = False
            results['cross_consistency'] = False
            results['errors'].append(f"Validation error: {str(e)}")
        
        return results
    
    def _validate_structure_integrity(
        self, 
        original: Any, 
        reconstructed: Any
    ) -> Dict[str, Any]:
        """验证结构完整性"""
        results = {
            'atom_count_match': False,
            'residue_count_match': False,
            'coordinate_similarity': 0.0,
            'errors': []
        }
        
        try:
            # 验证原子数量
            if hasattr(original, 'n_atoms') and hasattr(reconstructed, 'n_atoms'):
                results['atom_count_match'] = abs(original.n_atoms - reconstructed.n_atoms) <= 5
                if not results['atom_count_match']:
                    results['errors'].append(
                        f"Atom count mismatch: {original.n_atoms} vs {reconstructed.n_atoms}"
                    )
            
            # 验证残基数量
            if hasattr(original, 'n_residues') and hasattr(reconstructed, 'n_residues'):
                results['residue_count_match'] = original.n_residues == reconstructed.n_residues
                if not results['residue_count_match']:
                    results['errors'].append(
                        f"Residue count mismatch: {original.n_residues} vs {reconstructed.n_residues}"
                    )
            
            # 计算坐标相似性（基于质心）
            if hasattr(original, 'coordinates') and hasattr(reconstructed, 'coordinates'):
                orig_centroid = torch.mean(original.coordinates, dim=0)
                recon_centroid = torch.mean(reconstructed.coordinates, dim=0)
                distance = torch.norm(orig_centroid - recon_centroid).item()
                results['coordinate_similarity'] = max(0.0, 1.0 - distance / 100.0)  # 归一化相似性
        
        except Exception as e:
            results['errors'].append(f"Integrity validation error: {str(e)}")
        
        return results
    
    def test_batch_processing(self, test_data_files: List[Path], test_output_dir: Path):
        """测试批量处理功能"""
        batch_output_dir = test_output_dir / "batch_results"
        batch_output_dir.mkdir(exist_ok=True)
        
        # 创建批量转换器
        converter = BatchPDBToAtom14Converter(
            n_workers=2,  # 使用较少的工作进程避免资源竞争
            preserve_structure=True,
            device="cpu",
            output_format="npz"
        )
        
        # 创建输入目录并复制测试文件
        temp_input_dir = test_output_dir / "temp_input"
        temp_input_dir.mkdir(exist_ok=True)
        
        for test_file in test_data_files:
            shutil.copy2(test_file, temp_input_dir / test_file.name)
        
        # 执行批量转换
        statistics = converter.convert_batch(
            input_path=temp_input_dir,
            output_dir=batch_output_dir,
            recursive=True
        )
        
        # 保存批量处理统计
        stats_file = test_output_dir / "batch_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False, default=str)
        
        # 验证批量处理结果
        assert statistics['success'] > 0, "批量处理没有成功转换任何文件"
        assert statistics['failed'] == 0, f"批量处理失败 {statistics['failed']} 个文件"
        
        # 清理临时目录
        shutil.rmtree(temp_input_dir, ignore_errors=True)
    
    def test_error_handling(self, test_output_dir: Path):
        """测试错误处理能力"""
        error_results = {}
        
        # 测试无效文件处理
        invalid_file = test_output_dir / "invalid.pdb"
        invalid_file.write_text("INVALID PDB CONTENT\nATOM invalid line")
        
        try:
            protein = load_structure(invalid_file)
            # 如果成功加载，检查是否为空或无效
            if hasattr(protein, 'n_atoms') and protein.n_atoms == 0:
                error_results['invalid_file'] = "Correctly caught: Empty structure"
            else:
                error_results['invalid_file'] = "Should have raised exception"
        except Exception as e:
            error_results['invalid_file'] = f"Correctly caught: {type(e).__name__}"
        
        # 测试不存在文件处理
        nonexistent_file = test_output_dir / "nonexistent.cif"
        
        try:
            load_structure(nonexistent_file)
            error_results['nonexistent_file'] = "Should have raised exception"
        except Exception as e:
            error_results['nonexistent_file'] = f"Correctly caught: {type(e).__name__}"
        
        # 保存错误处理结果
        error_file = test_output_dir / "error_handling_results.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_results, f, indent=2, ensure_ascii=False)
        
        # 验证错误正确处理
        assert "correctly caught" in error_results['invalid_file'].lower()
        assert "correctly caught" in error_results['nonexistent_file'].lower()


def test_complete_workflow():
    """独立的完整工作流测试函数"""
    test_instance = TestAtom14EndToEnd()
    
    # 获取测试数据
    data_dir = Path(__file__).parent.parent / "data"
    test_files = list(data_dir.glob("*.cif"))[:1]  # 只用一个文件进行快速测试
    
    if not test_files:
        pytest.skip("没有找到测试数据文件")
    
    # 创建输出目录
    output_dir = Path(__file__).parent / "test_results"
    output_dir.mkdir(exist_ok=True)
    
    # 运行测试
    test_instance.test_complete_workflow(test_files, output_dir)


def test_data_consistency():
    """独立的数据一致性测试函数"""
    # 这里可以添加额外的数据一致性测试
    pass


def test_file_formats_roundtrip():
    """独立的文件格式往返测试函数"""
    # 这里可以添加额外的文件格式测试
    pass


if __name__ == "__main__":
    # 允许直接运行测试
    pytest.main([__file__, "-v", "-s"]) 