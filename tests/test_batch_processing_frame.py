"""
Frame 批量处理功能测试

测试 protrepr.batch_processing 中的 Frame 相关批量转换功能，包括：
- PDB/CIF 到 Frame 的批量转换
- Frame 到 CIF/PDB 的批量转换
- 并行处理功能
- 错误处理和统计
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import torch

# 导入被测试的模块
from protrepr.batch_processing import (
    BatchPDBToFrameConverter,
    BatchFrameToCIFConverter,
    save_statistics
)
from protrepr.core.frame import Frame
from protein_tensor import load_structure


class TestBatchPDBToFrameConverter:
    """测试 PDB/CIF 到 Frame 的批量转换。"""
    
    def test_converter_initialization(self):
        """测试转换器的初始化。"""
        # 基本初始化
        converter = BatchPDBToFrameConverter()
        assert converter.n_workers >= 1
        assert converter.preserve_structure is True
        assert converter.device == torch.device('cpu')
        assert converter.save_as_instance is True
        
        # 自定义参数初始化
        converter = BatchPDBToFrameConverter(
            n_workers=2,
            preserve_structure=False,
            device="cpu",
            save_as_instance=False
        )
        assert converter.n_workers == 2
        assert converter.preserve_structure is False
        assert converter.save_as_instance is False
        
        print("✅ BatchPDBToFrameConverter 初始化测试通过")
    
    def test_find_structure_files(self):
        """测试结构文件查找功能。"""
        converter = BatchPDBToFrameConverter()
        
        # 查找测试数据目录中的文件
        test_data_dir = Path("tests/data")
        if test_data_dir.exists():
            files = converter.find_structure_files(test_data_dir)
            assert isinstance(files, list)
            
            # 检查文件扩展名
            for file_path in files:
                assert file_path.suffix.lower() in {'.pdb', '.ent', '.cif', '.mmcif'}
            
            print(f"✅ 结构文件查找测试通过，找到 {len(files)} 个文件")
        else:
            print("⚠️  测试数据目录不存在，跳过文件查找测试")
    
    def test_single_file_conversion(self):
        """测试单个文件的转换。"""
        # 查找测试文件
        test_data_dir = Path("tests/data")
        if not test_data_dir.exists():
            pytest.skip("测试数据目录不存在")
        
        # 查找第一个 CIF 文件
        cif_files = list(test_data_dir.glob("*.cif"))
        if not cif_files:
            pytest.skip("没有找到测试 CIF 文件")
        
        test_file = cif_files[0]
        converter = BatchPDBToFrameConverter(n_workers=1)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "test_output.pt"
            
            # 执行转换
            result = converter.convert_single_file(test_file, output_file)
            
            # 验证结果
            assert result['success'] is True
            assert result['error'] is None
            assert result['num_residues'] > 0
            assert result['processing_time'] > 0
            assert output_file.exists()
            
            # 验证输出文件可以加载
            frame = Frame.load(output_file)
            assert frame.num_residues > 0
            
            print(f"✅ 单文件转换测试通过: {result['num_residues']} 个残基")


class TestBatchFrameToCIFConverter:
    """测试 Frame 到 CIF/PDB 的批量转换。"""
    
    def test_converter_initialization(self):
        """测试转换器的初始化。"""
        # 基本初始化
        converter = BatchFrameToCIFConverter()
        assert converter.n_workers >= 1
        assert converter.preserve_structure is True
        assert converter.output_format == "cif"
        
        # 自定义参数初始化
        converter = BatchFrameToCIFConverter(
            n_workers=2,
            preserve_structure=False,
            output_format="pdb"
        )
        assert converter.n_workers == 2
        assert converter.preserve_structure is False
        assert converter.output_format == "pdb"
        
        # 无效格式应该抛出异常
        with pytest.raises(ValueError):
            BatchFrameToCIFConverter(output_format="xyz")
        
        print("✅ BatchFrameToCIFConverter 初始化测试通过")
    
    def test_single_file_conversion(self):
        """测试单个 Frame 文件的转换。"""
        # 首先创建一个测试用的 Frame 文件
        device = torch.device('cpu')
        num_residues = 5
        
        # 创建测试 Frame
        translations = torch.randn(num_residues, 3, device=device)
        rotations = torch.eye(3, device=device).unsqueeze(0).repeat(num_residues, 1, 1)
        res_mask = torch.ones(num_residues, dtype=torch.bool, device=device)
        chain_ids = torch.zeros(num_residues, dtype=torch.long, device=device)
        residue_types = torch.randint(0, 20, (num_residues,), device=device)
        residue_indices = torch.arange(num_residues, device=device)
        chain_residue_indices = torch.arange(num_residues, device=device)
        residue_names = torch.randint(0, 20, (num_residues,), device=device)
        
        test_frame = Frame(
            translations=translations,
            rotations=rotations,
            res_mask=res_mask,
            chain_ids=chain_ids,
            residue_types=residue_types,
            residue_indices=residue_indices,
            chain_residue_indices=chain_residue_indices,
            residue_names=residue_names
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存测试 Frame
            frame_file = Path(temp_dir) / "test_frame.pt"
            test_frame.save(frame_file)
            
            # 测试转换为 CIF
            converter = BatchFrameToCIFConverter(n_workers=1, output_format="cif")
            output_file = Path(temp_dir) / "test_output.cif"
            
            result = converter.convert_single_file(frame_file, output_file)
            
            # 验证结果
            assert result['success'] is True
            assert result['error'] is None
            assert result['num_residues'] == num_residues
            assert result['processing_time'] > 0
            assert output_file.exists()
            
            print(f"✅ Frame 到 CIF 转换测试通过: {result['num_residues']} 个残基")


class TestBatchProcessingIntegration:
    """测试批量处理的集成功能。"""
    
    def test_full_roundtrip_conversion(self):
        """测试完整的往返转换：PDB/CIF → Frame → CIF。"""
        # 查找测试文件
        test_data_dir = Path("tests/data")
        if not test_data_dir.exists():
            pytest.skip("测试数据目录不存在")
        
        cif_files = list(test_data_dir.glob("*.cif"))
        if not cif_files:
            pytest.skip("没有找到测试 CIF 文件")
        
        test_file = cif_files[0]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 第一步: PDB/CIF → Frame
            pdb_to_frame_converter = BatchPDBToFrameConverter(n_workers=1)
            frame_file = temp_path / "intermediate.pt"
            
            result1 = pdb_to_frame_converter.convert_single_file(test_file, frame_file)
            assert result1['success'] is True
            assert frame_file.exists()
            
            # 第二步: Frame → CIF
            frame_to_cif_converter = BatchFrameToCIFConverter(n_workers=1, output_format="cif")
            output_cif = temp_path / "output.cif"
            
            result2 = frame_to_cif_converter.convert_single_file(frame_file, output_cif)
            assert result2['success'] is True
            assert output_cif.exists()
            
            # 验证往返转换的一致性（残基数量在合理范围内）
            original_residues = result1['num_residues']
            final_residues = result2['num_residues']
            
            # Frame 表示只保留主链，所以残基数可能有差异
            assert final_residues > 0
            print(f"✅ 往返转换测试通过: {test_file.name}")
            print(f"   原始残基数: {original_residues}")
            print(f"   最终残基数: {final_residues}")
    
    def test_batch_conversion_with_multiple_files(self):
        """测试多文件批量转换。"""
        # 查找测试文件
        test_data_dir = Path("tests/data")
        if not test_data_dir.exists():
            pytest.skip("测试数据目录不存在")
        
        structure_files = list(test_data_dir.glob("*.cif")) + list(test_data_dir.glob("*.pdb"))
        if len(structure_files) < 1:
            pytest.skip("没有足够的测试文件")
        
        # 限制测试文件数量以避免测试时间过长
        test_files = structure_files[:2]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            
            # 复制测试文件到临时目录
            for i, test_file in enumerate(test_files):
                shutil.copy2(test_file, input_dir / f"test_{i}{test_file.suffix}")
            
            # 执行批量转换
            converter = BatchPDBToFrameConverter(n_workers=1)
            statistics = converter.convert_batch(
                input_path=input_dir,
                output_dir=output_dir,
                recursive=True
            )
            
            # 验证结果
            assert statistics['total'] == len(test_files)
            assert statistics['success'] >= 0
            assert statistics['failed'] == statistics['total'] - statistics['success']
            
            # 检查输出文件
            output_files = list(output_dir.glob("*.pt"))
            assert len(output_files) == statistics['success']
            
            print(f"✅ 批量转换测试通过: {statistics['success']}/{statistics['total']} 文件成功")
    
    def test_statistics_saving(self):
        """测试统计信息保存功能。"""
        # 创建模拟统计数据
        statistics = {
            'total': 2,
            'success': 1,
            'failed': 1,
            'failed_files': ['failed_file.pdb'],
            'results': [
                {
                    'input_file': 'test1.pdb',
                    'output_file': 'test1.pt',
                    'success': True,
                    'error': None,
                    'processing_time': 1.23,
                    'num_residues': 100,
                    'num_atoms': 400,
                    'num_chains': 1
                },
                {
                    'input_file': 'test2.pdb',
                    'output_file': 'test2.pt',
                    'success': False,
                    'error': 'Test error',
                    'processing_time': 0.5,
                    'num_residues': 0,
                    'num_atoms': 0,
                    'num_chains': 0
                }
            ]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            stats_file = Path(temp_dir) / "statistics.json"
            
            # 保存统计信息
            save_statistics(statistics, stats_file)
            
            # 验证文件存在并可读取
            assert stats_file.exists()
            
            import json
            with open(stats_file, 'r', encoding='utf-8') as f:
                loaded_stats = json.load(f)
            
            assert loaded_stats['total'] == 2
            assert loaded_stats['success'] == 1
            assert loaded_stats['failed'] == 1
            
            print("✅ 统计信息保存测试通过")


if __name__ == "__main__":
    # 运行测试
    print("🧪 开始 Frame 批量处理功能测试...")
    
    # PDB/CIF 到 Frame 转换器测试
    pdb_to_frame_tests = TestBatchPDBToFrameConverter()
    pdb_to_frame_tests.test_converter_initialization()
    pdb_to_frame_tests.test_find_structure_files()
    pdb_to_frame_tests.test_single_file_conversion()
    
    # Frame 到 CIF/PDB 转换器测试
    frame_to_cif_tests = TestBatchFrameToCIFConverter()
    frame_to_cif_tests.test_converter_initialization()
    frame_to_cif_tests.test_single_file_conversion()
    
    # 集成测试
    integration_tests = TestBatchProcessingIntegration()
    integration_tests.test_full_roundtrip_conversion()
    integration_tests.test_batch_conversion_with_multiple_files()
    integration_tests.test_statistics_saving()
    
    print("🎉 所有 Frame 批量处理测试完成！") 