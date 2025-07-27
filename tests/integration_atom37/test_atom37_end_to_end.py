"""
Atom37 端到端集成测试 (完整版本)

本模块提供完整的 Atom37 工作流集成测试，覆盖从原始结构文件到最终输出的
完整数据流，确保所有转换步骤的正确性和数据一致性。

重点测试：
1. 调用 batch_pdb_to_atom37.py 脚本进行批量转换
2. 验证 Atom37 实例和字典格式的保存/加载
3. 将结果转换为 CIF 文件进行可视化验证
4. 纯 PyTorch 后端，不使用 NumPy
5. 保存所有中间结果供手动验证
6. 验证 37 个原子槽位的正确映射

测试流程：
1. CIF/PDB → 批量脚本 → .pt 文件
2. .pt 文件 → Atom37 实例/字典 → 验证一致性
3. Atom37 → CIF 重建 → 可视化验证
4. 完整工作流验证和性能统计
"""

import sys
import subprocess
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import pytest
import torch

# 添加源码路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from protein_tensor import load_structure
from protrepr.core.atom37 import Atom37
from protrepr.representations.atom37_converter import (
    protein_tensor_to_atom37,
    atom37_to_protein_tensor,
    ATOM37_ATOM_TYPES
)


class TestAtom37EndToEnd:
    """Atom37 端到端集成测试类"""
    
    @pytest.fixture(scope="class")
    def test_output_dir(self) -> Path:
        """创建测试输出目录"""
        output_dir = Path(__file__).parent / "test_results"
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    @pytest.fixture(scope="class") 
    def test_data_files(self) -> List[Path]:
        """获取测试数据文件列表"""
        test_data_dir = Path(__file__).resolve().parent.parent / "data"
        
        # 收集所有可用的测试文件
        cif_files = list(test_data_dir.rglob("*.cif"))
        pdb_files = list(test_data_dir.rglob("*.pdb"))
        
        all_files = cif_files + pdb_files
        
        # 确保有足够的测试文件
        assert len(all_files) >= 3, f"需要至少 3 个测试文件，但只找到 {len(all_files)} 个"
        
        # 返回前 5 个文件用于测试
        return all_files[:5]
    
    @pytest.fixture(scope="class")
    def batch_script_path(self) -> Path:
        """获取批量转换脚本路径"""
        script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "batch_pdb_to_atom37.py"
        assert script_path.exists(), f"批量转换脚本不存在: {script_path}"
        return script_path

    def test_batch_script_basic_conversion(
        self, 
        test_data_files: List[Path], 
        test_output_dir: Path,
        batch_script_path: Path
    ) -> None:
        """测试批量转换脚本的基本功能"""
        
        # 创建临时输入目录
        temp_input_dir = test_output_dir / "temp_input"
        temp_input_dir.mkdir(exist_ok=True)
        
        # 复制测试文件到临时目录
        for file_path in test_data_files[:3]:  # 使用前 3 个文件
            shutil.copy2(file_path, temp_input_dir)
        
        # 设置输出目录
        batch_output_dir = test_output_dir / "batch_conversion_output"
        batch_output_dir.mkdir(exist_ok=True)
        
        # 运行批量转换脚本
        cmd = [
            sys.executable,
            str(batch_script_path),
            str(temp_input_dir),
            str(batch_output_dir),
            "--workers", "2",
            "--device", "cpu",
            "--save-as-instance"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60秒超时
        )
        
        # 验证脚本执行成功
        assert result.returncode == 0, f"批量转换脚本执行失败:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        
        # 验证输出文件
        output_files = list(batch_output_dir.rglob("*.pt"))
        assert len(output_files) >= 1, "应该至少生成一个 .pt 文件"
        
        # 验证统计文件
        stats_file = batch_output_dir / "conversion_statistics.json"
        assert stats_file.exists(), "应该生成统计文件"
        
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        assert stats["success"] >= 1, "应该至少成功转换一个文件"
        assert "total_time" in stats, "统计信息应包含总时间"
        
        print(f"批量转换统计: {stats}")

    def test_atom37_instance_loading_and_validation(
        self,
        test_data_files: List[Path],
        test_output_dir: Path
    ) -> None:
        """测试 Atom37 实例的加载和验证"""
        
        # 使用第一个测试文件
        test_file = test_data_files[0]
        
        # 加载原始结构
        protein_tensor = load_structure(test_file)
        
        # 转换为 Atom37
        atom37_instance = Atom37.from_protein_tensor(protein_tensor)
        
        # 保存 Atom37 实例
        instance_file = test_output_dir / f"atom37_instance_{test_file.stem}.pt"
        atom37_instance.save(instance_file)
        
        # 重新加载并验证
        loaded_atom37 = Atom37.load(instance_file)
        
        # 验证数据一致性
        assert torch.allclose(atom37_instance.coords, loaded_atom37.coords, atol=1e-5), \
            "重新加载的坐标数据不一致"
        
        assert torch.equal(atom37_instance.atom_mask, loaded_atom37.atom_mask), \
            "重新加载的掩码数据不一致"
        
        assert torch.equal(atom37_instance.residue_types, loaded_atom37.residue_types), \
            "重新加载的残基类型不一致"
        
        assert torch.equal(atom37_instance.chain_ids, loaded_atom37.chain_ids), \
            "重新加载的链ID不一致"
        
        # 验证 Atom37 特有的 37 个原子槽位
        assert atom37_instance.coords.shape[-2] == 37, \
            f"Atom37 应该有 37 个原子槽位，但实际有 {atom37_instance.coords.shape[-2]} 个"
        
        assert atom37_instance.atom_mask.shape[-1] == 37, \
            f"Atom37 掩码应该有 37 个槽位，但实际有 {atom37_instance.atom_mask.shape[-1]} 个"
        
        print(f"Atom37 实例验证成功: {test_file.name}")

    def test_atom37_to_cif_conversion(
        self,
        test_data_files: List[Path],
        test_output_dir: Path
    ) -> None:
        """测试 Atom37 到 CIF 文件的转换"""
        
        # 使用第一个测试文件
        test_file = test_data_files[0]
        
        # 加载原始结构并转换为 Atom37
        protein_tensor = load_structure(test_file)
        atom37_instance = Atom37.from_protein_tensor(protein_tensor)
        
        # 转换为 CIF 文件
        output_cif = test_output_dir / f"reconstructed_{test_file.stem}.cif"
        atom37_instance.to_cif(output_cif)
        
        # 验证 CIF 文件存在且不为空
        assert output_cif.exists(), "CIF 文件应该被创建"
        assert output_cif.stat().st_size > 0, "CIF 文件不应为空"
        
        # 尝试重新加载生成的 CIF 文件
        reconstructed_protein = load_structure(output_cif)
        reconstructed_atom37 = Atom37.from_protein_tensor(reconstructed_protein)
        
        # 验证基本结构信息保持一致
        assert len(atom37_instance.residue_types) == len(reconstructed_atom37.residue_types), \
            "重构后的残基数量不一致"
        
        print(f"Atom37 到 CIF 转换成功: {output_cif.name}")

    def test_atom37_round_trip_conversion(
        self,
        test_data_files: List[Path],
        test_output_dir: Path
    ) -> None:
        """测试 Atom37 的往返转换 (round-trip)"""
        
        # 使用第二个测试文件
        test_file = test_data_files[1] if len(test_data_files) > 1 else test_data_files[0]
        
        # 原始结构 → Atom37 → ProteinTensor → Atom37
        
        # 第一步：加载原始结构
        original_protein = load_structure(test_file)
        
        # 第二步：转换为 Atom37
        atom37_instance_1 = Atom37.from_protein_tensor(original_protein)
        
        # 第三步：转换回 ProteinTensor
        reconstructed_protein = atom37_to_protein_tensor(atom37_instance_1)
        
        # 第四步：再次转换为 Atom37
        atom37_data_2 = protein_tensor_to_atom37(reconstructed_protein)
        atom37_instance_2 = Atom37(**atom37_data_2)
        
        # 验证往返转换的一致性
        assert torch.allclose(
            atom37_instance_1.coords, 
            atom37_instance_2.coords, 
            atol=1e-3
        ), "往返转换后坐标不一致"
        
        assert torch.equal(
            atom37_instance_1.mask, 
            atom37_instance_2.mask
        ), "往返转换后掩码不一致"
        
        assert atom37_instance_1.residue_types == atom37_instance_2.residue_types, \
            "往返转换后残基类型不一致"
        
        # 保存往返转换结果
        round_trip_file = test_output_dir / f"atom37_round_trip_{test_file.stem}.pt"
        atom37_instance_2.save(round_trip_file)
        
        print(f"Atom37 往返转换验证成功: {test_file.name}")

    def test_atom37_batch_processing_performance(
        self,
        test_data_files: List[Path],
        test_output_dir: Path
    ) -> None:
        """测试 Atom37 批量处理性能"""
        
        from protrepr.batch_processing import BatchPDBToAtom37Converter
        
        # 创建批量转换器
        converter = BatchPDBToAtom37Converter(
            n_workers=2,
            device='cpu',
            save_as_instance=True
        )
        
        # 创建临时输入目录
        perf_input_dir = test_output_dir / "perf_input"
        perf_input_dir.mkdir(exist_ok=True)
        
        # 复制所有测试文件
        for file_path in test_data_files:
            shutil.copy2(file_path, perf_input_dir)
        
        # 设置输出目录
        perf_output_dir = test_output_dir / "perf_output"
        perf_output_dir.mkdir(exist_ok=True)
        
        # 执行批量转换并测量时间
        start_time = time.time()
        
        statistics = converter.convert_batch(
            input_path=perf_input_dir,
            output_dir=perf_output_dir
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 验证性能统计
        assert statistics["success"] >= 1, "应该至少成功处理一个文件"
        assert statistics["total_time"] > 0, "总时间应该大于 0"
        assert total_time < 60, "批量处理时间不应超过 60 秒"
        
        # 计算平均处理时间
        avg_time_per_file = total_time / statistics["success"] if statistics["success"] > 0 else 0
        
        print(f"Atom37 批量处理性能统计:")
        print(f"  - 处理文件数: {statistics['success']}")
        print(f"  - 总时间: {total_time:.2f} 秒")
        print(f"  - 平均每文件: {avg_time_per_file:.2f} 秒")
        
        # 验证输出文件质量
        output_files = list(perf_output_dir.rglob("*.pt"))
        assert len(output_files) == statistics["success"], "输出文件数量应与成功处理数量一致"
        
        # 随机验证一个输出文件
        if output_files:
            test_output_file = output_files[0]
            loaded_atom37 = Atom37.load(test_output_file)
            
            # 验证数据完整性
            assert loaded_atom37.coords.numel() > 0, "加载的坐标数据不应为空"
            assert loaded_atom37.coords.shape[-2] == 37, "应该有 37 个原子槽位"
            assert len(loaded_atom37.residue_types) > 0, "残基类型列表不应为空"

    def test_atom37_device_transfer(
        self,
        test_data_files: List[Path],
        test_output_dir: Path
    ) -> None:
        """测试 Atom37 设备传输功能"""
        
        # 使用第一个测试文件
        test_file = test_data_files[0]
        
        # 加载并转换为 Atom37
        protein_tensor = load_structure(test_file)
        atom37_data = protein_tensor_to_atom37(protein_tensor)
        atom37_cpu = Atom37(**atom37_data)
        
        # 验证初始设备为 CPU
        assert atom37_cpu.device.type == 'cpu', "初始数据应在 CPU 上"
        
        # 如果有 CUDA 可用，测试 GPU 传输
        if torch.cuda.is_available():
            atom37_gpu = atom37_cpu.to('cuda')
            
            # 验证设备传输
            assert atom37_gpu.device.type == 'cuda', "数据应被传输到 GPU"
            assert atom37_cpu.device.type == 'cpu', "原始数据应仍在 CPU"
            
            # 验证数据一致性
            assert torch.allclose(
                atom37_cpu.coords.cuda(), 
                atom37_gpu.coords, 
                atol=1e-5
            ), "GPU 传输后数据不一致"
            
            # 传输回 CPU
            atom37_back_to_cpu = atom37_gpu.to('cpu')
            assert atom37_back_to_cpu.device.type == 'cpu', "数据应被传输回 CPU"
            
            print("Atom37 CUDA 设备传输测试成功")
        else:
            print("跳过 CUDA 测试 (设备不可用)")

    def test_atom37_error_handling(
        self,
        test_output_dir: Path
    ) -> None:
        """测试 Atom37 错误处理"""
        
        # 测试无效数据的处理
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            # 尝试创建形状不匹配的 Atom37
            invalid_coords = torch.randn(10, 25, 3)  # 错误的原子数量 (应该是 37)
            invalid_mask = torch.ones(10, 37)  # 正确的掩码形状
            
            Atom37(
                coords=invalid_coords,
                mask=invalid_mask,
                residue_types=['ALA'] * 10,
                chain_ids=['A'] * 10,
                residue_indices=list(range(10))
            )
        
        # 测试文件不存在的处理
        with pytest.raises(FileNotFoundError):
            Atom37.load("nonexistent_file.pt")
        
        print("Atom37 错误处理测试成功")

    def test_atom37_complete_workflow_summary(
        self,
        test_data_files: List[Path],
        test_output_dir: Path
    ) -> None:
        """完整工作流总结测试"""
        
        # 收集所有测试结果
        results_summary = {
            "input_files": len(test_data_files),
            "output_directory": str(test_output_dir),
            "generated_files": [],
            "atom37_specific_checks": []
        }
        
        # 收集生成的文件
        for file_path in test_output_dir.rglob("*"):
            if file_path.is_file():
                results_summary["generated_files"].append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "type": file_path.suffix
                })
        
        # Atom37 特有验证
        pt_files = list(test_output_dir.rglob("*.pt"))
        for pt_file in pt_files[:3]:  # 验证前 3 个文件
            try:
                atom37 = Atom37.load(pt_file)
                
                check_result = {
                    "file": pt_file.name,
                    "atom_slots": atom37.coords.shape[-2],
                    "residue_count": len(atom37.residue_types),
                    "coordinate_range": {
                        "min": float(atom37.coords.min()),
                        "max": float(atom37.coords.max())
                    },
                    "valid": atom37.coords.shape[-2] == 37
                }
                
                results_summary["atom37_specific_checks"].append(check_result)
                
            except Exception as e:
                results_summary["atom37_specific_checks"].append({
                    "file": pt_file.name,
                    "error": str(e),
                    "valid": False
                })
        
        # 保存测试总结
        summary_file = test_output_dir / "atom37_workflow_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        # 验证关键指标
        assert len(results_summary["generated_files"]) >= 5, "应该生成至少 5 个文件"
        assert all(check.get("valid", False) for check in results_summary["atom37_specific_checks"]), \
            "所有 Atom37 文件都应该有效"
        
        print(f"Atom37 完整工作流测试完成，总结保存至: {summary_file}")
        print(f"生成文件数量: {len(results_summary['generated_files'])}")
        print(f"Atom37 验证通过数量: {len([c for c in results_summary['atom37_specific_checks'] if c.get('valid')])}")


if __name__ == "__main__":
    # 直接运行此文件时的测试
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建简单的测试用例
        test_data_dir = Path(__file__).resolve().parent.parent / "data"
        test_files = list(test_data_dir.rglob("*.cif"))[:2]
        
        if test_files:
            print("运行 Atom37 端到端测试示例...")
            
            # 测试基本转换
            protein_tensor = load_structure(test_files[0])
            atom37_data = protein_tensor_to_atom37(protein_tensor)
            atom37_instance = Atom37(**atom37_data)
            
            # 保存和加载测试
            test_file = temp_path / "test_atom37.pt"
            atom37_instance.save(test_file)
            loaded_atom37 = Atom37.load(test_file)
            
            print(f"✓ 基本转换测试成功")
            print(f"  - 原子槽位数: {atom37_instance.coords.shape[-2]}")
            print(f"  - 残基数量: {len(atom37_instance.residue_types)}")
            print(f"  - 文件大小: {test_file.stat().st_size} bytes")
            
        else:
            print("未找到测试数据文件") 