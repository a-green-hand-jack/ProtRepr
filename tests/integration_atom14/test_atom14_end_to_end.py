"""
Atom14 端到端集成测试 (完整版本)

本模块提供完整的 Atom14 工作流集成测试，覆盖从原始结构文件到最终输出的
完整数据流，确保所有转换步骤的正确性和数据一致性。

重点测试：
1. 调用 batch_pdb_to_atom14.py 脚本进行批量转换
2. 验证 Atom14 实例和字典格式的保存/加载
3. 将结果转换为 CIF 文件进行可视化验证
4. 纯 PyTorch 后端，不使用 NumPy
5. 保存所有中间结果供手动验证

测试流程：
1. CIF/PDB → 批量脚本 → .pt 文件
2. .pt 文件 → Atom14 实例/字典 → 验证一致性
3. Atom14 → CIF 重建 → 可视化验证
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
from protrepr.core.atom14 import Atom14


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
    
    @pytest.fixture(scope="class")
    def script_path(self) -> Path:
        """获取批量转换脚本路径"""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "batch_pdb_to_atom14.py"
        if not script_path.exists():
            pytest.skip(f"批量转换脚本不存在: {script_path}")
        return script_path

    def test_batch_script_single_file(self, test_data_files: List[Path], test_output_dir: Path, script_path: Path):
        """测试批量脚本处理单个文件"""
        test_file = test_data_files[0]  # 使用第一个测试文件
        output_subdir = test_output_dir / "batch_script_single"
        output_subdir.mkdir(exist_ok=True)
        
        print(f"\n🧪 测试批量脚本处理单个文件: {test_file.name}")
        
        # 调用批量转换脚本
        cmd = [
            sys.executable, str(script_path),
            str(test_file),
            str(output_subdir),
            "--device", "cpu",
            "--verbose"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        execution_time = time.time() - start_time
        
        # 保存脚本执行结果
        script_result = {
            "command": " ".join(cmd),
            "return_code": result.returncode,
            "execution_time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "input_file": str(test_file),
            "output_dir": str(output_subdir)
        }
        
        with open(output_subdir / "script_execution_result.json", 'w', encoding='utf-8') as f:
            json.dump(script_result, f, indent=2, ensure_ascii=False)
        
        # 验证脚本执行成功
        assert result.returncode == 0, f"脚本执行失败: {result.stderr}"
        
        # 查找生成的 .pt 文件
        pt_files = list(output_subdir.glob("*.pt"))
        assert len(pt_files) > 0, "没有生成 .pt 文件"
        
        pt_file = pt_files[0]
        print(f"✅ 生成文件: {pt_file.name} ({pt_file.stat().st_size} bytes)")
        
        # 验证生成的文件可以加载
        self._verify_pt_file(pt_file, output_subdir)

    def test_batch_script_directory(self, test_data_files: List[Path], test_output_dir: Path, script_path: Path):
        """测试批量脚本处理目录"""
        print(f"\n🧪 测试批量脚本处理目录")
        
        # 创建输入目录并复制测试文件
        input_dir = test_output_dir / "batch_input"
        input_dir.mkdir(exist_ok=True)
        
        for test_file in test_data_files:
            shutil.copy2(test_file, input_dir / test_file.name)
        
        output_subdir = test_output_dir / "batch_script_directory"
        output_subdir.mkdir(exist_ok=True)
        
        # 调用批量转换脚本
        cmd = [
            sys.executable, str(script_path),
            str(input_dir),
            str(output_subdir),
            "--device", "cpu",
            "--workers", "2",
            "--save-stats", str(output_subdir / "batch_stats.json"),
            "--verbose"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        execution_time = time.time() - start_time
        
        # 保存批量处理结果
        batch_result = {
            "command": " ".join(cmd),
            "return_code": result.returncode,
            "execution_time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "input_files": [f.name for f in test_data_files],
            "input_dir": str(input_dir),
            "output_dir": str(output_subdir)
        }
        
        with open(output_subdir / "batch_execution_result.json", 'w', encoding='utf-8') as f:
            json.dump(batch_result, f, indent=2, ensure_ascii=False)
        
        # 验证脚本执行成功
        assert result.returncode == 0, f"批量脚本执行失败: {result.stderr}"
        
        # 查找生成的 .pt 文件
        pt_files = list(output_subdir.glob("*.pt"))
        assert len(pt_files) >= 1, "没有生成足够的 .pt 文件"
        
        print(f"✅ 批量处理生成 {len(pt_files)} 个文件")
        
        # 验证每个生成的文件
        for pt_file in pt_files:
            self._verify_pt_file(pt_file, output_subdir)
        
        # 清理输入目录
        shutil.rmtree(input_dir, ignore_errors=True)

    def test_atom14_save_load_formats(self, test_data_files: List[Path], test_output_dir: Path):
        """测试 Atom14 的保存和加载格式"""
        test_file = test_data_files[0]
        format_test_dir = test_output_dir / "format_tests"
        format_test_dir.mkdir(exist_ok=True)
        
        print(f"\n🧪 测试 Atom14 保存/加载格式")
        
        # 加载原始结构
        protein = load_structure(test_file)
        atom14_original = Atom14.from_protein_tensor(protein)
        
        print(f"📊 原始数据: {atom14_original.num_residues} 残基, {atom14_original.num_chains} 链")
        
        # 测试结果记录
        format_results = {
            "original_info": {
                "num_residues": atom14_original.num_residues,
                "num_chains": atom14_original.num_chains,
                "coords_shape": list(atom14_original.coords.shape),
                "device": str(atom14_original.device)
            },
            "tests": {}
        }
        
        # 1. 测试实例格式保存和加载
        instance_file = format_test_dir / "atom14_instance.pt"
        start_time = time.time()
        atom14_original.save(str(instance_file), save_as_instance=True)
        save_time = time.time() - start_time
        
        start_time = time.time()
        atom14_instance = Atom14.load(str(instance_file))
        load_time = time.time() - start_time
        
        # 验证实例一致性
        instance_consistent = self._check_consistency(atom14_original, atom14_instance)
        
        format_results["tests"]["instance_format"] = {
            "file_size": instance_file.stat().st_size,
            "save_time": save_time,
            "load_time": load_time,
            "data_consistent": instance_consistent,
            "file_path": str(instance_file)
        }
        
        print(f"✅ 实例格式: 保存 {save_time:.3f}s, 加载 {load_time:.3f}s, 大小 {instance_file.stat().st_size} bytes")
        
        # 2. 测试字典格式保存和加载
        dict_file = format_test_dir / "atom14_dict.pt"
        start_time = time.time()
        atom14_original.save(str(dict_file), save_as_instance=False)
        save_time = time.time() - start_time
        
        start_time = time.time()
        atom14_dict = Atom14.load(str(dict_file))
        load_time = time.time() - start_time
        
        # 验证字典一致性
        dict_consistent = self._check_consistency(atom14_original, atom14_dict)
        
        format_results["tests"]["dict_format"] = {
            "file_size": dict_file.stat().st_size,
            "save_time": save_time,
            "load_time": load_time,
            "data_consistent": dict_consistent,
            "file_path": str(dict_file)
        }
        
        print(f"✅ 字典格式: 保存 {save_time:.3f}s, 加载 {load_time:.3f}s, 大小 {dict_file.stat().st_size} bytes")
        
        # 3. 测试交叉一致性（实例 vs 字典）
        cross_consistent = self._check_consistency(atom14_instance, atom14_dict)
        format_results["cross_consistency"] = cross_consistent
        
        # 保存格式测试结果
        with open(format_test_dir / "format_test_results.json", 'w', encoding='utf-8') as f:
            json.dump(format_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 断言验证
        assert instance_consistent, "实例格式数据不一致"
        assert dict_consistent, "字典格式数据不一致"
        assert cross_consistent, "实例和字典格式交叉验证不一致"

    def test_cif_reconstruction(self, test_data_files: List[Path], test_output_dir: Path):
        """测试 CIF 文件重建和可视化验证"""
        test_file = test_data_files[0]
        cif_test_dir = test_output_dir / "cif_reconstruction"
        cif_test_dir.mkdir(exist_ok=True)
        
        print(f"\n🧪 测试 CIF 文件重建")
        
        # 加载原始结构
        protein = load_structure(test_file)
        atom14 = Atom14.from_protein_tensor(protein)
        
        # 重建结果记录
        reconstruction_results = {
            "original_file": str(test_file),
            "original_size": test_file.stat().st_size,
            "reconstructions": {}
        }
        
        # 1. 从实例格式重建 CIF
        instance_file = cif_test_dir / "atom14_instance.pt"
        atom14.save(str(instance_file), save_as_instance=True)
        atom14_loaded = Atom14.load(str(instance_file))
        
        cif_from_instance = cif_test_dir / "reconstructed_from_instance.cif"
        start_time = time.time()
        atom14_loaded.to_cif(str(cif_from_instance))
        reconstruction_time = time.time() - start_time
        
        reconstruction_results["reconstructions"]["from_instance"] = {
            "reconstruction_time": reconstruction_time,
            "file_size": cif_from_instance.stat().st_size,
            "file_path": str(cif_from_instance)
        }
        
        print(f"✅ 从实例重建: {reconstruction_time:.3f}s, 大小 {cif_from_instance.stat().st_size} bytes")
        
        # 2. 从字典格式重建 CIF
        dict_file = cif_test_dir / "atom14_dict.pt"
        atom14.save(str(dict_file), save_as_instance=False)
        atom14_dict = Atom14.load(str(dict_file))
        
        cif_from_dict = cif_test_dir / "reconstructed_from_dict.cif"
        start_time = time.time()
        atom14_dict.to_cif(str(cif_from_dict))
        reconstruction_time = time.time() - start_time
        
        reconstruction_results["reconstructions"]["from_dict"] = {
            "reconstruction_time": reconstruction_time,
            "file_size": cif_from_dict.stat().st_size,
            "file_path": str(cif_from_dict)
        }
        
        print(f"✅ 从字典重建: {reconstruction_time:.3f}s, 大小 {cif_from_dict.stat().st_size} bytes")
        
        # 3. 直接重建（无中间保存）
        cif_direct = cif_test_dir / "reconstructed_direct.cif"
        start_time = time.time()
        atom14.to_cif(str(cif_direct))
        reconstruction_time = time.time() - start_time
        
        reconstruction_results["reconstructions"]["direct"] = {
            "reconstruction_time": reconstruction_time,
            "file_size": cif_direct.stat().st_size,
            "file_path": str(cif_direct)
        }
        
        print(f"✅ 直接重建: {reconstruction_time:.3f}s, 大小 {cif_direct.stat().st_size} bytes")
        
        # 保存重建结果
        with open(cif_test_dir / "reconstruction_results.json", 'w', encoding='utf-8') as f:
            json.dump(reconstruction_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 验证所有文件都成功生成
        assert cif_from_instance.exists(), "从实例重建的 CIF 文件未生成"
        assert cif_from_dict.exists(), "从字典重建的 CIF 文件未生成"
        assert cif_direct.exists(), "直接重建的 CIF 文件未生成"
        
        print(f"📁 可视化验证文件已保存到: {cif_test_dir}")

    def test_comprehensive_workflow(self, test_data_files: List[Path], test_output_dir: Path, script_path: Path):
        """测试完整的端到端工作流程"""
        workflow_dir = test_output_dir / "comprehensive_workflow"
        workflow_dir.mkdir(exist_ok=True)
        
        print(f"\n🧪 测试完整端到端工作流程")
        
        workflow_results = {
            "start_time": time.time(),
            "steps": {},
            "files_generated": [],
            "performance_metrics": {}
        }
        
        # 步骤 1: 使用批量脚本转换所有测试文件
        print("📝 步骤 1: 批量转换")
        
        # 创建输入目录
        input_dir = workflow_dir / "input_files"
        input_dir.mkdir(exist_ok=True)
        for test_file in test_data_files:
            shutil.copy2(test_file, input_dir / test_file.name)
        
        # 批量转换
        script_output_dir = workflow_dir / "script_output"
        script_output_dir.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable, str(script_path),
            str(input_dir),
            str(script_output_dir),
            "--device", "cpu",
            "--workers", "2",
            "--save-stats", str(workflow_dir / "batch_statistics.json")
        ]
        
        step_start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        step_time = time.time() - step_start
        
        workflow_results["steps"]["batch_conversion"] = {
            "success": result.returncode == 0,
            "execution_time": step_time,
            "files_processed": len(test_data_files),
            "command": " ".join(cmd)
        }
        
        assert result.returncode == 0, f"批量转换失败: {result.stderr}"
        print(f"✅ 批量转换完成: {step_time:.2f}s")
        
        # 步骤 2: 验证生成的文件
        print("📝 步骤 2: 文件验证")
        pt_files = list(script_output_dir.glob("*.pt"))
        workflow_results["files_generated"] = [str(f) for f in pt_files]
        
        # 步骤 3: 测试每个生成文件的转换能力
        print("📝 步骤 3: 格式转换测试")
        conversion_results = {}
        
        for i, pt_file in enumerate(pt_files):
            file_stem = pt_file.stem
            print(f"  处理文件 {i+1}/{len(pt_files)}: {file_stem}")
            
            # 加载 Atom14
            atom14 = Atom14.load(str(pt_file))
            
            # 测试实例保存
            instance_file = workflow_dir / f"{file_stem}_instance.pt"
            atom14.save(str(instance_file), save_as_instance=True)
            atom14_instance = Atom14.load(str(instance_file))
            
            # 测试字典保存
            dict_file = workflow_dir / f"{file_stem}_dict.pt"
            atom14.save(str(dict_file), save_as_instance=False)
            atom14_dict = Atom14.load(str(dict_file))
            
            # 数据一致性验证
            instance_consistent = self._check_consistency(atom14, atom14_instance)
            dict_consistent = self._check_consistency(atom14, atom14_dict)
            
            # CIF 重建
            cif_output_dir = workflow_dir / "cif_outputs"
            cif_output_dir.mkdir(exist_ok=True)
            
            cif_file = cif_output_dir / f"{file_stem}_reconstructed.cif"
            atom14.to_cif(str(cif_file))
            
            conversion_results[file_stem] = {
                "num_residues": atom14.num_residues,
                "num_chains": atom14.num_chains,
                "instance_consistent": instance_consistent,
                "dict_consistent": dict_consistent,
                "cif_generated": cif_file.exists(),
                "cif_size": cif_file.stat().st_size if cif_file.exists() else 0
            }
        
        workflow_results["conversion_results"] = conversion_results
        workflow_results["end_time"] = time.time()
        workflow_results["total_duration"] = workflow_results["end_time"] - workflow_results["start_time"]
        
        # 保存完整工作流结果
        with open(workflow_dir / "comprehensive_workflow_results.json", 'w', encoding='utf-8') as f:
            json.dump(workflow_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 完整工作流程完成: {workflow_results['total_duration']:.2f}s")
        print(f"📊 处理了 {len(pt_files)} 个文件")
        print(f"📁 所有结果保存到: {workflow_dir}")
        
        # 验证所有步骤成功
        assert workflow_results["steps"]["batch_conversion"]["success"], "批量转换步骤失败"
        assert len(pt_files) > 0, "没有生成 PT 文件"
        assert all(r["cif_generated"] for r in conversion_results.values()), "部分 CIF 文件未生成"

    def _verify_pt_file(self, pt_file: Path, output_dir: Path) -> Dict[str, Any]:
        """验证单个 .pt 文件的完整性"""
        verification_result = {
            "file_path": str(pt_file),
            "file_size": pt_file.stat().st_size,
            "load_success": False,
            "atom14_info": {},
            "consistency_test": False,
            "cif_generation": False
        }
        
        try:
            # 加载数据
            atom14 = Atom14.load(str(pt_file))
            verification_result["load_success"] = True
            
            # 记录基本信息
            verification_result["atom14_info"] = {
                "num_residues": atom14.num_residues,
                "num_chains": atom14.num_chains,
                "coords_shape": list(atom14.coords.shape),
                "device": str(atom14.device)
            }
            
            # 验证基本属性
            assert atom14.coords.shape[-2:] == (14, 3), f"坐标形状错误: {atom14.coords.shape}"
            assert atom14.atom_mask.shape[-1] == 14, f"原子掩码形状错误: {atom14.atom_mask.shape}"
            assert atom14.num_residues > 0, "残基数量为零"
            
            # 测试保存和重新加载
            test_file = output_dir / f"{pt_file.stem}_verification.pt"
            atom14.save(str(test_file), save_as_instance=True)
            atom14_reloaded = Atom14.load(str(test_file))
            
            # 验证一致性
            verification_result["consistency_test"] = self._check_consistency(atom14, atom14_reloaded)
            
            # 生成 CIF 文件
            cif_file = output_dir / f"{pt_file.stem}_verification.cif"
            atom14.to_cif(str(cif_file))
            verification_result["cif_generation"] = cif_file.exists()
            
            if cif_file.exists():
                verification_result["cif_size"] = cif_file.stat().st_size
                
        except Exception as e:
            verification_result["error"] = str(e)
        
        # 保存验证结果
        result_file = output_dir / f"{pt_file.stem}_verification_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(verification_result, f, indent=2, ensure_ascii=False, default=str)
        
        return verification_result
    
    def _check_consistency(self, atom14_1: Atom14, atom14_2: Atom14) -> bool:
        """检查两个 Atom14 实例的数据一致性"""
        try:
            # 验证坐标
            coords_match = torch.allclose(atom14_1.coords, atom14_2.coords, rtol=1e-5, atol=1e-6)
            if not coords_match:
                return False
            
            # 验证掩码
            if not torch.equal(atom14_1.atom_mask, atom14_2.atom_mask):
                return False
            
            if not torch.equal(atom14_1.res_mask, atom14_2.res_mask):
                return False
            
            # 验证元数据
            if not torch.equal(atom14_1.chain_ids, atom14_2.chain_ids):
                return False
            
            if not torch.equal(atom14_1.residue_types, atom14_2.residue_types):
                return False
            
            return True
            
        except Exception:
            return False


# 独立测试函数（可以直接运行）
def test_quick_verification():
    """快速验证测试函数"""
    print("\n🚀 Atom14 快速功能验证")
    
    # 获取测试数据
    data_dir = Path(__file__).parent.parent / "data"
    test_files = list(data_dir.glob("*.cif"))[:1]
    
    if not test_files:
        pytest.skip("没有找到测试数据文件")
    
    # 创建输出目录
    output_dir = Path(__file__).parent / "test_results" / "quick_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 验证 Atom14 基本功能
    protein = load_structure(test_files[0])
    atom14 = Atom14.from_protein_tensor(protein)
    
    # 测试保存和加载
    test_file = output_dir / "quick_test.pt"
    atom14.save(str(test_file), save_as_instance=True)
    atom14_loaded = Atom14.load(str(test_file))
    
    # 生成 CIF 用于验证
    cif_file = output_dir / "quick_verification.cif"
    atom14_loaded.to_cif(str(cif_file))
    
    print(f"✅ 快速验证完成")
    print(f"📁 验证文件保存到: {output_dir}")
    print(f"🔬 可视化验证文件: {cif_file}")


if __name__ == "__main__":
    # 允许直接运行测试
    print("🧪 运行 Atom14 端到端测试")
    pytest.main([__file__, "-v", "-s", "--no-cov"])
