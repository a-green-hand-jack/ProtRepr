#!/usr/bin/env python3
"""
Atom14 性能基准测试脚本

测试 ProtRepr 中 Atom14 转换的性能，比较优化前后的速度差异。
"""

import sys
from pathlib import Path
import time
import statistics
from typing import List, Tuple, Dict, Any

# 将项目根目录下的 src 目录添加到 Python 解释器的搜索路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import logging
import numpy as np

from protrepr.representations.atom14_converter import (
    protein_tensor_to_atom14, 
    CHAIN_GAP
)

# 设置日志
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkProteinTensor:
    """用于基准测试的 ProteinTensor 模拟类"""
    
    def __init__(self, coordinates, atom_types, residue_types, chain_ids, residue_numbers):
        self.coordinates = coordinates
        self.atom_types = atom_types
        self.residue_types = residue_types
        self.chain_ids = chain_ids
        self.residue_numbers = residue_numbers
        self.n_atoms = len(coordinates)
        self.n_residues = len(set((c.item(), r.item()) for c, r in zip(chain_ids, residue_numbers)))
    
    def to_torch(self):
        """返回 torch 格式的数据"""
        return {
            "coordinates": self.coordinates,
            "atom_types": self.atom_types,
            "residue_types": self.residue_types,
            "chain_ids": self.chain_ids,
            "residue_numbers": self.residue_numbers
        }


def create_benchmark_data(num_chains: int, residues_per_chain: int, atoms_per_residue: int = 4) -> BenchmarkProteinTensor:
    """
    创建基准测试数据。
    
    Args:
        num_chains: 链的数量
        residues_per_chain: 每条链的残基数量
        atoms_per_residue: 每个残基的原子数量
        
    Returns:
        BenchmarkProteinTensor: 测试用的蛋白质数据
    """
    device = torch.device("cpu")
    
    total_residues = num_chains * residues_per_chain
    total_atoms = total_residues * atoms_per_residue
    
    # 生成坐标（随机位置）
    coordinates = torch.randn(total_atoms, 3, device=device) * 10.0
    
    # 原子类型 (0=N, 1=CA, 2=C, 3=O)
    atom_types = torch.tensor([0, 1, 2, 3] * total_residues, device=device, dtype=torch.long)
    
    # 残基类型（随机选择标准氨基酸 0-19）
    residue_types_per_residue = torch.randint(0, 20, (total_residues,), device=device)
    residue_types = torch.repeat_interleave(residue_types_per_residue, atoms_per_residue)
    
    # 链ID
    chain_ids = []
    for chain_id in range(num_chains):
        chain_atoms = residues_per_chain * atoms_per_residue
        chain_ids.append(torch.full((chain_atoms,), chain_id, device=device, dtype=torch.long))
    chain_ids = torch.cat(chain_ids)
    
    # 残基编号
    residue_numbers = []
    for chain_id in range(num_chains):
        for res_id in range(residues_per_chain):
            residue_numbers.extend([res_id + 1] * atoms_per_residue)
    residue_numbers = torch.tensor(residue_numbers, device=device, dtype=torch.long)
    
    return BenchmarkProteinTensor(
        coordinates=coordinates,
        atom_types=atom_types,
        residue_types=residue_types,
        chain_ids=chain_ids,
        residue_numbers=residue_numbers
    )


def time_function(func, *args, **kwargs) -> Tuple[Any, float]:
    """
    测量函数执行时间。
    
    Args:
        func: 要测试的函数
        *args: 函数参数
        **kwargs: 函数关键字参数
        
    Returns:
        Tuple[Any, float]: (函数返回值, 执行时间秒)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return result, execution_time


def benchmark_conversion(protein_tensor: BenchmarkProteinTensor, num_runs: int = 10) -> Dict[str, float]:
    """
    基准测试 ProteinTensor 到 Atom14 的转换性能。
    
    Args:
        protein_tensor: 测试用的蛋白质数据
        num_runs: 运行次数
        
    Returns:
        Dict[str, float]: 性能统计数据
    """
    times = []
    
    # 预热运行
    for _ in range(2):
        _, _ = time_function(protein_tensor_to_atom14, protein_tensor, device=torch.device("cpu"))
    
    # 正式基准测试
    for run in range(num_runs):
        result, exec_time = time_function(protein_tensor_to_atom14, protein_tensor, device=torch.device("cpu"))
        times.append(exec_time)
        
        # 验证结果有效性
        coords, atom_mask, res_mask, chain_ids, residue_types, residue_indices, chain_residue_indices, residue_names, atom_names = result
        assert coords.shape[0] > 0, "转换结果无效"
    
    return {
        'mean_time': statistics.mean(times),
        'median_time': statistics.median(times),
        'std_time': statistics.stdev(times) if len(times) > 1 else 0.0,
        'min_time': min(times),
        'max_time': max(times),
        'total_time': sum(times),
        'runs': num_runs
    }


def run_benchmark_suite():
    """运行完整的基准测试套件"""
    print("🚀 开始 Atom14 性能基准测试")
    print("=" * 60)
    
    # 测试场景
    test_scenarios = [
        {"name": "小型蛋白质", "chains": 1, "residues_per_chain": 50, "atoms_per_residue": 4},
        {"name": "中型蛋白质", "chains": 2, "residues_per_chain": 150, "atoms_per_residue": 4},
        {"name": "大型蛋白质", "chains": 4, "residues_per_chain": 300, "atoms_per_residue": 4},
        {"name": "复杂多链", "chains": 8, "residues_per_chain": 100, "atoms_per_residue": 4},
        {"name": "超大蛋白质", "chains": 2, "residues_per_chain": 1000, "atoms_per_residue": 4},
    ]
    
    results = {}
    
    for scenario in test_scenarios:
        print(f"\n📊 测试场景: {scenario['name']}")
        print(f"   - 链数: {scenario['chains']}")
        print(f"   - 每链残基数: {scenario['residues_per_chain']}")
        print(f"   - 每残基原子数: {scenario['atoms_per_residue']}")
        
        # 创建测试数据
        protein_data = create_benchmark_data(
            num_chains=scenario['chains'],
            residues_per_chain=scenario['residues_per_chain'],
            atoms_per_residue=scenario['atoms_per_residue']
        )
        
        total_atoms = protein_data.n_atoms
        total_residues = protein_data.n_residues
        
        print(f"   - 总原子数: {total_atoms:,}")
        print(f"   - 总残基数: {total_residues:,}")
        
        # 运行基准测试
        print("   - 运行基准测试...")
        
        benchmark_stats = benchmark_conversion(protein_data, num_runs=10)
        results[scenario['name']] = {
            **scenario,
            'total_atoms': total_atoms,
            'total_residues': total_residues,
            **benchmark_stats
        }
        
        # 输出结果
        print(f"   ✅ 平均时间: {benchmark_stats['mean_time']:.4f}s")
        print(f"   📊 中位时间: {benchmark_stats['median_time']:.4f}s")
        print(f"   📈 标准差: {benchmark_stats['std_time']:.4f}s")
        print(f"   ⚡ 最快时间: {benchmark_stats['min_time']:.4f}s")
        print(f"   🐌 最慢时间: {benchmark_stats['max_time']:.4f}s")
        
        # 计算性能指标
        atoms_per_second = total_atoms / benchmark_stats['mean_time']
        residues_per_second = total_residues / benchmark_stats['mean_time']
        
        print(f"   🔥 处理速度: {atoms_per_second:,.0f} 原子/秒")
        print(f"   🔥 处理速度: {residues_per_second:,.0f} 残基/秒")
    
    # 生成总结报告
    print("\n" + "=" * 60)
    print("📈 性能基准测试总结")
    print("=" * 60)
    
    print(f"{'场景名称':<12} {'原子数':<10} {'残基数':<8} {'平均时间':<10} {'原子/秒':<12} {'残基/秒':<10}")
    print("-" * 70)
    
    for name, stats in results.items():
        atoms_per_sec = stats['total_atoms'] / stats['mean_time']
        residues_per_sec = stats['total_residues'] / stats['mean_time']
        
        print(f"{name:<12} {stats['total_atoms']:<10,} {stats['total_residues']:<8,} "
              f"{stats['mean_time']:<10.4f} {atoms_per_sec:<12,.0f} {residues_per_sec:<10,.0f}")
    
    # 保存结果到文件
    import json
    output_file = Path(__file__).parent.parent / "benchmark_results_original.json"
    with open(output_file, 'w') as f:
        # 转换numpy类型以便JSON序列化
        json_results = {}
        for name, stats in results.items():
            json_results[name] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                for k, v in stats.items()}
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 基准测试结果已保存到: {output_file}")
    
    return results


def main():
    """主函数"""
    print("⚡ Atom14 转换性能基准测试")
    print(f"🔧 PyTorch 版本: {torch.__version__}")
    print(f"💻 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"🧮 链间间隔: {CHAIN_GAP}")
    
    try:
        results = run_benchmark_suite()
        
        # 输出优化建议
        print("\n🎯 性能优化建议:")
        print("   1. 减少 Python 循环，使用 PyTorch 张量操作")
        print("   2. 优化残基分组和原子映射逻辑")
        print("   3. 批量处理链间间隔计算")
        print("   4. 减少字典查找和列表操作")
        
        print("\n✅ 基准测试完成！请保存此结果作为优化前的基线。")
        return 0
        
    except Exception as e:
        print(f"💥 基准测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 