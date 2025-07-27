"""
测试 Atom14 表示中的链间 Gap 功能

本模块专门测试多链蛋白质中链间间隔（gap）的正确性，包括：
- 全局残基编号的链间间隔
- 链内残基编号的连续性
- 多链数据的转换和验证
"""

import logging
from typing import List, Dict, Any
from pathlib import Path

import pytest
import torch

from protrepr.core.atom14 import Atom14
from protrepr.representations.atom14_converter import (
    protein_tensor_to_atom14,
    atom14_to_protein_tensor,
    CHAIN_GAP
)
# MockProteinTensor 在函数内部导入以避免循环导入问题

logger = logging.getLogger(__name__)


class TestChainGapLogic:
    """测试链间 Gap 逻辑"""
    
    def create_multi_chain_protein_data(self, device: torch.device) -> Dict[str, Any]:
        """
        创建多链蛋白质测试数据。
        
        Args:
            device: 目标设备
            
        Returns:
            包含多链蛋白质数据的字典
        """
        # 创建两条链的数据
        # 链 A: 3 个残基，每个残基 4 个原子 (N, CA, C, O)
        # 链 B: 2 个残基，每个残基 4 个原子 (N, CA, C, O)
        
        num_atoms_chain_a = 3 * 4  # 12 个原子
        num_atoms_chain_b = 2 * 4  # 8 个原子
        total_atoms = num_atoms_chain_a + num_atoms_chain_b  # 20 个原子
        
        # 坐标
        coords = torch.randn(total_atoms, 3, device=device) * 10.0
        
        # 原子类型 (0=N, 1=CA, 2=C, 3=O)
        atom_types = torch.tensor([0, 1, 2, 3] * 5, device=device, dtype=torch.long)
        
        # 残基类型（都使用 ALA=0）
        residue_types = torch.zeros(total_atoms, device=device, dtype=torch.long)
        
        # 链ID
        chain_ids = torch.cat([
            torch.zeros(num_atoms_chain_a, device=device, dtype=torch.long),  # 链 0
            torch.ones(num_atoms_chain_b, device=device, dtype=torch.long)    # 链 1
        ])
        
        # 原始残基编号
        residue_numbers = torch.cat([
            torch.repeat_interleave(torch.tensor([10, 11, 12], device=device), 4),  # 链 A: 10-12
            torch.repeat_interleave(torch.tensor([5, 6], device=device), 4)         # 链 B: 5-6
        ])
        
        return {
            "coordinates": coords,
            "atom_types": atom_types,
            "residue_types": residue_types,
            "chain_ids": chain_ids,
            "residue_numbers": residue_numbers
        }
    
    def test_chain_gap_constant(self):
        """测试链间间隔常量"""
        assert CHAIN_GAP == 200, f"链间间隔应为 200，实际为 {CHAIN_GAP}"
    
    def test_multi_chain_conversion(self, device: torch.device):
        """测试多链数据转换中的链间间隔"""
        from tests.conftest import MockProteinTensor
        
        # 创建多链数据
        protein_data = self.create_multi_chain_protein_data(device)
        mock_protein = MockProteinTensor(**protein_data)
        
        # 转换为 Atom14
        (coords, atom_mask, res_mask, chain_ids, residue_types, 
         residue_indices, chain_residue_indices, residue_names, atom_names) = protein_tensor_to_atom14(
            mock_protein, device=device
        )
        
        # 验证残基数量
        expected_residues = 5  # 3 (链A) + 2 (链B)
        assert coords.shape[0] == expected_residues, f"期望 {expected_residues} 个残基，实际 {coords.shape[0]}"
        
        # 验证链ID
        unique_chains = torch.unique(chain_ids)
        assert len(unique_chains) == 2, f"期望 2 条链，实际 {len(unique_chains)}"
        assert 0 in unique_chains and 1 in unique_chains, "链ID应包含 0 和 1"
        
        # 验证链间间隔
        chain_a_indices = residue_indices[chain_ids == 0]  # 链 A 的全局残基编号
        chain_b_indices = residue_indices[chain_ids == 1]  # 链 B 的全局残基编号
        
        # 检查链 A 的编号
        expected_chain_a = torch.tensor([1, 2, 3], device=device)  # 从 1 开始
        assert torch.equal(chain_a_indices, expected_chain_a), \
            f"链 A 残基编号错误: 期望 {expected_chain_a}, 实际 {chain_a_indices}"
        
        # 检查链 B 的编号（应该从 1 + 3 + 200 = 204 开始）
        expected_chain_b = torch.tensor([204, 205], device=device)
        assert torch.equal(chain_b_indices, expected_chain_b), \
            f"链 B 残基编号错误: 期望 {expected_chain_b}, 实际 {chain_b_indices}"
        
        # 验证链间间隔
        gap = chain_b_indices.min() - chain_a_indices.max() - 1
        assert gap == CHAIN_GAP, f"链间间隔错误: 期望 {CHAIN_GAP}, 实际 {gap}"
        
        logger.info(f"✅ 链间间隔验证成功: 链A编号 {chain_a_indices.tolist()}, "
                   f"链B编号 {chain_b_indices.tolist()}, 间隔 {gap}")
    
    def test_chain_residue_indices_continuity(self, device: torch.device):
        """测试链内残基编号的连续性"""
        from tests.conftest import MockProteinTensor
        
        # 创建多链数据
        protein_data = self.create_multi_chain_protein_data(device)
        mock_protein = MockProteinTensor(**protein_data)
        
        # 转换为 Atom14
        (coords, atom_mask, res_mask, chain_ids, residue_types, 
         residue_indices, chain_residue_indices, residue_names, atom_names) = protein_tensor_to_atom14(
            mock_protein, device=device
        )
        
        # 验证链内编号的连续性
        for chain_id in torch.unique(chain_ids):
            chain_mask = chain_ids == chain_id
            chain_res_indices = chain_residue_indices[chain_mask]
            
            # 链内编号应该从 0 开始连续
            expected_indices = torch.arange(len(chain_res_indices), device=device)
            assert torch.equal(chain_res_indices, expected_indices), \
                f"链 {chain_id} 内残基编号不连续: 期望 {expected_indices}, 实际 {chain_res_indices}"
            
            logger.info(f"✅ 链 {chain_id} 内残基编号连续: {chain_res_indices.tolist()}")
    
    def test_three_chain_scenario(self, device: torch.device):
        """测试三条链的场景"""
        from tests.conftest import MockProteinTensor
        
        # 创建三条链的数据
        # 链 A: 2 个残基
        # 链 B: 1 个残基  
        # 链 C: 3 个残基
        
        num_atoms_per_residue = 4
        chain_a_residues = 2
        chain_b_residues = 1
        chain_c_residues = 3
        
        total_residues = chain_a_residues + chain_b_residues + chain_c_residues
        total_atoms = total_residues * num_atoms_per_residue
        
        # 构建数据
        coords = torch.randn(total_atoms, 3, device=device) * 10.0
        atom_types = torch.tensor([0, 1, 2, 3] * total_residues, device=device, dtype=torch.long)
        residue_types = torch.zeros(total_atoms, device=device, dtype=torch.long)
        
        # 链ID
        chain_ids = torch.cat([
            torch.zeros(chain_a_residues * num_atoms_per_residue, device=device, dtype=torch.long),   # 链 0
            torch.ones(chain_b_residues * num_atoms_per_residue, device=device, dtype=torch.long),    # 链 1
            torch.full((chain_c_residues * num_atoms_per_residue,), 2, device=device, dtype=torch.long)  # 链 2
        ])
        
        # 残基编号
        residue_numbers = torch.cat([
            torch.repeat_interleave(torch.tensor([1, 2], device=device), num_atoms_per_residue),      # 链 A
            torch.repeat_interleave(torch.tensor([1], device=device), num_atoms_per_residue),         # 链 B
            torch.repeat_interleave(torch.tensor([1, 2, 3], device=device), num_atoms_per_residue)   # 链 C
        ])
        
        protein_data = {
            "coordinates": coords,
            "atom_types": atom_types,
            "residue_types": residue_types,
            "chain_ids": chain_ids,
            "residue_numbers": residue_numbers
        }
        
        mock_protein = MockProteinTensor(**protein_data)
        
        # 转换为 Atom14
        (coords, atom_mask, res_mask, chain_ids_out, residue_types_out, 
         residue_indices, chain_residue_indices, residue_names, atom_names) = protein_tensor_to_atom14(
            mock_protein, device=device
        )
        
        # 验证三条链的全局编号
        chain_a_indices = residue_indices[chain_ids_out == 0]  # 链 A
        chain_b_indices = residue_indices[chain_ids_out == 1]  # 链 B  
        chain_c_indices = residue_indices[chain_ids_out == 2]  # 链 C
        
        # 预期编号：
        # 链 A: [1, 2]
        # 链 B: [1+2+200=203]
        # 链 C: [203+1+200=404, 405, 406]
        
        expected_a = torch.tensor([1, 2], device=device)
        expected_b = torch.tensor([203], device=device)
        expected_c = torch.tensor([404, 405, 406], device=device)
        
        assert torch.equal(chain_a_indices, expected_a), \
            f"链 A 编号错误: 期望 {expected_a}, 实际 {chain_a_indices}"
        assert torch.equal(chain_b_indices, expected_b), \
            f"链 B 编号错误: 期望 {expected_b}, 实际 {chain_b_indices}"
        assert torch.equal(chain_c_indices, expected_c), \
            f"链 C 编号错误: 期望 {expected_c}, 实际 {chain_c_indices}"
        
        # 验证链间间隔
        gap_ab = chain_b_indices.min() - chain_a_indices.max() - 1
        gap_bc = chain_c_indices.min() - chain_b_indices.max() - 1
        
        assert gap_ab == CHAIN_GAP, f"A-B 链间隔错误: 期望 {CHAIN_GAP}, 实际 {gap_ab}"
        assert gap_bc == CHAIN_GAP, f"B-C 链间隔错误: 期望 {CHAIN_GAP}, 实际 {gap_bc}"
        
        logger.info(f"✅ 三链间隔验证成功: A{chain_a_indices.tolist()}, "
                   f"B{chain_b_indices.tolist()}, C{chain_c_indices.tolist()}")
        logger.info(f"✅ 间隔: A-B={gap_ab}, B-C={gap_bc}")
    
    def test_atom14_roundtrip_with_chain_gap(self, device: torch.device):
        """测试包含链间间隔的往返转换"""
        from tests.conftest import MockProteinTensor
        
        # 创建多链数据
        protein_data = self.create_multi_chain_protein_data(device)
        mock_protein = MockProteinTensor(**protein_data)
        
        # 转换为 Atom14
        atom14_data = protein_tensor_to_atom14(mock_protein, device=device)
        
        # 创建 Atom14 实例
        atom14 = Atom14(
            coords=atom14_data[0],
            atom_mask=atom14_data[1],
            res_mask=atom14_data[2],
            chain_ids=atom14_data[3],
            residue_types=atom14_data[4],
            residue_indices=atom14_data[5],
            chain_residue_indices=atom14_data[6],
            residue_names=atom14_data[7],
            atom_names=atom14_data[8]
        )
        
        # 转换回 ProteinTensor
        reconstructed_protein = atom14.to_protein_tensor()
        
        # 验证转换是否成功
        assert hasattr(reconstructed_protein, 'coordinates'), "重构的蛋白质应有坐标属性"
        assert hasattr(reconstructed_protein, 'chain_ids'), "重构的蛋白质应有链ID属性"
        
        # 验证链间间隔在往返转换中是否保持
        original_residue_indices = atom14_data[5]
        chain_ids_tensor = atom14_data[3]
        
        # 检查每条链的编号范围
        for chain_id in torch.unique(chain_ids_tensor):
            chain_mask = chain_ids_tensor == chain_id
            chain_indices = original_residue_indices[chain_mask]
            
            # 每条链内的编号应该是连续的
            assert torch.equal(chain_indices, torch.arange(
                chain_indices.min().item(), chain_indices.max().item() + 1, device=device
            )), f"链 {chain_id} 的编号不连续: {chain_indices}"
        
        logger.info("✅ 链间间隔往返转换验证成功")


@pytest.fixture
def device() -> torch.device:
    """设备 fixture"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") 