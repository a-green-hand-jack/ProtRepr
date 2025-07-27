"""
Atom14 批量转换核心实现

提供高效的批量 PDB/CIF 到 Atom14 转换功能，支持并行处理、
多种输出格式和完整的错误处理。
"""

import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import torch
import numpy as np
from protein_tensor import load_structure

# 本地导入
from ..core.atom14 import Atom14
from ..representations.atom14_converter import save_atom14_to_cif

logger = logging.getLogger(__name__)


class BatchPDBToAtom14Converter:
    """
    批量 PDB 到 Atom14 转换器。
    
    支持：
    - 并行处理多个文件
    - 递归目录扫描
    - 保持目录结构
    - 多种输出格式 (npz, pt)
    - 错误处理和统计
    """
    
    def __init__(
        self,
        n_workers: Optional[int] = None,
        preserve_structure: bool = True,
        device: str = "cpu",
        output_format: str = "npz"
    ) -> None:
        """
        初始化批量转换器。
        
        Args:
            n_workers: 并行工作进程数，默认为 CPU 核心数的一半
            preserve_structure: 是否保持目录结构
            device: 计算设备 ("cpu" 或 "cuda")
            output_format: 输出格式 ("npz" 或 "pt")
        """
        self.n_workers = n_workers or max(1, mp.cpu_count() // 2)
        self.preserve_structure = preserve_structure
        self.device = torch.device(device)
        self.output_format = output_format.lower()
        
        if self.output_format not in ["npz", "pt"]:
            raise ValueError(f"不支持的输出格式: {output_format}")
        
        logger.info(f"批量转换器初始化完成:")
        logger.info(f"  - 工作进程数: {self.n_workers}")
        logger.info(f"  - 保持目录结构: {preserve_structure}")
        logger.info(f"  - 设备: {self.device}")
        logger.info(f"  - 输出格式: {self.output_format}")
    
    def find_structure_files(
        self,
        input_path: Path,
        recursive: bool = True
    ) -> List[Path]:
        """
        查找所有结构文件。
        
        Args:
            input_path: 输入路径
            recursive: 是否递归搜索
            
        Returns:
            结构文件路径列表
        """
        if input_path.is_file():
            return [input_path]
        
        if not input_path.is_dir():
            raise ValueError(f"输入路径不存在: {input_path}")
        
        # 支持的文件扩展名
        extensions = {'.pdb', '.ent', '.cif', '.mmcif'}
        
        files = []
        if recursive:
            for ext in extensions:
                files.extend(input_path.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                files.extend(input_path.glob(f"*{ext}"))
        
        files = sorted(files)
        logger.info(f"在 {input_path} 中找到 {len(files)} 个结构文件")
        return files
    
    def convert_single_file(
        self,
        input_file: Path,
        output_file: Path
    ) -> Dict[str, Any]:
        """
        转换单个文件。
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            
        Returns:
            转换结果统计
        """
        result = {
            'input_file': str(input_file),
            'output_file': str(output_file),
            'success': False,
            'error': None,
            'processing_time': 0.0,
            'num_residues': 0,
            'num_atoms': 0,
            'num_chains': 0
        }
        
        start_time = time.perf_counter()
        
        try:
            logger.debug(f"开始转换: {input_file}")
            
            # 1. 加载蛋白质结构
            protein_tensor = load_structure(input_file)
            
            # 2. 转换为 Atom14
            atom14 = Atom14.from_protein_tensor(protein_tensor)
            
            # 3. 保存结果
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if self.output_format == "npz":
                self._save_atom14_npz(atom14, output_file)
            else:  # pt
                self._save_atom14_pt(atom14, output_file)
            
            # 4. 收集统计信息
            result.update({
                'success': True,
                'num_residues': atom14.num_residues,
                'num_atoms': atom14.atom_mask.sum().item(),
                'num_chains': atom14.num_chains,
                'processing_time': time.perf_counter() - start_time
            })
            
            logger.debug(f"转换成功: {input_file} -> {output_file}")
            
        except Exception as e:
            result.update({
                'success': False,
                'error': str(e),
                'processing_time': time.perf_counter() - start_time
            })
            logger.error(f"转换失败 {input_file}: {e}")
        
        return result
    
    def _save_atom14_npz(self, atom14: Atom14, output_file: Path) -> None:
        """保存 Atom14 数据为 NPZ 格式。"""
        # 保存主要数据
        np.savez_compressed(
            output_file,
            coords=atom14.coords.cpu().numpy(),
            atom_mask=atom14.atom_mask.cpu().numpy(),
            res_mask=atom14.res_mask.cpu().numpy(),
            chain_ids=atom14.chain_ids.cpu().numpy(),
            residue_types=atom14.residue_types.cpu().numpy(),
            residue_indices=atom14.residue_indices.cpu().numpy(),
            chain_residue_indices=atom14.chain_residue_indices.cpu().numpy(),
            residue_names=atom14.residue_names.cpu().numpy(),
            atom_names=atom14.atom_names.cpu().numpy(),
            # 简单的元数据
            num_residues=np.array(atom14.num_residues),
            num_chains=np.array(atom14.num_chains)
        )
    
    def _save_atom14_pt(self, atom14: Atom14, output_file: Path) -> None:
        """保存 Atom14 数据为 PyTorch 格式。"""
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
                'num_chains': atom14.num_chains,
                'device': str(atom14.device),
                'batch_shape': list(atom14.batch_shape)
            }
        }
        
        torch.save(data, output_file)
    
    def convert_batch(
        self,
        input_path: Path,
        output_dir: Path,
        recursive: bool = True
    ) -> Dict[str, Any]:
        """
        批量转换文件。
        
        Args:
            input_path: 输入文件或目录路径
            output_dir: 输出目录路径
            recursive: 是否递归搜索
            
        Returns:
            转换统计信息
        """
        # 查找所有结构文件
        structure_files = self.find_structure_files(input_path, recursive)
        
        if not structure_files:
            logger.warning("没有找到结构文件")
            return {'total': 0, 'success': 0, 'failed': 0, 'failed_files': [], 'results': []}
        
        # 准备转换任务
        tasks = []
        for file_path in structure_files:
            # 确定输出路径
            if self.preserve_structure and input_path.is_dir():
                # 保持相对目录结构
                rel_path = file_path.relative_to(input_path)
            else:
                # 扁平输出结构
                rel_path = file_path.name
            
            # 更改扩展名
            if self.output_format == "npz":
                output_name = Path(rel_path).with_suffix('.npz')
            else:  # pt
                output_name = Path(rel_path).with_suffix('.pt')
            
            output_path = output_dir / output_name
            tasks.append((file_path, output_path))
        
        logger.info(f"开始批量转换 {len(tasks)} 个文件，使用 {self.n_workers} 个工作进程")
        
        # 执行转换
        results = []
        failed_files = []
        
        if self.n_workers == 1:
            # 串行处理
            for input_file, output_file in tasks:
                result = self.convert_single_file(input_file, output_file)
                results.append(result)
                if not result['success']:
                    failed_files.append(result['input_file'])
        else:
            # 并行处理
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                # 提交所有任务
                future_to_task = {
                    executor.submit(convert_single_worker, str(input_file), str(output_file), 
                                  self.output_format, str(self.device)): (input_file, output_file)
                    for input_file, output_file in tasks
                }
                
                # 收集结果
                for future in as_completed(future_to_task):
                    input_file, output_file = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if not result['success']:
                            failed_files.append(result['input_file'])
                    except Exception as e:
                        logger.error(f"工作进程失败 {input_file}: {e}")
                        result = {
                            'input_file': str(input_file),
                            'output_file': str(output_file),
                            'success': False,
                            'error': str(e),
                            'processing_time': 0.0,
                            'num_residues': 0,
                            'num_atoms': 0,
                            'num_chains': 0
                        }
                        results.append(result)
                        failed_files.append(str(input_file))
        
        # 统计结果
        success_count = sum(1 for r in results if r['success'])
        failed_count = len(results) - success_count
        
        statistics = {
            'total': len(results),
            'success': success_count,
            'failed': failed_count,
            'failed_files': failed_files,
            'results': results
        }
        
        logger.info(f"批量转换完成: {success_count} 成功, {failed_count} 失败")
        return statistics


def convert_single_worker(
    input_file: str,
    output_file: str,
    output_format: str,
    device: str
) -> Dict[str, Any]:
    """
    工作进程函数，用于并行处理。
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        output_format: 输出格式
        device: 设备
        
    Returns:
        转换结果
    """
    try:
        # 创建新的转换器实例
        converter = BatchPDBToAtom14Converter(
            n_workers=1,
            output_format=output_format,
            device=device
        )
        return converter.convert_single_file(Path(input_file), Path(output_file))
    except Exception as e:
        logger.error(f"工作进程错误 {input_file}: {e}")
        return {
            'input_file': input_file,
            'output_file': output_file,
            'success': False,
            'error': str(e),
            'processing_time': 0.0,
            'num_residues': 0,
            'num_atoms': 0,
            'num_chains': 0
        }


def save_statistics(statistics: Dict[str, Any], output_file: Path) -> None:
    """保存转换统计信息到 JSON 文件。"""
    # 清理结果数据以便 JSON 序列化
    clean_stats = statistics.copy()
    clean_results = []
    
    for result in statistics['results']:
        clean_result = result.copy()
        # 确保所有数值都是 JSON 可序列化的
        for key, value in clean_result.items():
            if isinstance(value, (np.integer, np.floating)):
                clean_result[key] = float(value)
        clean_results.append(clean_result)
    
    clean_stats['results'] = clean_results
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"统计信息已保存到: {output_file}") 