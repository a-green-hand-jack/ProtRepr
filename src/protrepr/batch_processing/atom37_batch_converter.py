"""
Atom37 批量转换器模块

本模块提供高性能的批量 PDB/CIF 文件到 Atom37 格式转换功能。
基于 Atom14 批处理器的成熟架构，专门适配 Atom37 的 37 个重原子槽位。

核心特性：
- 多进程并行处理，充分利用多核 CPU 资源
- 自动错误处理和统计收集
- 支持递归目录搜索和目录结构保持
- 内存友好的流式处理
- 完整的进度跟踪和详细的错误报告
- 支持 Atom37 实例和字典两种保存格式
- PyTorch 原生支持，可选 GPU 加速

使用示例：
    >>> from protrepr.batch_processing import BatchPDBToAtom37Converter
    >>> 
    >>> converter = BatchPDBToAtom37Converter(
    ...     n_workers=8,
    ...     device='cpu',
    ...     save_as_instance=True
    ... )
    >>> 
    >>> statistics = converter.convert_batch(
    ...     input_path='/path/to/pdb_files',
    ...     output_dir='/path/to/atom37_output'
    ... )
    >>> 
    >>> print(f"转换了 {statistics['success']} 个文件")
"""

import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

import torch
from protein_tensor import load_structure

from ..core.atom37 import Atom37

logger = logging.getLogger(__name__)


def convert_single_worker_atom37(
    task: Dict[str, Any]
) -> Dict[str, Any]:
    """
    单个工作进程的 PDB/CIF 到 Atom37 转换函数。
    
    Args:
        task: 包含转换任务信息的字典
            - input_file: 输入文件路径
            - output_file: 输出文件路径
            - device: 计算设备
            - save_as_instance: 保存格式选择
            
    Returns:
        Dict: 转换结果统计
    """
    input_file = Path(task['input_file'])
    output_file = Path(task['output_file'])
    device = task['device']
    save_as_instance = task['save_as_instance']
    
    start_time = time.perf_counter()
    
    try:
        # 设置 PyTorch 设备
        if device == 'cuda' and torch.cuda.is_available():
            torch_device = torch.device('cuda')
        else:
            torch_device = torch.device('cpu')
        
        # 加载蛋白质结构
        protein_tensor = load_structure(input_file)
        
        # 转换为 Atom37
        atom37 = Atom37.from_protein_tensor(protein_tensor, device=torch_device)
        
        # 创建输出目录
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存文件
        atom37.save(output_file, save_as_instance=save_as_instance)
        
        processing_time = time.perf_counter() - start_time
        
        # 收集统计信息
        return {
            'input_file': str(input_file),
            'output_file': str(output_file),
            'success': True,
            'processing_time': processing_time,
            'num_residues': atom37.num_residues,
            'num_chains': atom37.num_chains,
            'num_atoms': int(atom37.atom_mask.sum().item()),
            'file_size': output_file.stat().st_size,
            'error': None
        }
        
    except Exception as e:
        processing_time = time.perf_counter() - start_time
        logger.error(f"转换失败 {input_file}: {e}")
        
        return {
            'input_file': str(input_file),
            'output_file': str(output_file),
            'success': False,
            'processing_time': processing_time,
            'num_residues': 0,
            'num_chains': 0,
            'num_atoms': 0,
            'file_size': 0,
            'error': str(e)
        }


class BatchPDBToAtom37Converter:
    """
    批量 PDB/CIF 到 Atom37 转换器。
    
    该类提供高性能的批量转换功能，支持多进程并行处理和完整的统计收集。
    """
    
    def __init__(
        self,
        n_workers: Optional[int] = None,
        preserve_structure: bool = True,
        device: str = 'cpu',
        save_as_instance: bool = True
    ):
        """
        初始化批量转换器。
        
        Args:
            n_workers: 并行工作进程数，None 表示使用 CPU 核心数的一半
            preserve_structure: 是否保持输入目录结构
            device: 计算设备 ('cpu' 或 'cuda')
            save_as_instance: 是否保存为 Atom37 实例格式（否则保存为字典格式）
        """
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() // 2)
        
        self.n_workers = n_workers
        self.preserve_structure = preserve_structure
        self.device = device
        self.save_as_instance = save_as_instance
        
        logger.info(f"初始化 Atom37 批量转换器:")
        logger.info(f"  并行进程数: {self.n_workers}")
        logger.info(f"  保持目录结构: {self.preserve_structure}")
        logger.info(f"  计算设备: {self.device}")
        logger.info(f"  保存格式: {'Atom37实例' if self.save_as_instance else '字典格式'}")
    
    def find_structure_files(
        self,
        input_path: Union[str, Path],
        recursive: bool = True
    ) -> List[Path]:
        """
        查找输入路径中的所有结构文件。
        
        Args:
            input_path: 输入文件或目录路径
            recursive: 是否递归搜索子目录
            
        Returns:
            List[Path]: 找到的结构文件列表
        """
        input_path = Path(input_path)
        
        if input_path.is_file():
            return [input_path]
        
        if not input_path.is_dir():
            raise FileNotFoundError(f"输入路径不存在: {input_path}")
        
        # 支持的文件扩展名
        extensions = {'.pdb', '.cif', '.ent', '.pdb.gz', '.cif.gz'}
        
        structure_files = []
        
        if recursive:
            for ext in extensions:
                pattern = f"**/*{ext}"
                structure_files.extend(input_path.glob(pattern))
        else:
            for ext in extensions:
                pattern = f"*{ext}"
                structure_files.extend(input_path.glob(pattern))
        
        return sorted(structure_files)
    
    def prepare_tasks(
        self,
        input_files: List[Path],
        input_path: Path,
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        准备转换任务列表。
        
        Args:
            input_files: 输入文件列表
            input_path: 输入根路径
            output_dir: 输出目录
            
        Returns:
            List[Dict]: 任务列表
        """
        tasks = []
        
        for input_file in input_files:
            # 确定输出文件路径
            if self.preserve_structure and input_path.is_dir():
                # 保持目录结构
                relative_path = input_file.relative_to(input_path)
                output_file = output_dir / relative_path.with_suffix('.pt')
            else:
                # 平铺到输出目录
                output_file = output_dir / f"{input_file.stem}.pt"
            
            task = {
                'input_file': input_file,
                'output_file': output_file,
                'device': self.device,
                'save_as_instance': self.save_as_instance
            }
            tasks.append(task)
        
        return tasks
    
    def convert_batch(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        recursive: bool = True
    ) -> Dict[str, Any]:
        """
        执行批量转换。
        
        Args:
            input_path: 输入文件或目录路径
            output_dir: 输出目录路径
            recursive: 是否递归搜索子目录
            
        Returns:
            Dict: 转换统计信息
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        
        logger.info(f"开始 Atom37 批量转换:")
        logger.info(f"  输入路径: {input_path}")
        logger.info(f"  输出目录: {output_dir}")
        
        # 查找所有结构文件
        structure_files = self.find_structure_files(input_path, recursive)
        
        if not structure_files:
            logger.warning("没有找到任何结构文件")
            return {
                'total': 0,
                'success': 0,
                'failed': 0,
                'results': [],
                'failed_files': [],
                'total_time': 0.0
            }
        
        logger.info(f"找到 {len(structure_files)} 个结构文件")
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备任务
        tasks = self.prepare_tasks(structure_files, input_path, output_dir)
        
        # 执行并行转换
        logger.info(f"使用 {self.n_workers} 个进程开始并行转换")
        start_time = time.perf_counter()
        
        if self.n_workers == 1:
            # 单进程模式
            results = [convert_single_worker_atom37(task) for task in tasks]
        else:
            # 多进程模式
            with mp.Pool(self.n_workers) as pool:
                results = pool.map(convert_single_worker_atom37, tasks)
        
        total_time = time.perf_counter() - start_time
        
        # 统计结果
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        statistics = {
            'total': len(results),
            'success': len(successful_results),
            'failed': len(failed_results),
            'results': results,
            'failed_files': [r['input_file'] for r in failed_results],
            'total_time': total_time,
            'avg_time_per_file': total_time / len(results) if results else 0.0,
            'total_residues': sum(r['num_residues'] for r in successful_results),
            'total_atoms': sum(r['num_atoms'] for r in successful_results),
            'total_output_size': sum(r['file_size'] for r in successful_results),
            'device': self.device,
            'save_format': 'Atom37实例' if self.save_as_instance else '字典格式',
            'preserve_structure': self.preserve_structure
        }
        
        logger.info(f"批量转换完成:")
        logger.info(f"  总文件数: {statistics['total']}")
        logger.info(f"  成功转换: {statistics['success']}")
        logger.info(f"  转换失败: {statistics['failed']}")
        logger.info(f"  总用时: {total_time:.2f} 秒")
        logger.info(f"  平均用时: {statistics['avg_time_per_file']:.3f} 秒/文件")
        
        if statistics['success'] > 0:
            logger.info(f"  总残基数: {statistics['total_residues']:,}")
            logger.info(f"  总原子数: {statistics['total_atoms']:,}")
            logger.info(f"  输出大小: {statistics['total_output_size'] / 1024 / 1024:.1f} MB")
        
        if statistics['failed'] > 0:
            logger.warning(f"失败的文件: {statistics['failed_files'][:5]}...")
        
        return statistics
    
    def convert_single(
        self,
        input_file: Union[str, Path],
        output_file: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        转换单个文件。
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            
        Returns:
            Dict: 转换结果
        """
        task = {
            'input_file': Path(input_file),
            'output_file': Path(output_file),
            'device': self.device,
            'save_as_instance': self.save_as_instance
        }
        
        return convert_single_worker_atom37(task)


# 为了与 Atom14 模块保持一致性，复用统计保存函数
def save_statistics(statistics: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    保存批量转换统计信息到 JSON 文件。
    
    Args:
        statistics: 统计信息字典
        output_path: 输出文件路径
    """
    # 直接导入避免循环导入
    from .atom14_batch_converter import save_statistics as _save_statistics
    _save_statistics(statistics, Path(output_path)) 