"""
Atom37 到 CIF/PDB 批量转换核心实现

提供高效的批量 Atom37 到 CIF/PDB 转换功能，支持并行处理、
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

# 本地导入
from ..core.atom37 import Atom37

logger = logging.getLogger(__name__)


class BatchAtom37ToCIFConverter:
    """
    批量 Atom37 到 CIF/PDB 转换器。
    
    支持：
    - 并行处理多个文件
    - 递归目录扫描
    - 保持目录结构
    - CIF 和 PDB 格式输出
    - 错误处理和统计
    """
    
    def __init__(
        self,
        n_workers: Optional[int] = None,
        preserve_structure: bool = True,
        output_format: str = "cif"
    ) -> None:
        """
        初始化批量转换器。
        
        Args:
            n_workers: 并行工作进程数，默认为 CPU 核心数的一半
            preserve_structure: 是否保持目录结构
            output_format: 输出格式 ("cif" 或 "pdb")
        """
        self.n_workers = n_workers or max(1, mp.cpu_count() // 2)
        self.preserve_structure = preserve_structure
        self.output_format = output_format.lower()
        
        if self.output_format not in ["cif", "pdb"]:
            raise ValueError(f"不支持的输出格式: {output_format}，支持 'cif' 或 'pdb'")
        
        logger.info(f"Atom37 到 {self.output_format.upper()} 批量转换器初始化完成:")
        logger.info(f"  - 工作进程数: {self.n_workers}")
        logger.info(f"  - 保持目录结构: {preserve_structure}")
        logger.info(f"  - 输出格式: {self.output_format.upper()}")
    
    def find_atom37_files(
        self,
        input_path: Path,
        recursive: bool = True
    ) -> List[Path]:
        """
        查找所有 Atom37 文件 (.pt 格式)。
        
        Args:
            input_path: 输入路径
            recursive: 是否递归搜索
            
        Returns:
            Atom37 文件路径列表
        """
        if input_path.is_file():
            if input_path.suffix == '.pt':
                return [input_path]
            else:
                raise ValueError(f"输入文件必须是 .pt 格式: {input_path}")
        
        if not input_path.is_dir():
            raise ValueError(f"输入路径不存在: {input_path}")
        
        # 查找所有 .pt 文件
        files = []
        if recursive:
            files.extend(input_path.rglob("*.pt"))
        else:
            files.extend(input_path.glob("*.pt"))
        
        files = sorted(files)
        logger.info(f"在 {input_path} 中找到 {len(files)} 个 Atom37 文件")
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
            'num_chains': 0,
            'output_format': self.output_format
        }
        
        start_time = time.perf_counter()
        
        try:
            logger.debug(f"开始转换: {input_file}")
            
            # 1. 加载 Atom37 数据
            atom37 = Atom37.load(input_file)
            
            # 2. 确保输出目录存在
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 3. 根据格式保存文件
            if self.output_format == "cif":
                atom37.to_cif(str(output_file))
            else:  # pdb
                # 转换为 ProteinTensor 然后保存为 PDB
                protein_tensor = atom37.to_protein_tensor()
                from protein_tensor import save_structure
                save_structure(protein_tensor, str(output_file), format_type="pdb")
            
            # 4. 收集统计信息
            result.update({
                'success': True,
                'num_residues': atom37.num_residues,
                'num_atoms': atom37.atom_mask.sum().item(),
                'num_chains': atom37.num_chains,
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
        # 查找所有 Atom37 文件
        atom37_files = self.find_atom37_files(input_path, recursive)
        
        if not atom37_files:
            logger.warning("没有找到 Atom37 文件")
            return {'total': 0, 'success': 0, 'failed': 0, 'failed_files': [], 'results': []}
        
        # 准备转换任务
        tasks = []
        for file_path in atom37_files:
            # 确定输出路径
            if self.preserve_structure and input_path.is_dir():
                # 保持相对目录结构
                rel_path = file_path.relative_to(input_path)
            else:
                # 扁平输出结构
                rel_path = file_path.name
            
            # 根据输出格式确定扩展名
            extension = f".{self.output_format}"
            output_name = Path(rel_path).with_suffix(extension)
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
                                  self.output_format): (input_file, output_file)
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
                            'num_chains': 0,
                            'output_format': self.output_format
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
            'results': results,
            'converter_settings': {
                'output_format': self.output_format,
                'workers': self.n_workers,
                'preserve_structure': self.preserve_structure
            }
        }
        
        logger.info(f"批量转换完成: {success_count} 成功, {failed_count} 失败")
        return statistics


def convert_single_worker(
    input_file: str,
    output_file: str,
    output_format: str
) -> Dict[str, Any]:
    """
    工作进程函数，用于并行处理。
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        output_format: 输出格式
        
    Returns:
        转换结果
    """
    try:
        # 创建新的转换器实例
        converter = BatchAtom37ToCIFConverter(
            n_workers=1,
            output_format=output_format
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
            'num_chains': 0,
            'output_format': output_format
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
            if hasattr(value, 'item'):  # torch tensor 或 numpy scalar
                clean_result[key] = value.item() if hasattr(value, 'item') else float(value)
        clean_results.append(clean_result)
    
    clean_stats['results'] = clean_results
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"统计信息已保存到: {output_file}") 