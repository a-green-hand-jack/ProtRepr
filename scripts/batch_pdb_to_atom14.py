#!/usr/bin/env python3
"""
批量 PDB 到 Atom14 转换脚本

这个脚本提供命令行接口，用于批量将 PDB/CIF 文件转换为 ProtRepr 的 Atom14 格式。
核心实现位于 protrepr.batch_processing 模块中。

使用方法:
    python batch_pdb_to_atom14.py input_dir output_dir [options]

示例:
    # 基本用法 (保存为 Atom14 实例)
    python batch_pdb_to_atom14.py /path/to/pdb_files /path/to/output
    
    # 保存为字典格式
    python batch_pdb_to_atom14.py /path/to/pdb_files /path/to/output --save-as-dict
    
    # 使用并行处理
    python batch_pdb_to_atom14.py /path/to/pdb_files /path/to/output --workers 8
    
    # 不保持目录结构
    python batch_pdb_to_atom14.py /path/to/pdb_files /path/to/output --no-preserve-structure
    
    # 指定设备
    python batch_pdb_to_atom14.py /path/to/pdb_files /path/to/output --device cuda
"""

import sys
from pathlib import Path
import argparse
import logging
import time

# 将项目根目录下的 src 目录添加到 Python 解释器的搜索路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from protrepr.batch_processing import (
    BatchPDBToAtom14Converter,
    save_statistics
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description="批量将 PDB/CIF 文件转换为 ProtRepr Atom14 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s /data/pdb_files /data/atom14_output
  %(prog)s /data/pdb_files /data/atom14_output --workers 8 --save-as-dict
  %(prog)s input.pdb output_dir --no-preserve-structure --device cuda
        """
    )
    
    parser.add_argument(
        'input_path',
        type=Path,
        help='输入文件或目录路径'
    )
    
    parser.add_argument(
        'output_dir',
        type=Path,
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='并行工作进程数 (默认: CPU核心数的一半)'
    )
    
    parser.add_argument(
        '--no-preserve-structure',
        action='store_true',
        help='不保持目录结构，所有输出文件放在同一目录'
    )
    
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        default=True,
        help='递归搜索子目录 (默认: True)'
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cpu',
        help='计算设备 (默认: cpu)'
    )
    
    parser.add_argument(
        '--save-as-dict',
        action='store_true',
        help='保存为字典格式而非 Atom14 实例 (默认: 保存为实例)'
    )
    
    parser.add_argument(
        '--save-stats',
        type=Path,
        help='保存统计信息到指定的 JSON 文件'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出模式'
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 验证输入
    if not args.input_path.exists():
        logger.error(f"输入路径不存在: {args.input_path}")
        return 1
    
    # 检查 CUDA 可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退到 CPU")
        args.device = 'cpu'
    
    try:
        # 创建转换器
        converter = BatchPDBToAtom14Converter(
            n_workers=args.workers,
            preserve_structure=not args.no_preserve_structure,
            device=args.device,
            save_as_instance=not args.save_as_dict  # 默认保存为实例
        )
        
        # 执行批量转换
        logger.info(f"开始批量转换: {args.input_path} -> {args.output_dir}")
        save_format = "字典格式" if args.save_as_dict else "Atom14实例"
        logger.info(f"保存格式: {save_format}")
        
        start_time = time.perf_counter()
        
        statistics = converter.convert_batch(
            input_path=args.input_path,
            output_dir=args.output_dir,
            recursive=args.recursive
        )
        
        total_time = time.perf_counter() - start_time
        
        # 输出结果摘要
        logger.info("=" * 60)
        logger.info("转换完成摘要:")
        logger.info(f"  总文件数: {statistics['total']}")
        logger.info(f"  成功转换: {statistics['success']}")
        logger.info(f"  转换失败: {statistics['failed']}")
        logger.info(f"  保存格式: {save_format}")
        logger.info(f"  总用时: {total_time:.2f} 秒")
        
        if statistics['success'] > 0:
            avg_time = total_time / statistics['success']
            logger.info(f"  平均时间: {avg_time:.4f} 秒/文件")
            
            # 计算一些统计数据
            successful_results = [r for r in statistics['results'] if r['success']]
            if successful_results:
                total_residues = sum(r['num_residues'] for r in successful_results)
                total_atoms = sum(r['num_atoms'] for r in successful_results)
                logger.info(f"  总残基数: {total_residues:,}")
                logger.info(f"  总原子数: {total_atoms:,}")
        
        if statistics['failed'] > 0:
            logger.warning(f"失败的文件:")
            for failed_file in statistics['failed_files']:
                logger.warning(f"  - {failed_file}")
        
        # 保存统计信息
        if args.save_stats:
            save_statistics(statistics, args.save_stats)
        
        # 返回适当的退出码
        return 0 if statistics['failed'] == 0 else 1
        
    except Exception as e:
        logger.error(f"批量转换过程中发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 