"""结构文件到 Atom37 转换命令行工具"""

import sys
import logging
import time
import argparse
from pathlib import Path
from typing import Optional

from ..batch_processing import BatchPDBToAtom37Converter, save_statistics


def main() -> Optional[int]:
    """
    命令行入口函数。
    
    Returns:
        Optional[int]: 退出码，0表示成功，非0表示错误
    """
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(
        description="批量将结构文件（PDB/CIF/ENT/MMCIF）转换为 ProtRepr Atom37 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s /data/structure_files /data/atom37_output
  %(prog)s /data/structure_files /data/atom37_output --workers 8 --save-as-dict
  %(prog)s input.pdb output_dir --no-preserve-structure --device cuda
        """
    )
    
    parser.add_argument(
        'input_path',
        type=Path,
        help='输入结构文件或目录路径（支持 PDB/CIF/ENT/MMCIF 格式）'
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
        '--device',
        choices=['cpu', 'cuda'],
        default='cpu',
        help='计算设备 (默认: cpu)'
    )
    
    parser.add_argument(
        '--save-as-dict',
        action='store_true',
        help='保存为字典格式而非 Atom37 实例'
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
    
    try:
        logger.info("开始批量转换")
        start_time = time.perf_counter()
        
        # 创建转换器
        converter = BatchPDBToAtom37Converter(
            n_workers=args.workers,
            preserve_structure=not args.no_preserve_structure,
            device=args.device,
            save_as_instance=not args.save_as_dict
        )
        
        # 执行转换
        statistics = converter.convert_batch(
            input_path=args.input_path,
            output_dir=args.output_dir,
            recursive=True
        )
        
        total_time = time.perf_counter() - start_time
        
        # 输出结果
        logger.info(f"转换完成! 总用时: {total_time:.2f} 秒")
        logger.info(f"成功: {statistics['success']} 个文件")
        logger.info(f"失败: {statistics['failed']} 个文件")
        
        # 保存统计信息
        if args.save_stats:
            save_statistics(statistics, args.save_stats)
            logger.info(f"统计信息已保存到: {args.save_stats}")
        
        # 显示失败文件
        if statistics['failed'] > 0:
            logger.warning("失败的文件:")
            for failed_file in statistics['failed_files'][:5]:
                logger.warning(f"  - {failed_file}")
            if len(statistics['failed_files']) > 5:
                logger.warning(f"  ... 以及其他 {len(statistics['failed_files']) - 5} 个文件")
        
        return 0
        
    except Exception as e:
        logger.error(f"转换过程中发生错误: {e}")
        if args.verbose:
            import traceback
            logger.error(f"详细错误信息:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 