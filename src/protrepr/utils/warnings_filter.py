"""
警告过滤器工具

用于过滤第三方库产生的非关键警告信息。
"""

import warnings
import logging

logger = logging.getLogger(__name__)


def suppress_biopython_warnings():
    """
    抑制 BioPython 中的非关键警告。
    
    主要过滤：
    - PDBConstructionWarning: 元素自动推断警告
    - PDBIOWarning: PDB 文件 I/O 相关警告
    """
    try:
        from Bio.PDB import PDBConstructionWarning, PDBIOWarning  # type: ignore
        warnings.filterwarnings('ignore', category=PDBConstructionWarning)  # type: ignore
        warnings.filterwarnings('ignore', category=PDBIOWarning)  # type: ignore
        logger.debug("已过滤 BioPython 非关键警告")
    except ImportError:
        logger.debug("BioPython 未安装，跳过警告过滤")


def suppress_protein_tensor_warnings():
    """
    抑制 ProteinTensor 及相关库的非关键警告。
    """
    # 过滤常见的张量操作警告
    warnings.filterwarnings('ignore', message='.*grad_fn.*')
    warnings.filterwarnings('ignore', message='.*non_blocking.*')
    logger.debug("已过滤 ProteinTensor 相关警告")


def apply_default_filters():
    """
    应用默认的警告过滤器设置。
    
    这个函数会在包导入时自动调用，或者可以手动调用以确保警告过滤器生效。
    """
    suppress_biopython_warnings()
    suppress_protein_tensor_warnings()
    logger.info("警告过滤器已应用")


def reset_warnings():
    """
    重置警告过滤器到默认状态。
    
    用于调试时恢复所有警告信息。
    """
    warnings.resetwarnings()
    logger.info("警告过滤器已重置")


# 可选：在导入时自动应用过滤器
# apply_default_filters() 