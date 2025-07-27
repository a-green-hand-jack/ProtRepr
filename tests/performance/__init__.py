"""
性能测试包

包含 ProtRepr 各种表示方法的性能基准测试。
"""

from .benchmark_atom14_performance import *
from .benchmark_optimized_performance import *

__all__ = [
    "benchmark_atom14_performance",
    "benchmark_optimized_performance",
] 