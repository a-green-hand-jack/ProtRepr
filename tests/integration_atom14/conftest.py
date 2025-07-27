"""
Atom14 集成测试配置

为 Atom14 集成测试提供共享的配置和 fixtures。
"""

import pytest
import torch
from pathlib import Path


def pytest_configure(config):
    """pytest 配置钩子"""
    # 设置 PyTorch 默认设置
    torch.set_default_dtype(torch.float32)
    torch.set_grad_enabled(False)
    
    # 设置随机种子确保可重现性
    torch.manual_seed(42)


@pytest.fixture(scope="session")
def integration_test_dir() -> Path:
    """集成测试根目录"""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """测试数据目录"""
    return Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session") 
def device() -> torch.device:
    """计算设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") 