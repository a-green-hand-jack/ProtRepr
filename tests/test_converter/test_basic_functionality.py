"""
批量转换脚本基础功能测试

简单测试批量转换脚本的基本功能。
"""

import subprocess
import sys
from pathlib import Path

import pytest


class TestBatchConverterScript:
    """测试批量转换脚本的基本功能。"""
    
    def test_script_exists(self):
        """测试脚本文件是否存在。"""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "batch_pdb_to_atom14.py"
        assert script_path.exists(), "批量转换脚本不存在"
        assert script_path.is_file(), "脚本路径不是文件"
    
    def test_script_help(self):
        """测试脚本的帮助功能。"""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "batch_pdb_to_atom14.py"
        
        if not script_path.exists():
            pytest.skip("批量转换脚本不存在")
        
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, "脚本帮助命令失败"
        assert "input_path" in result.stdout, "帮助信息缺少输入路径说明"
        assert "output_dir" in result.stdout, "帮助信息缺少输出目录说明"
    
    def test_script_missing_args(self):
        """测试脚本缺少参数时的行为。"""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "batch_pdb_to_atom14.py"
        
        if not script_path.exists():
            pytest.skip("批量转换脚本不存在")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # 缺少参数应该返回错误代码
        assert result.returncode != 0, "脚本应该在缺少参数时返回错误代码" 