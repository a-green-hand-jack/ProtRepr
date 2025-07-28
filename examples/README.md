# 下载数据并测试命令行工具

## 🚀 主要改进

### 1. **完全参数化配置**
- 所有目录名都可以通过环境变量自定义：
  ```bash
  RAW_DATA_DIR    # 原始结构数据目录 (默认: ./raw_structures)
  ATOM14_DIR      # atom14 数据目录 (默认: ./atom14)
  ATOM37_DIR      # atom37 数据目录 (默认: ./atom37)
  FRAME_DIR       # frame 数据目录 (默认: ./frame)
  VENV_DIR        # 虚拟环境目录 (默认: ./.venv)
  ```

### 2. **丰富的命令行选项**
```bash
./test_prot_repr.sh -h                    # 查看完整帮助
./test_prot_repr.sh -s 10 -j 16          # 测试10个结构，16并行下载
./test_prot_repr.sh --only-atom14        # 只运行 atom14 转换测试
./test_prot_repr.sh --skip-download      # 跳过下载，使用现有数据
```

### 3. **智能化功能**
- **进度显示**：`进度: [50%] (3/6) 结构 -> atom14`
- **文件统计**：自动统计各阶段生成的文件数量
- **环境检查**：自动验证必需命令和文件
- **错误恢复**：出错时提供详细的错误信息

### 4. **专业级日志系统**
```bash
2024-01-15 18:22:45 [信息] 检查运行环境...
2024-01-15 18:22:45 [成功] 环境检查通过
2024-01-15 18:22:46 [警告] ✗ protrepr-frame-to-struct 不可用
```

### 5. **完整测试报告**
脚本会自动生成详细的测试报告，包括：
- 测试配置信息
- 文件统计
- 目录结构
- 总耗时

### 6. **健壮性增强**
- **严格模式**：`set -euo pipefail` 确保脚本遇到错误立即停止
- **资源清理**：自动清理临时文件
- **模块化设计**：每个功能都是独立的函数，便于维护

## 📋 使用示例

```bash
# 基本使用（测试前5个结构）
./test_prot_repr.sh

# 测试更多结构，提高并行度
./test_prot_repr.sh -s 20 -j 16

# 只测试特定转换类型
./test_prot_repr.sh --only-atom14

# 使用自定义目录
RAW_DATA_DIR=./my_structures ./test_prot_repr.sh

# 跳过某些步骤（适合调试）
./test_prot_repr.sh --skip-download --only-frame
```

这个脚本现在符合你的编程规范：
- ✅ 中文注释和日志
- ✅ 严格的错误处理
- ✅ 参数化配置
- ✅ 模块化设计
- ✅ 用户友好的界面
- ✅ 专业的测试流程

脚本已经可以直接使用了！你可以先用 `./test_prot_repr.sh -h` 查看所有可用选项。