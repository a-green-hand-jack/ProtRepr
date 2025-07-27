# ProtRepr 测试运行指南

## 概述

ProtRepr 拥有完整的测试体系，包括端到端集成测试、单元测试和性能基准测试。本指南将帮您快速运行和理解这些测试。

## 🚀 快速开始

### 运行所有核心测试

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行 Atom14 完整集成测试
pytest tests/integration_atom14/ -v

# 运行所有 Atom14 相关测试
pytest tests/integration_atom14/ tests/test_converter/ tests/test_representations/test_atom14_chain_gap.py -v
```

### 运行单个核心测试

```bash
# 端到端工作流测试
pytest tests/integration_atom14/test_atom14_end_to_end.py::test_complete_workflow -v -s

# 批量处理测试  
pytest tests/integration_atom14/test_atom14_end_to_end.py::TestAtom14EndToEnd::test_batch_processing -v

# 数据一致性测试
pytest tests/integration_atom14/test_atom14_end_to_end.py::TestAtom14EndToEnd::test_complete_workflow -v
```

## 📊 测试类型说明

### 1. 集成测试 (`tests/integration_atom14/`)

**目标**: 验证完整的端到端工作流
- ✅ CIF/PDB 文件加载
- ✅ Atom14 转换
- ✅ NPZ/PyTorch 格式保存/加载
- ✅ 数据一致性验证
- ✅ CIF/PDB 重建

**运行方式**:
```bash
pytest tests/integration_atom14/ -v --tb=short
```

**预期结果**: 所有测试通过，生成详细的测试结果文件在 `tests/integration_atom14/test_results/`

### 2. 单元测试 (`tests/test_converter/`)

**目标**: 验证批量转换器的具体功能
- ✅ 转换器初始化
- ✅ 文件发现逻辑
- ✅ 单文件转换
- ✅ 错误处理
- ✅ 统计功能

**运行方式**:
```bash
pytest tests/test_converter/ -v
```

### 3. 专项测试 (`tests/test_representations/`)

**目标**: 验证特定算法逻辑
- ✅ 链间 Gap 计算正确性
- ✅ 多链蛋白质处理
- ✅ 残基编号映射

**运行方式**:
```bash
pytest tests/test_representations/test_atom14_chain_gap.py -v
```

### 4. 性能测试 (`tests/performance/`)

**目标**: 监控性能表现
- ⚡ 转换速度基准
- 📊 内存使用监控
- 📈 性能对比分析

**运行方式**:
```bash
pytest tests/performance/ -v
```

## 🔍 测试结果解析

### 测试输出文件

运行集成测试后，会在 `tests/integration_atom14/test_results/` 生成以下文件：

```
test_results/
├── workflow_results.json      # 详细的工作流测试结果
├── batch_statistics.json      # 批量处理统计数据
├── 9is2_atom14.npz           # NPZ 格式测试输出
├── 9is2_atom14.pt            # PyTorch 格式测试输出
├── 9is2_rebuilt.cif          # 从 Atom14 重建的 CIF
├── 9is2_reconstructed.cif    # 从 ProteinTensor 重建的 CIF
└── batch_results/            # 批量转换输出目录
```

### 关键性能指标

基于真实数据 (368 残基, 8985 原子) 的参考性能：

| 操作 | 预期时间 | 状态指标 |
|-----|---------|----------|
| 文件加载 | ~0.2s | 正常 |
| Atom14 转换 | ~0.07s | 优秀 |
| 格式保存 | ~0.005s | 极快 |
| 格式加载 | ~0.002s | 极快 |
| **总体工作流** | **~0.8s** | **优秀** |

### 数据一致性验证

所有格式间的数据必须满足以下一致性要求：
- 🎯 坐标精度: `rtol=1e-5, atol=1e-6`
- ✅ 掩码完全一致
- ✅ 元数据完全一致
- ✅ 张量形状完全一致

## 🐛 故障排除

### 常见问题

1. **测试文件缺失**
   ```bash
   # 确保测试数据存在
   ls tests/data/
   # 应该看到 *.cif 文件
   ```

2. **依赖包问题**
   ```bash
   # 重新安装依赖
   uv pip install -e .
   ```

3. **内存不足**
   ```bash
   # 运行较小的测试集
   pytest tests/integration_atom14/test_atom14_end_to_end.py::test_complete_workflow -v
   ```

4. **CUDA 相关错误**
   ```bash
   # 强制使用 CPU
   export CUDA_VISIBLE_DEVICES=""
   pytest tests/integration_atom14/ -v
   ```

### 测试跳过说明

某些测试可能被跳过，这通常是正常的：
- ⏭️ **特定文件缺失**: 测试需要特定的测试文件
- ⏭️ **硬件要求**: 某些测试需要 GPU 环境
- ⏭️ **可选依赖**: 某些功能依赖可选的第三方库

## 📈 持续集成

### 在 CI/CD 中运行

```yaml
# GitHub Actions 示例
- name: Run ProtRepr Tests
  run: |
    source .venv/bin/activate
    pytest tests/integration_atom14/ tests/test_converter/ -v --cov=src/protrepr --cov-report=xml
```

### 性能监控

定期运行性能基准测试：
```bash
# 每周运行一次性能基准
pytest tests/performance/ -v --benchmark-json=benchmark_results.json
```

## 🎯 测试最佳实践

### 开发新功能时

1. **先写测试**: 采用 TDD 方法
2. **运行相关测试**: 确保不破坏现有功能
3. **更新集成测试**: 如果涉及端到端流程

### 提交代码前

```bash
# 运行核心测试套件
pytest tests/integration_atom14/ tests/test_converter/ -v --tb=short

# 检查测试覆盖率
pytest tests/integration_atom14/ --cov=src/protrepr --cov-report=term-missing
```

### 性能回归检测

```bash
# 比较性能基准
pytest tests/performance/benchmark_optimized_performance.py -v
```

## 📚 延伸阅读

- [Atom14 集成测试报告](atom14_integration_test_report.md) - 详细的测试验证报告
- [测试完成总结](atom14_testing_completion_summary.md) - 测试体系建设总结
- [性能优化总结](tensor_optimization_summary.md) - 性能基准和优化历程

---

**测试理念**: 通过全面的测试确保 ProtRepr 的可靠性和性能，为科研和生产环境提供坚实保障。 