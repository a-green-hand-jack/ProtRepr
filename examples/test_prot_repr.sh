#!/bin/bash

# ProtRepr 蛋白质表示转换测试脚本
# 支持完整的端到端测试流程：下载数据 -> 环境准备 -> 数据转换 -> 验证结果
# 

set -euo pipefail  # 严格模式：遇到错误立即退出，未定义变量报错，管道错误传播

# ================================
# 配置参数 (可自定义的配置项)
# ================================

# 基础目录配置
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$SCRIPT_DIR"

# 数据目录配置
readonly RAW_DATA_DIR="${RAW_DATA_DIR:-${PROJECT_ROOT}/raw_structures}"
readonly ATOM14_DIR="${ATOM14_DIR:-${PROJECT_ROOT}/atom14}"
readonly ATOM37_DIR="${ATOM37_DIR:-${PROJECT_ROOT}/atom37}"
readonly FRAME_DIR="${FRAME_DIR:-${PROJECT_ROOT}/frame}"
readonly VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"

# 下载配置
readonly PDB_LIST_FILE="${PDB_LIST_FILE:-${PROJECT_ROOT}/list_file.txt}"
readonly DOWNLOAD_PARALLEL_JOBS="${DOWNLOAD_PARALLEL_JOBS:-8}"
readonly DOWNLOAD_SCRIPT="${PROJECT_ROOT}/batch_download.sh"

# ProtRepr 配置
readonly PROTREPR_REPO="${PROTREPR_REPO:-git+ssh://git@github.com/a-green-hand-jack/ProtRepr.git}"

# 测试配置
readonly TEST_SUBSET_SIZE="${TEST_SUBSET_SIZE:-5}"  # 默认只测试前5个结构
readonly CLEANUP_TEMP_FILES="${CLEANUP_TEMP_FILES:-true}"

# ================================
# 工具函数
# ================================

# 日志函数
log_info() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [信息] $*"
}

log_warning() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [警告] $*" >&2
}

log_error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [错误] $*" >&2
}

log_success() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [成功] $*"
}

# 进度显示函数
show_progress() {
    local current=$1
    local total=$2
    local desc=$3
    local percent=$((current * 100 / total))
    printf "进度: [%3d%%] (%d/%d) %s\n" "$percent" "$current" "$total" "$desc"
}

# 检查命令是否存在
check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "找不到必需的命令: $1"
        log_error "请安装 $1 后重试"
        exit 1
    fi
}

# 检查文件是否存在
check_file() {
    if [[ ! -f "$1" ]]; then
        log_error "找不到必需的文件: $1"
        exit 1
    fi
}

# 创建目录（如果不存在）
ensure_directory() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        log_info "创建目录: $dir"
        mkdir -p "$dir"
    fi
}

# 统计目录中的文件数量
count_files() {
    local dir="$1"
    local pattern="${2:-*}"
    if [[ -d "$dir" ]]; then
        find "$dir" -name "$pattern" -type f 2>/dev/null | wc -l
    else
        echo "0"
    fi
}

# ================================
# 主要功能函数
# ================================

# 显示使用帮助
show_usage() {
    cat << EOF
用法: $0 [选项]

ProtRepr 蛋白质表示转换测试脚本

选项:
  -h, --help                显示此帮助信息
  -s, --subset SIZE         只测试前 SIZE 个结构 (默认: $TEST_SUBSET_SIZE)
  -j, --jobs JOBS           下载并行任务数 (默认: $DOWNLOAD_PARALLEL_JOBS)
  -c, --cleanup             测试完成后清理临时文件 (默认: $CLEANUP_TEMP_FILES)
  --skip-download           跳过数据下载步骤
  --skip-env                跳过环境设置步骤
  --only-atom14             只运行 atom14 转换测试
  --only-atom37             只运行 atom37 转换测试
  --only-frame              只运行 frame 转换测试

环境变量:
  RAW_DATA_DIR              原始结构数据目录 (默认: ./raw_structures)
  ATOM14_DIR                atom14 数据目录 (默认: ./atom14)
  ATOM37_DIR                atom37 数据目录 (默认: ./atom37)
  FRAME_DIR                 frame 数据目录 (默认: ./frame)
  VENV_DIR                  虚拟环境目录 (默认: ./.venv)
  PDB_LIST_FILE             PDB ID 列表文件 (默认: ./list_file.txt)
  DOWNLOAD_PARALLEL_JOBS    下载并行数 (默认: 8)
  TEST_SUBSET_SIZE          测试结构数量 (默认: 5)

示例:
  $0                        # 使用默认设置运行完整测试
  $0 -s 10 -j 16           # 测试10个结构，使用16个并行下载
  $0 --only-atom14         # 只运行 atom14 转换测试
  $0 --skip-download       # 跳过下载，使用现有数据
EOF
}

# 解析命令行参数
parse_arguments() {
    local subset_size="$TEST_SUBSET_SIZE"
    local parallel_jobs="$DOWNLOAD_PARALLEL_JOBS"
    local cleanup="$CLEANUP_TEMP_FILES"
    local skip_download=false
    local skip_env=false
    local only_atom14=false
    local only_atom37=false
    local only_frame=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -s|--subset)
                subset_size="$2"
                shift 2
                ;;
            -j|--jobs)
                parallel_jobs="$2"
                shift 2
                ;;
            -c|--cleanup)
                cleanup=true
                shift
                ;;
            --skip-download)
                skip_download=true
                shift
                ;;
            --skip-env)
                skip_env=true
                shift
                ;;
            --only-atom14)
                only_atom14=true
                shift
                ;;
            --only-atom37)
                only_atom37=true
                shift
                ;;
            --only-frame)
                only_frame=true
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # 导出解析后的参数
    export TEST_SUBSET_SIZE="$subset_size"
    export DOWNLOAD_PARALLEL_JOBS="$parallel_jobs"
    export CLEANUP_TEMP_FILES="$cleanup"
    export SKIP_DOWNLOAD="$skip_download"
    export SKIP_ENV="$skip_env"
    export ONLY_ATOM14="$only_atom14"
    export ONLY_ATOM37="$only_atom37"
    export ONLY_FRAME="$only_frame"
}

# 环境检查
check_environment() {
    log_info "检查运行环境..."
    
    # 检查必需的命令
    local required_commands=(curl gzip uv)
    for cmd in "${required_commands[@]}"; do
        check_command "$cmd"
    done
    
    # 检查必需的文件
    check_file "$DOWNLOAD_SCRIPT"
    check_file "$PDB_LIST_FILE"
    
    # 检查下载脚本是否可执行
    if [[ ! -x "$DOWNLOAD_SCRIPT" ]]; then
        log_info "设置下载脚本执行权限"
        chmod +x "$DOWNLOAD_SCRIPT"
    fi
    
    log_success "环境检查通过"
}

# 下载蛋白质结构数据
download_structures() {
    if [[ "$SKIP_DOWNLOAD" == "true" ]]; then
        log_info "跳过数据下载步骤"
        return 0
    fi

    log_info "开始下载蛋白质结构数据..."
    log_info "目标目录: $RAW_DATA_DIR"
    log_info "并行任务数: $DOWNLOAD_PARALLEL_JOBS"
    
    ensure_directory "$RAW_DATA_DIR"
    
    # 如果需要测试子集，创建临时的PDB列表文件
    local pdb_list_file="$PDB_LIST_FILE"
    if [[ "$TEST_SUBSET_SIZE" != "all" ]] && [[ "$TEST_SUBSET_SIZE" -gt 0 ]]; then
        local temp_list_file="${PROJECT_ROOT}/temp_pdb_list.txt"
        log_info "创建测试子集 (前 $TEST_SUBSET_SIZE 个结构)"
        
        # 读取原始列表并取前N个
        local original_list
        original_list=$(cat "$PDB_LIST_FILE")
        echo "$original_list" | cut -d',' -f1-"$TEST_SUBSET_SIZE" > "$temp_list_file"
        pdb_list_file="$temp_list_file"
    fi
    
    # 执行下载
    if "$DOWNLOAD_SCRIPT" -f "$pdb_list_file" -o "$RAW_DATA_DIR" -c -j "$DOWNLOAD_PARALLEL_JOBS"; then
        local downloaded_count
        downloaded_count=$(count_files "$RAW_DATA_DIR" "*.cif")
        log_success "数据下载完成！下载了 $downloaded_count 个结构文件"
        
        # 清理临时文件
        [[ -f "${PROJECT_ROOT}/temp_pdb_list.txt" ]] && rm -f "${PROJECT_ROOT}/temp_pdb_list.txt"
    else
        log_error "数据下载失败"
        exit 1
    fi
}

# 设置Python环境
setup_environment() {
    if [[ "$SKIP_ENV" == "true" ]]; then
        log_info "跳过环境设置步骤"
        return 0
    fi

    log_info "设置Python虚拟环境..."
    
    # 创建虚拟环境
    if [[ ! -d "$VENV_DIR" ]]; then
        log_info "创建虚拟环境: $VENV_DIR"
        uv venv "$VENV_DIR"
    else
        log_info "使用现有虚拟环境: $VENV_DIR"
    fi
    
    # 激活虚拟环境
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    
    # 安装 ProtRepr
    log_info "安装 ProtRepr 库..."
    log_info "仓库地址: $PROTREPR_REPO"
    
    if uv pip install "$PROTREPR_REPO"; then
        log_success "ProtRepr 安装成功"
        
        # 验证安装
        log_info "验证 ProtRepr 命令..."
        local commands=(
            "protrepr-struct-to-atom14"
            "protrepr-atom14-to-struct"
            "protrepr-struct-to-atom37"
            "protrepr-atom37-to-struct"
            "protrepr-struct-to-frame"
            "protrepr-frame-to-struct"
        )
        
        for cmd in "${commands[@]}"; do
            if command -v "$cmd" &> /dev/null; then
                log_info "✓ $cmd 可用"
            else
                log_warning "✗ $cmd 不可用"
            fi
        done
    else
        log_error "ProtRepr 安装失败"
        exit 1
    fi
}

# 运行转换测试
run_conversion_tests() {
    log_info "开始运行数据转换测试..."
    
    # 创建输出目录
    local dirs_to_create=(
        "$ATOM14_DIR/e2e"
        "$ATOM14_DIR/reconstructed"
        "$ATOM37_DIR/e2e"
        "$ATOM37_DIR/reconstructed"
        "$FRAME_DIR/e2e"
        "$FRAME_DIR/reconstructed"
    )
    
    for dir in "${dirs_to_create[@]}"; do
        ensure_directory "$dir"
    done
    
    local total_steps=6
    local current_step=0
    
    # 根据参数决定运行哪些测试
    local run_atom14=true
    local run_atom37=true
    local run_frame=true
    
    if [[ "$ONLY_ATOM14" == "true" ]]; then
        run_atom37=false
        run_frame=false
        total_steps=2
    elif [[ "$ONLY_ATOM37" == "true" ]]; then
        run_atom14=false
        run_frame=false
        total_steps=2
    elif [[ "$ONLY_FRAME" == "true" ]]; then
        run_atom14=false
        run_atom37=false
        total_steps=2
    fi
    
    # Atom14 转换测试
    if [[ "$run_atom14" == "true" ]]; then
        current_step=$((current_step + 1))
        show_progress "$current_step" "$total_steps" "结构 -> atom14"
        log_info "执行: protrepr-struct-to-atom14 $RAW_DATA_DIR $ATOM14_DIR/e2e"
        if protrepr-struct-to-atom14 "$RAW_DATA_DIR" "$ATOM14_DIR/e2e"; then
            local atom14_count
            atom14_count=$(count_files "$ATOM14_DIR/e2e")
            log_success "atom14 转换完成 ($atom14_count 个文件)"
        else
            log_error "atom14 转换失败"
            exit 1
        fi
        
        current_step=$((current_step + 1))
        show_progress "$current_step" "$total_steps" "atom14 -> 结构"
        log_info "执行: protrepr-atom14-to-struct $ATOM14_DIR/e2e $ATOM14_DIR/reconstructed"
        if protrepr-atom14-to-struct "$ATOM14_DIR/e2e" "$ATOM14_DIR/reconstructed"; then
            local reconstructed_count
            reconstructed_count=$(count_files "$ATOM14_DIR/reconstructed")
            log_success "atom14 重构完成 ($reconstructed_count 个文件)"
        else
            log_error "atom14 重构失败"
            exit 1
        fi
    fi
    
    # Atom37 转换测试
    if [[ "$run_atom37" == "true" ]]; then
        current_step=$((current_step + 1))
        show_progress "$current_step" "$total_steps" "结构 -> atom37"
        log_info "执行: protrepr-struct-to-atom37 $RAW_DATA_DIR $ATOM37_DIR/e2e"
        if protrepr-struct-to-atom37 "$RAW_DATA_DIR" "$ATOM37_DIR/e2e"; then
            local atom37_count
            atom37_count=$(count_files "$ATOM37_DIR/e2e")
            log_success "atom37 转换完成 ($atom37_count 个文件)"
        else
            log_error "atom37 转换失败"
            exit 1
        fi
        
        current_step=$((current_step + 1))
        show_progress "$current_step" "$total_steps" "atom37 -> 结构"
        log_info "执行: protrepr-atom37-to-struct $ATOM37_DIR/e2e $ATOM37_DIR/reconstructed"
        if protrepr-atom37-to-struct "$ATOM37_DIR/e2e" "$ATOM37_DIR/reconstructed"; then
            local reconstructed_count
            reconstructed_count=$(count_files "$ATOM37_DIR/reconstructed")
            log_success "atom37 重构完成 ($reconstructed_count 个文件)"
        else
            log_error "atom37 重构失败"
            exit 1
        fi
    fi
    
    # Frame 转换测试
    if [[ "$run_frame" == "true" ]]; then
        current_step=$((current_step + 1))
        show_progress "$current_step" "$total_steps" "结构 -> frame"
        log_info "执行: protrepr-struct-to-frame $RAW_DATA_DIR $FRAME_DIR/e2e"
        if protrepr-struct-to-frame "$RAW_DATA_DIR" "$FRAME_DIR/e2e"; then
            local frame_count
            frame_count=$(count_files "$FRAME_DIR/e2e")
            log_success "frame 转换完成 ($frame_count 个文件)"
        else
            log_error "frame 转换失败"
            exit 1
        fi
        
        current_step=$((current_step + 1))
        show_progress "$current_step" "$total_steps" "frame -> 结构"
        log_info "执行: protrepr-frame-to-struct $FRAME_DIR/e2e $FRAME_DIR/reconstructed"
        if protrepr-frame-to-struct "$FRAME_DIR/e2e" "$FRAME_DIR/reconstructed"; then
            local reconstructed_count
            reconstructed_count=$(count_files "$FRAME_DIR/reconstructed")
            log_success "frame 重构完成 ($reconstructed_count 个文件)"
        else
            log_error "frame 重构失败"
            exit 1
        fi
    fi
}

# 生成测试报告
generate_report() {
    log_info "生成测试报告..."
    
    echo ""
    echo "=================================="
    echo "        ProtRepr 测试报告"
    echo "=================================="
    echo "测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "测试配置:"
    echo "  - 结构数量: $TEST_SUBSET_SIZE"
    echo "  - 下载并行数: $DOWNLOAD_PARALLEL_JOBS"
    echo "  - 原始数据目录: $RAW_DATA_DIR"
    echo ""
    
    echo "文件统计:"
    printf "  - 原始结构: %d 个文件\n" "$(count_files "$RAW_DATA_DIR" "*.cif")"
    
    if [[ "$ONLY_ATOM14" != "true" ]] || [[ "$ONLY_ATOM14" == "true" ]]; then
        printf "  - Atom14 转换: %d 个文件\n" "$(count_files "$ATOM14_DIR/e2e")"
        printf "  - Atom14 重构: %d 个文件\n" "$(count_files "$ATOM14_DIR/reconstructed")"
    fi
    
    if [[ "$ONLY_ATOM37" != "true" ]] || [[ "$ONLY_ATOM37" == "true" ]]; then
        printf "  - Atom37 转换: %d 个文件\n" "$(count_files "$ATOM37_DIR/e2e")"
        printf "  - Atom37 重构: %d 个文件\n" "$(count_files "$ATOM37_DIR/reconstructed")"
    fi
    
    if [[ "$ONLY_FRAME" != "true" ]] || [[ "$ONLY_FRAME" == "true" ]]; then
        printf "  - Frame 转换: %d 个文件\n" "$(count_files "$FRAME_DIR/e2e")"
        printf "  - Frame 重构: %d 个文件\n" "$(count_files "$FRAME_DIR/reconstructed")"
    fi
    
    echo ""
    echo "目录结构:"
    echo "  $PROJECT_ROOT/"
    [[ -d "$RAW_DATA_DIR" ]] && echo "  ├── $(basename "$RAW_DATA_DIR")/ (原始结构)"
    [[ -d "$ATOM14_DIR" ]] && echo "  ├── $(basename "$ATOM14_DIR")/ (atom14数据)"
    [[ -d "$ATOM37_DIR" ]] && echo "  ├── $(basename "$ATOM37_DIR")/ (atom37数据)"
    [[ -d "$FRAME_DIR" ]] && echo "  ├── $(basename "$FRAME_DIR")/ (frame数据)"
    [[ -d "$VENV_DIR" ]] && echo "  └── $(basename "$VENV_DIR")/ (Python环境)"
    echo ""
    echo "=================================="
}

# 清理临时文件
cleanup_temp_files() {
    if [[ "$CLEANUP_TEMP_FILES" == "true" ]]; then
        log_info "清理临时文件..."
        
        # 可以在这里添加清理逻辑，比如删除中间文件
        local temp_files=(
            "${PROJECT_ROOT}/temp_pdb_list.txt"
        )
        
        for file in "${temp_files[@]}"; do
            if [[ -f "$file" ]]; then
                rm -f "$file"
                log_info "已删除: $file"
            fi
        done
        
        log_success "临时文件清理完成"
    fi
}

# ================================
# 主函数
# ================================

main() {
    local start_time
    start_time=$(date +%s)
    
    echo "========================================"
    echo "    ProtRepr 蛋白质表示转换测试脚本"
    echo "========================================"
    echo ""
    
    # 解析命令行参数
    parse_arguments "$@"
    
    # 运行测试流程
    check_environment
    download_structures
    setup_environment
    run_conversion_tests
    generate_report
    cleanup_temp_files
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo ""
    log_success "所有测试完成！总耗时: ${duration}秒"
    echo "========================================"
}

# 脚本入口点
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi