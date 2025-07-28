#!/bin/bash
,.
# RCSB蛋白质数据库批量下载脚本
# 支持自动解压、进度显示和并行下载
# 使用 -h 选项获取使用帮助

# 检查必要的命令是否可用
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "错误：找不到 '$1' 命令。请安装 '$1' 后再运行此脚本。"
        exit 1
    fi
}

check_command curl
check_command gzip

PROGNAME=$0
BASE_URL="https://files.rcsb.org/download"
# 默认并行数为CPU核心数
DEFAULT_PARALLEL=$(nproc 2>/dev/null || echo 4)

usage() {
    cat << EOF >&2
用法: $PROGNAME -f <文件> [-o <目录>] [-j <并行数>] [文件类型选项]

必需参数:
 -f <文件>    : 包含PDB ID列表的输入文件（逗号分隔）

可选参数:
 -o <目录>    : 输出目录，默认：当前目录
 -j <并行数>  : 并行下载数，默认：${DEFAULT_PARALLEL}（CPU核心数）
 
文件类型选项（至少选择一种）:
 -c           : 下载 cif.gz 文件
 -p           : 下载 pdb.gz 文件（大型结构不可用）
 -a           : 下载 pdb1.gz 文件（第1个生物学装配体，大型结构不可用）
 -A           : 下载 assembly1.cif.gz 文件（第1个生物学装配体）
 -x           : 下载 xml.gz 文件
 -s           : 下载 sf.cif.gz 文件（仅衍射数据）
 -m           : 下载 mr.gz 文件（仅NMR数据）
 -r           : 下载 mr.str.gz 文件（仅NMR数据）

注意: 所有下载的压缩文件将自动解压，原压缩文件将被删除。
EOF
    exit 1
}



# 参数解析
listfile=""
outdir="."
parallel_jobs=$DEFAULT_PARALLEL
cif=false
pdb=false
pdb1=false
cifassembly1=false
xml=false
sf=false
mr=false
mrstr=false

while getopts f:o:j:cpaAxsmrh o; do
    case $o in
        (f) listfile=$OPTARG;;
        (o) outdir=$OPTARG;;
        (j) parallel_jobs=$OPTARG;;
        (c) cif=true;;
        (p) pdb=true;;
        (a) pdb1=true;;
        (A) cifassembly1=true;;
        (x) xml=true;;
        (s) sf=true;;
        (m) mr=true;;
        (r) mrstr=true;;
        (h) usage;;
        (*) usage;;
    esac
done
shift "$((OPTIND - 1))"

# 验证参数
if [ "$listfile" == "" ]; then
    echo "错误: 必须提供 -f 参数指定PDB ID列表文件"
    exit 1
fi

if [ ! -f "$listfile" ]; then
    echo "错误: 文件不存在: $listfile"
    exit 1
fi

# 检查是否至少选择了一种文件类型
if [ "$cif" = false ] && [ "$pdb" = false ] && [ "$pdb1" = false ] && \
   [ "$cifassembly1" = false ] && [ "$xml" = false ] && [ "$sf" = false ] && \
   [ "$mr" = false ] && [ "$mrstr" = false ]; then
    echo "错误: 必须至少选择一种文件类型（-c, -p, -a, -A, -x, -s, -m, -r）"
    exit 1
fi

# 创建输出目录
if [ ! -d "$outdir" ]; then
    echo "创建输出目录: $outdir"
    mkdir -p "$outdir"
fi

# 验证并行任务数
if ! [[ "$parallel_jobs" =~ ^[0-9]+$ ]] || [ "$parallel_jobs" -lt 1 ]; then
    echo "错误: 并行任务数必须是正整数，当前值: $parallel_jobs"
    exit 1
fi

echo "=== RCSB蛋白质数据库批量下载器 ==="
echo "输入文件: $listfile"
echo "输出目录: $outdir"
echo "并行任务数: $parallel_jobs"
echo "==================================="

# 读取PDB ID列表
contents=$(cat "$listfile")
IFS=',' read -ra pdb_ids <<< "$contents"

# 构建下载文件列表
download_list=()
for pdb_id in "${pdb_ids[@]}"; do
    # 去除空格
    pdb_id=$(echo "$pdb_id" | tr -d ' ')
    [ -z "$pdb_id" ] && continue
    
    [ "$cif" = true ] && download_list+=("${pdb_id}.cif.gz")
    [ "$pdb" = true ] && download_list+=("${pdb_id}.pdb.gz")
    [ "$pdb1" = true ] && download_list+=("${pdb_id}.pdb1.gz")
    [ "$cifassembly1" = true ] && download_list+=("${pdb_id}-assembly1.cif.gz")
    [ "$xml" = true ] && download_list+=("${pdb_id}.xml.gz")
    [ "$sf" = true ] && download_list+=("${pdb_id}-sf.cif.gz")
    [ "$mr" = true ] && download_list+=("${pdb_id}.mr.gz")
    [ "$mrstr" = true ] && download_list+=("${pdb_id}_mr.str.gz")
done

total_files=${#download_list[@]}
echo "总共需要下载 $total_files 个文件"
echo "开始下载..."

# 使用xargs进行并行下载
printf '%s\n' "${download_list[@]}" | \
    nl -nln | \
    xargs -n2 -P"$parallel_jobs" bash -c '
        index="$1"
        filename="$2"
        pdb_id=$(echo "$filename" | cut -d"." -f1 | cut -d"-" -f1 | cut -d"_" -f1)
        
        url="'"$BASE_URL"'/$filename"
        output_path="'"$outdir"'/$filename"
        
        echo "[$index/'"$total_files"'] 正在下载: $filename"
        
        # 使用curl下载，显示进度条
        if curl --progress-bar -f "$url" -o "$output_path"; then
            echo "[$index/'"$total_files"'] 下载完成: $filename"
            
            # 自动解压
            echo "[$index/'"$total_files"'] 正在解压: $filename"
            if gzip -d "$output_path"; then
                echo "[$index/'"$total_files"'] 解压完成: ${filename%.gz}"
            else
                echo "[$index/'"$total_files"'] 警告: 解压失败: $filename"
            fi
        else
            echo "[$index/'"$total_files"'] 错误: 下载失败: $url"
        fi
    ' --

echo "==================================="
echo "所有下载任务已完成！"
echo "下载的文件已自动解压到: $outdir"
echo "==================================="








