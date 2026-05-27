#!/bin/bash

# WORKSPACE: 所有依赖的根目录，默认 /workspace
WORKSPACE="${WORKSPACE:-/workspace}"

# LLVM_BUILD_DIR: LLVM 编译产物的主目录，默认为 $WORKSPACE/llvm-release，支持环境变量覆盖。
# 方便用户直接通过指定 LLVM 构建目录来自定义编译环境，无需通过其他间接变量。
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-$WORKSPACE/llvm-release}"


clean_env() {
    # 清理导出的 LLVM 相关路径变量（但不清理用户输入的 LLVM_BUILD_DIR）
    unset LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR LLVM_SYSPATH LLVM_BINARY_DIR
    echo "LLVM 环境变量已成功清理。"
}

prepend_path_once() {
    local dir="$1"
    # 在 PATH 前部插入路径，但避免重复插入导致 PATH 变量臃肿及命令查找冗延。
    case ":$PATH:" in
        *":$dir:"*) ;;
        *) export PATH="$dir:$PATH" ;;
    esac
}

setup_env() {
    # 确保 LLVM_BUILD_DIR 被导出为环境变量，使得后续构建工具和子进程均可正常访问。
    export LLVM_BUILD_DIR="$LLVM_BUILD_DIR"

    # Triton 的 setup.py 通过 LLVM_SYSPATH 来寻找 LLVM 依赖及其头文件与库。
    export LLVM_SYSPATH="$LLVM_BUILD_DIR"
    export LLVM_INCLUDE_DIRS="$LLVM_BUILD_DIR/include"
    export LLVM_LIBRARY_DIR="$LLVM_BUILD_DIR/lib"
    export LLVM_BINARY_DIR="$LLVM_BUILD_DIR/bin"

    # 将 LLVM 的可执行文件 bin 目录加入 PATH，以便编译脚本能直接调用 llvm-config, clang 等工具。
    prepend_path_once "$LLVM_BINARY_DIR"

    echo "========================================="
    echo "triton-anchor LLVM 编译环境已设置完成！"
    echo "WORKSPACE:         $WORKSPACE"
    echo "LLVM_BUILD_DIR:    $LLVM_BUILD_DIR"
    echo "LLVM_SYSPATH:      $LLVM_SYSPATH"
    echo "LLVM_INCLUDE_DIRS: $LLVM_INCLUDE_DIRS"
    echo "LLVM_LIBRARY_DIR:  $LLVM_LIBRARY_DIR"
    echo "LLVM_BINARY_DIR:   $LLVM_BINARY_DIR"
    echo "========================================="
}

status() {
    # 方便用户直观地查看当前的 LLVM 环境变量状态，确保配置正确生效。
    echo "--- 当前 LLVM 环境变量状态 ---"
    echo "WORKSPACE:           ${WORKSPACE:-未设置}"
    echo "LLVM_BUILD_DIR:      ${LLVM_BUILD_DIR:-未设置}"
    echo "LLVM_SYSPATH:        ${LLVM_SYSPATH:-未设置}"
    echo "LLVM_INCLUDE_DIRS:   ${LLVM_INCLUDE_DIRS:-未设置}"
    echo "LLVM_LIBRARY_DIR:    ${LLVM_LIBRARY_DIR:-未设置}"
    echo "LLVM_BINARY_DIR:     ${LLVM_BINARY_DIR:-未设置}"
}

# 提供统一的入口处理参数，支持环境配置、清除以及状态查看功能。
case "${1:-setup}" in
    setup)
        clean_env
        setup_env
        ;;
    clean)
        clean_env
        ;;
    status)
        status
        ;;
    *)
        echo "用法:"
        echo "  source envsetup.sh [setup]          # 配置 LLVM 环境变量 (默认)"
        echo "  source envsetup.sh clean            # 清理 LLVM 环境变量"
        echo "  source envsetup.sh status           # 查看当前 LLVM 环境变量"
        echo ""
        echo "全局环境变量:"
        echo "  WORKSPACE       依赖根目录 (默认: /workspace)"
        echo "  LLVM_BUILD_DIR  LLVM 构建主目录 (默认: /workspace/llvm-release)"
        ;;
esac
