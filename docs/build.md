# triton-anchor 构建与环境配置指南

本文档介绍了如何从零开始配置开发环境，并构建 `triton-anchor` 及其底层的相关依赖。由于 Triton 编译栈涉及大量的 C++ 和 LLVM 依赖，强烈建议在 Docker 环境中进行构建。

## 1. 基础容器与系统依赖 (Docker & APT)

推荐使用 Ubuntu 24.04 作为基础系统镜像，并在启动容器时挂载你的代码目录：

```bash
# 后台启动基础镜像（可根据需要挂载硬件特定的驱动目录，例如 /opt/tpuv7）
docker run --privileged -itd --name triton-anchor-dev \
    -v $PWD/triton:/triton \
    ubuntu:24.04

# 进入容器
docker exec -it triton-anchor-dev bash

# 更新包索引
apt-get update

# 安装基础编译工具链和依赖
apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libz-dev \
    libzstd-dev \
    libxml2-dev
```

## 2. Python 虚拟环境与包管理器

为避免污染系统环境，推荐使用 [uv](https://github.com/astral-sh/uv) 管理 Python 虚拟环境，它比标准的 `pip` 快很多，且能够有效处理复杂的依赖树。

```bash
# 安装 uv
pip3 install uv --break-system-packages -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建并激活虚拟环境
uv venv /opt/venv
source /opt/venv/bin/activate
```

## 3. 安装 triton-anchor

`triton-anchor` 自身是一个轻量级的 Python 前端，主要负责分发 TTIR 和 AnchorIR 规范。通过 `uv` 可以在隔离模式下极速安装：

```bash
# 假设你已经克隆了代码到 /triton/triton-anchor
cd /triton/triton-anchor

# 1. 直接安装（开发模式）
uv pip install [--no-build-isolation] -e .

# 2. 如果需要运行单元测试，安装 dev 依赖
uv pip install -e ".[dev]"

# 3. 如果需要构建分发包 (wheel / sdist)
uv build --wheel [--no-build-isolation]
```

## 4. 构建与集成硬件后端 (Out-of-Tree)

`triton-anchor` 安装完毕后，需要配合具体的硬件后端（如 Sophgo TPU、SpacemiT AME）才能完成端到端的编译。

### 以外挂 Sophgo TPU 后端为例：

由于硬件后端通常包含 C++ 代码（如自定义 pass、驱动封装），并依赖特定版本的 LLVM，你需要：

1. **准备 LLVM 等工具链**：
   下载并解压硬件对应的 LLVM release 包：
   ```bash
   tar -xzf llvm-release-***.tar.gz -C /triton/
   ```

2. **配置环境变量**：
   通过 source 项目的 `envsetup.sh` 或手动设置路径：
   ```bash
   export LLVM_SYSPATH=/triton/llvm-release
   ```

3. **构建硬件后端包**：
   ```bash
   cd /triton/triton-all-backends/sophgo
   uv build --wheel --no-build-isolation
   uv pip install --no-build-isolation .
   ```
   > **注意**：构建外挂后端时，通常**必须**添加 `--no-build-isolation` 参数。这样 CMake 在构建时才能从当前环境的 `triton-anchor` 和 `triton` 包中正确获取到头文件和库的路径。

## 5. 验证安装

完成构建后，Triton 会通过 `entry_points` 自动发现已安装的后端。

```bash
# 验证 Python 包是否安装
python3 -c "import triton_anchor; print('triton-anchor loaded')"

# 验证底层硬件后端是否被 Triton 自动发现
python3 -c "from triton.backends import backends; print('sophgo' in backends)"
```
