# 升级上游 Triton 版本指南

本文档描述如何将 `triton-anchor/triton/` 中 vendor 的上游 Triton 源码升级到新版本。

## 前提知识

triton-anchor 将上游 `triton-lang/triton` 的源码以 vendor 方式放在 `triton/` 子目录中，
并做了**极少量的定制修改**。升级时只需要：

1. 替换 `triton/` 目录为新版上游源码
2. 裁剪掉不需要的部分
3. 重新应用定制修改（仅 2 个文件）

当前基线版本记录在 `triton/TRITON_VERSION` 中。

---

## 步骤一：获取新版上游源码

```bash
# 克隆上游仓库（如果还没有的话）
git clone https://github.com/triton-lang/triton.git /tmp/triton-upstream
cd /tmp/triton-upstream

# 切到目标版本
git checkout <TARGET_COMMIT_HASH>

# 记录 commit 信息，后面更新 TRITON_VERSION 需要
git log -1 --format="%H %ci"
```

---

## 步骤二：替换 triton/ 目录

```bash
cd <triton-anchor 项目根目录>

# 备份当前的 triton/ 目录（可选）
mv triton triton.bak

# 从上游复制需要的目录
mkdir -p triton
cp -r /tmp/triton-upstream/include triton/
cp -r /tmp/triton-upstream/lib triton/
cp -r /tmp/triton-upstream/cmake triton/
mkdir -p triton/python
cp -r /tmp/triton-upstream/python/src triton/python/
cp -r /tmp/triton-upstream/python/triton triton/python/
mkdir -p triton/third_party
cp -r /tmp/triton-upstream/third_party/f2reduce triton/third_party/
```

> **注意**: 只复制上面列出的目录。不要复制 `third_party/nvidia/`、`third_party/amd/`、
> `third_party/proton/` 等 NVIDIA/AMD 专属内容。

---

## 步骤三：裁剪

### 3.1 删除不需要的目录和文件

```bash
# 删除 NVIDIA 专属算子
rm -rf triton/python/triton/ops/

# 删除上游的构建文件（triton-anchor 有自己的顶层 setup.py 和 pyproject.toml）
rm -f triton/python/setup.py
rm -f triton/python/pyproject.toml
rm -f triton/python/MANIFEST.in
rm -rf triton/python/examples/
rm -rf triton/python/test/
rm -rf triton/python/tutorials/

# 删除不需要的 cmake 文件
rm -f triton/cmake/nvidia-toolchain-version.txt
rm -f triton/cmake/pybind11-version.txt
```

### 3.2 从 `__init__.py` 中移除 `ops` 导出

编辑 `triton/python/triton/__init__.py`，从 `__all__` 列表中删除 `"ops"` 一行：

```diff
  __all__ = [
      ...
      "next_power_of_2",
-     "ops",
      "OutOfResources",
      ...
  ]
```

### 3.3 从 CMakeLists.txt 中移除 NVGPUIR 依赖

编辑 `triton/lib/Conversion/TritonGPUToLLVM/CMakeLists.txt`，
从 `LINK_LIBS PUBLIC` 列表末尾删除 `NVGPUIR`：

```diff
      TritonGPUTransforms
      TritonNvidiaGPUTransforms
-     NVGPUIR
  )
```

> **为什么?** `NVGPUIR` 是 NVIDIA 后端的 CMake target，裁掉 `third_party/nvidia/` 后该 target 不存在。
> `TritonNvidiaGPUTransforms` 虽然名字带 nvidia，但它定义在 triton 核心代码中（非 third_party），
> 所以需要保留。如果新版上游移除了该依赖或改了名字，请根据实际编译错误调整。

---

## 步骤四：应用定制修改

需要修改的文件**只有 2 个**。以下给出完整的目标代码。

### 4.1 修改 `python/src/main.cc`

对上游的 `main.cc` 做 3 处修改：

**修改 A: 删除 in-tree 后端宏定义块**

删除 `namespace py = pybind11;` 之后、函数声明之前的**整块宏代码**
（从 `#define FOR_EACH_1` 到 `FOR_EACH_P(DECLARE_BACKEND, TRITON_BACKENDS_TUPLE)` 的所有行）：

```diff
  namespace py = pybind11;

- #define FOR_EACH_1(MACRO, X) MACRO(X)
- #define FOR_EACH_2(MACRO, X, ...) MACRO(X) FOR_EACH_1(MACRO, __VA_ARGS__)
- ...（删除全部 FOR_EACH_* / CONCATENATE / REMOVE_PARENS / DECLARE_BACKEND / INIT_BACKEND 宏定义）...
- FOR_EACH_P(DECLARE_BACKEND, TRITON_BACKENDS_TUPLE)
-
  void init_triton_env_vars(pybind11::module &m);
```

**修改 B: 新增 `init_triton_anchor` 声明，删除宏展开的后端声明**

在函数声明区域，删除 `FOR_EACH_P(DECLARE_BACKEND, ...)` 并添加 `init_triton_anchor`：

```diff
  void init_triton_passes(pybind11::module &&m);
- FOR_EACH_P(DECLARE_BACKEND, TRITON_BACKENDS_TUPLE)
+ void init_triton_anchor(pybind11::module &&m);
```

> 如果上游此处有其他 `init_triton_xxx` 声明，保留它们即可。

**修改 C: 在 `PYBIND11_MODULE` 中添加 anchor 调用，删除后端宏调用**

```diff
  PYBIND11_MODULE(libtriton, m) {
    m.doc() = "Python bindings to the C++ Triton API";
    ...（保留上游已有的 init_triton_xxx 调用）...
    init_triton_llvm(m.def_submodule("llvm"));
+   init_triton_anchor(m.def_submodule("anchor"));
-   FOR_EACH_P(INIT_BACKEND, TRITON_BACKENDS_TUPLE)
  }
```

> **注意**: 保留上游已有的所有 `init_triton_xxx` 调用，只需新增 `init_triton_anchor`
> 并删除 `FOR_EACH_P(INIT_BACKEND, ...)` 一行。

---

### 4.2 修改 `python/triton/backends/__init__.py`

triton-anchor 不使用 in-tree 后端，将文件简化为只保留 entry_points 发现机制。

**修改 A: 删除不需要的 import 和辅助函数**

删除以下内容（triton-anchor 无 in-tree 后端，不需要文件系统扫描逻辑）：

```diff
- import os
- import importlib.util
- import inspect
+ import sys
+ import importlib.metadata
  from dataclasses import dataclass
  from .driver import DriverBase
  from .compiler import BaseBackend
-
-
- def _load_module(name, path):
-     ...
-
-
- def _find_concrete_subclasses(module, base_class):
-     ...
```

**修改 B: 替换 `_discover_backends()` 函数体**

删除 in-tree 的 `os.listdir` 发现循环，替换为 entry_points 发现：

```diff
  def _discover_backends():
      backends = dict()
-     root = os.path.dirname(__file__)
-     for name in os.listdir(root):
-         ...（删除整个 in-tree 发现循环）...
+
+     try:
+         eps = importlib.metadata.entry_points(group="triton.backends")
+     except TypeError:
+         # Python 3.9 兼容
+         eps = importlib.metadata.entry_points().get("triton.backends", [])
+
+     for ep in eps:
+         try:
+             plugin_obj = ep.load()
+             plugin = plugin_obj() if isinstance(plugin_obj, type) else plugin_obj
+
+             compiler_cls = getattr(plugin, 'compiler_cls', None)
+             driver_cls = getattr(plugin, 'driver_cls', None)
+
+             if compiler_cls and driver_cls:
+                 backends[ep.name] = Backend(compiler=compiler_cls, driver=driver_cls)
+         except Exception as e:
+             print(f"Warning: Failed to load out-of-tree backend '{ep.name}': {e}",
+                   file=sys.stderr)
+
      return backends
```

---

## 步骤五：更新版本记录

编辑 `triton/TRITON_VERSION`，更新 commit 和日期：

```
# Vendored from: https://github.com/openai/triton
# Commit: <新的 COMMIT_HASH>
# Date: <YYYY-MM-DD>
# 升级指南见 docs/upgrade_triton.md
```

---

## 步骤六：编译验证

参考 [build.md](build.md) 进行编译和验证。

---

## 附录：定制修改速查表

升级时只需关注以下文件，其余所有文件均可直接从上游复制：

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `python/src/main.cc` | 替换 | 删除后端宏，新增 `init_triton_anchor` |
| `python/triton/backends/__init__.py` | 替换 | 删除 in-tree 发现，替换为 entry_points 发现 |
| `python/triton/__init__.py` | 删一行 | 从 `__all__` 中移除 `"ops"` |
| `lib/Conversion/TritonGPUToLLVM/CMakeLists.txt` | 删一行 | 移除 `NVGPUIR` 依赖 |
| `python/triton/ops/` | 删目录 | 删除 NVIDIA 专属算子 |
| `third_party/nvidia,amd,proton/` | 不复制 | 不需要的上游后端 |


