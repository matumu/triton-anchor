# triton-anchor Python 绑定
#
# 此目标被编译进 libtriton.so（TRITON_BUILD_PYTHON_MODULE=ON 时），
# 对外暴露 triton._C.libtriton.anchor 模块。
#
# 对应入口函数：void init_triton_anchor(py::module &&m)

set(TRITON_ANCHOR_BINDINGS_SRC
    triton_anchor_ext.cc
)

# bindings 不独立编译为库，而是将源文件路径暴露给父 CMakeLists.txt，
# 在 libtriton.so 的 add_library 时一并编译。
# 这里用 PARENT_SCOPE 将源文件路径传递给上层。
set(TRITON_ANCHOR_BINDING_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_anchor_ext.cc
    PARENT_SCOPE
)
