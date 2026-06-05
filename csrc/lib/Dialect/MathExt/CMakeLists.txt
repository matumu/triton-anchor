#===------------------------------------------------------------------------===#
#
# mlir-ext MathExt 方言库
#
# 来自 flir/lib/Dialect/MathExt/IR/，经过简化：
# - 移除 ub::PoisonOp 依赖（MLIRUBDialect 不在 llvm21 工具链中）
# - materializeConstant 仅保留 arith::ConstantOp 的实现
#
#===------------------------------------------------------------------------===#

add_triton_library(MLIRMathExtDialect
  MathExtDialect.cpp
  MathExtOps.cpp

  DEPENDS
  MLIRMathExtDialectIncGen
  MLIRMathExtOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
)
