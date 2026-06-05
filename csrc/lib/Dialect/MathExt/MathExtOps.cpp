#include "mlir-ext/Dialect/MathExt/IR/MathExt.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include <optional>

using namespace mlir;
using namespace mlir::mathext;

#define GET_OP_CLASSES
#include "mlir-ext/Dialect/MathExt/IR/MathExtOps.cpp.inc"

//===----------------------------------------------------------------------===//
// FModOp folder
//===----------------------------------------------------------------------===//

OpFoldResult mathext::FModOp::fold(FoldAdaptor adaptor) {
  return constFoldBinaryOp<FloatAttr>(adaptor.getOperands(),
                                     [](const APFloat &a, const APFloat &b) {
                                       APFloat result(a);
                                       (void)result.mod(b);
                                       return result;
                                     });
}

//===----------------------------------------------------------------------===//
// DivRzOp folder
//===----------------------------------------------------------------------===//

OpFoldResult mathext::DivRzOp::fold(FoldAdaptor adaptor) {
  return constFoldBinaryOp<FloatAttr>(adaptor.getOperands(),
                                     [](const APFloat &a, const APFloat &b) {
                                       APFloat result(a);
                                       result.divide(b, APFloat::rmTowardZero);
                                       return result;
                                     });
}

/// Materialize an integer or floating point constant.
/// triton-anchor: 不依赖 ub::PoisonOp（需要 MLIRUBDialect，不在此工具链中），
/// 仅支持 arith::ConstantOp 的常量物化。
Operation *mathext::MathExtDialect::materializeConstant(OpBuilder &builder,
                                                        Attribute value,
                                                        Type type,
                                                        Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}