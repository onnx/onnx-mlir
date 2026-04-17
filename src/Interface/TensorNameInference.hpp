// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <memory>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpDefinition.h>

namespace onnx_mlir {

/// Base class that represents Transformations applied on an original ONNX
/// Tensor to obtain respective tensor in the modified graph
class Transform {
public:
  /// Enum to allow LLVM-style casting with isa<>, dyn_cast<> etc.
  enum class Kind {
    Reshape,
    Transpose,
    Pad,
    Slice,
    Dequantize,
    Quantize,
    List,
  };

  Transform(
      Kind k, mlir::ArrayRef<int64_t> inShape, mlir::ArrayRef<int64_t> outShape)
      : kind(k), inShape(inShape), outShape(outShape) {}

  // Attribute conversion
  [[nodiscard]] virtual mlir::Attribute toAttr(
      mlir::MLIRContext *context) const = 0;

  /// Creates a new transform which is inversion of the current transform
  [[nodiscard]] virtual std::unique_ptr<Transform> invert() const = 0;

  virtual ~Transform() = default;

  [[nodiscard]] Kind getKind() const { return kind; };

  [[nodiscard]] mlir::ArrayRef<int64_t> getInShape() const { return inShape; }
  [[nodiscard]] mlir::ArrayRef<int64_t> getOutShape() const { return outShape; }

private:
  const Kind kind;

protected:
  mlir::SmallVector<int64_t> inShape;
  mlir::SmallVector<int64_t> outShape;
};

} // namespace onnx_mlir

#include "src/Interface/TensorNameInference.hpp.inc"
