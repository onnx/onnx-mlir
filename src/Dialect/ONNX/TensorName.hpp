// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

/// This files defines the TensorName structure.
/// TensorName tracks the original name of a tensor along with a sequence of
/// transformations (reshape, transpose, pad, slice) that have been applied to
/// it.
/// Then TensorName API abstracts the underlying storage mechanism (attributes
/// on operations).

#include <cstdint>
#include <memory>
#include <string>

#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

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

  Transform(Kind k, mlir::ArrayRef<int64_t> inShape,
      mlir::ArrayRef<int64_t> outShape);

  // Attribute conversions
  static std::unique_ptr<Transform> fromAttr(mlir::ArrayAttr arrayAttr);
  [[nodiscard]] virtual mlir::Attribute toAttr(
      mlir::MLIRContext *context) const = 0;

  // Op conversions
  static std::unique_ptr<Transform> fromOp(mlir::Operation *op);

  /// Creates a new transform which is inversion of the current transform
  [[nodiscard]] virtual std::unique_ptr<Transform> invert() const = 0;

  virtual ~Transform() = default;

  /// Utility methods to be used by sub-classes
  static mlir::SmallVector<int64_t> arrayToVector(mlir::ArrayAttr arrayAttr);
  static mlir::SmallVector<int64_t> denseToVector(
      mlir::DenseIntElementsAttr denseAttr);
  static mlir::SmallVector<int64_t> valToVector(mlir::Value val);
  static mlir::SmallVector<int64_t> axesToVector(mlir::Value val, size_t rank);
  static mlir::ArrayAttr vecToAttr(
      mlir::MLIRContext *context, mlir::ArrayRef<int64_t> vector);

  [[nodiscard]] Kind getKind() const { return kind; };

  [[nodiscard]] mlir::ArrayRef<int64_t> getInShape() const { return inShape; }
  [[nodiscard]] mlir::ArrayRef<int64_t> getOutShape() const { return outShape; }

private:
  const Kind kind;

protected:
  friend class TensorName;

  mlir::SmallVector<int64_t> inShape;
  mlir::SmallVector<int64_t> outShape;
};

class ReshapeTransform : public Transform {
public:
  ReshapeTransform(
      mlir::ArrayRef<int64_t> inShape, mlir::ArrayRef<int64_t> outShape);

  ReshapeTransform(mlir::ArrayAttr attr);
  [[nodiscard]] mlir::Attribute toAttr(
      mlir::MLIRContext *context) const override;

  [[nodiscard]] std::unique_ptr<Transform> invert() const override;

  static bool classof(const Transform *transform) {
    return transform->getKind() == Kind::Reshape;
  }
};

class TransposeTransform : public Transform {
public:
  TransposeTransform(mlir::ArrayRef<int64_t> inShape,
      mlir::ArrayRef<int64_t> perm, mlir::ArrayRef<int64_t> outShape);

  TransposeTransform(mlir::ArrayAttr attr);
  [[nodiscard]] mlir::Attribute toAttr(
      mlir::MLIRContext *context) const override;

  [[nodiscard]] std::unique_ptr<Transform> invert() const override;

  static bool classof(const Transform *transform) {
    return transform->getKind() == Kind::Transpose;
  }

  [[nodiscard]] mlir::ArrayRef<int64_t> getPerm() const { return perm; }

private:
  mlir::SmallVector<int64_t> perm;
};

class PadTransform : public Transform {
public:
  PadTransform(mlir::ArrayRef<int64_t> inShape, mlir::ArrayRef<int64_t> starts,
      mlir::ArrayRef<int64_t> ends, mlir::ArrayRef<int64_t> axes,
      mlir::Attribute constant, mlir::ArrayRef<int64_t> outShape);

  PadTransform(mlir::ArrayAttr attr);
  [[nodiscard]] mlir::Attribute toAttr(
      mlir::MLIRContext *context) const override;

  [[nodiscard]] std::unique_ptr<Transform> invert() const override;

  static bool classof(const Transform *transform) {
    return transform->getKind() == Kind::Pad;
  }

  [[nodiscard]] mlir::ArrayRef<int64_t> getStarts() const { return starts; }
  [[nodiscard]] mlir::ArrayRef<int64_t> getEnds() const { return ends; }
  [[nodiscard]] mlir::ArrayRef<int64_t> getAxes() const { return axes; }

private:
  mlir::SmallVector<int64_t> starts;
  mlir::SmallVector<int64_t> ends;
  mlir::SmallVector<int64_t> axes;
  mlir::Attribute constant;
};

class SliceTransform : public Transform {
public:
  SliceTransform(mlir::ArrayRef<int64_t> inShape,
      mlir::ArrayRef<int64_t> starts, mlir::ArrayRef<int64_t> ends,
      mlir::ArrayRef<int64_t> axes, mlir::ArrayRef<int64_t> outShape);

  SliceTransform(mlir::ArrayAttr attr);
  [[nodiscard]] mlir::Attribute toAttr(
      mlir::MLIRContext *context) const override;

  [[nodiscard]] std::unique_ptr<Transform> invert() const override;

  static bool classof(const Transform *transform) {
    return transform->getKind() == Kind::Slice;
  }

  [[nodiscard]] mlir::ArrayRef<int64_t> getStarts() const { return starts; }
  [[nodiscard]] mlir::ArrayRef<int64_t> getEnds() const { return ends; }
  [[nodiscard]] mlir::ArrayRef<int64_t> getAxes() const { return axes; }

private:
  mlir::SmallVector<int64_t> starts;
  mlir::SmallVector<int64_t> ends;
  mlir::SmallVector<int64_t> axes;
};

/// A convenience transform to hold multiple transforms
class ListTransform : public Transform {
public:
  ListTransform(mlir::SmallVector<std::unique_ptr<Transform>> &&transforms);

  mlir::Attribute toAttr(mlir::MLIRContext *context) const override;

  [[nodiscard]] std::unique_ptr<Transform> invert() const override;

  static bool classof(const Transform *transform) {
    return transform->getKind() == Kind::List;
  }

private:
  friend class TensorName;
  mlir::SmallVector<std::unique_ptr<Transform>> transforms;
};

/// A TensorName represents the name of a tensor along with the sequence of
/// transformations that have been applied to it.
class TensorName {
public:
  TensorName(std::string name);

  /// Constructs TensorName from MLIR value by extracting information from
  /// the "ResultNames" attribute or the function argument attribute.
  /// Does NOT `infer` the transforms.
  explicit TensorName(mlir::Value value);

  /// Infer the TensorName based on transforms
  static TensorName infer(mlir::Value value);

  /// True if TensorName corresponds to a valid tensor in ONNX
  explicit operator bool() const { return !name.empty(); }

  /// Get original name of the tensor
  [[nodiscard]] mlir::StringRef getNameStr() const { return name; }

  /// List of transforms to be applied
  [[nodiscard]] mlir::SmallVector<Transform *> getTransforms() const {
    return llvm::map_to_vector(
        transforms, [](const std::unique_ptr<Transform> &uptr) -> Transform * {
          return uptr.get();
        });
  }

  /// Add new transform at the end of list of tranforms
  void push_back(std::unique_ptr<Transform> transform);

  /// Attribute conversion
  [[nodiscard]] mlir::Attribute toAttr(mlir::MLIRContext *context) const;

  /// Set the TensorName to value by setting attribute to defining op
  mlir::LogicalResult setTo(mlir::Value value) const;

  static TensorName inferWithUse(mlir::Value value);
  static TensorName inferWithDef(mlir::Value value);

private:
  /// The original name of this tensor (e.g. the edge name in the ONNX model).
  std::string name;

  /// The list of transformations that need to be applied on top of the
  /// original tensor.
  mlir::SmallVector<std::unique_ptr<Transform>> transforms;
};

} // namespace onnx_mlir
