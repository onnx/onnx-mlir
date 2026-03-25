// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <mlir/IR/PatternMatch.h>

namespace onnx_mlir {

class ResultNamesUpdater : public mlir::RewriterBase::Listener {
public:
  void notifyOperationReplaced(
      mlir::Operation *op, mlir::Operation *replacement) override;

  void notifyOperationReplaced(
      mlir::Operation *op, mlir::ValueRange replacement) override;
};

} // namespace onnx_mlir
