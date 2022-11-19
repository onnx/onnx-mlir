/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- LineForwardingRawOstream.hpp --------------------===//
//
// Output stream that forwards the data line by line to a sink.
// This can be used to post-process the output of the mlir assembly printer.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <string>

namespace onnx_mlir {

class LineForwardingRawOstream : public llvm::raw_ostream {
public:
  using LineForwarder =
      std::function<void(llvm::StringRef, llvm::raw_ostream &)>;

  explicit LineForwardingRawOstream(llvm::raw_ostream &out);
  ~LineForwardingRawOstream() override;

  void setForwarder(LineForwarder fwd) { this->fwd = std::move(fwd); }

  llvm::raw_ostream &os() { return *this; }

private:
  void write_impl(const char *ptr, size_t size) override;

  uint64_t current_pos() const override { return pos; }

  llvm::raw_ostream &out;
  LineForwarder fwd;
  std::string buffer;
  uint64_t pos = 0;
};

} // namespace onnx_mlir