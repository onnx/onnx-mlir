/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- LineForwardingRawOstream.cpp --------------------===//
//
// Output stream that forwards the data line by line to a sink.
// This can be used to post-process the output of the mlir assembly printer.
//
//===----------------------------------------------------------------------===//

#include "src/Compiler/LineForwardingRawOstream.hpp"

#include <algorithm>

namespace onnx_mlir {

LineForwardingRawOstream::LineForwardingRawOstream(llvm::raw_ostream &out)
    : out(out) {
  SetUnbuffered();
}

LineForwardingRawOstream::~LineForwardingRawOstream() {
  if (!buffer.empty())
    fwd(buffer, out);
}

void LineForwardingRawOstream::write_impl(const char *ptr, size_t size) {
  pos += size;
  const char *end = ptr + size;
  const char *eol = std::find(ptr, end, '\n');
  if (eol != end) {
    buffer.append(ptr, eol + 1);
    fwd(buffer, out);
    buffer.clear();
    ptr = eol + 1;
    while ((eol = std::find(ptr, end, '\n')) != end) {
      fwd(llvm::StringRef(ptr, end - (eol + 1)), out);
      ptr = eol + 1;
    }
  }
  buffer.append(ptr, end);
}

} // namespace onnx_mlir