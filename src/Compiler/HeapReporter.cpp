/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- HeapReporter.cpp --------------------------===//
//
// Reports heap usage before and after compiler passes.
//
//===----------------------------------------------------------------------===//

#include "src/Compiler/HeapReporter.hpp"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

#if defined(__APPLE__)
#include <unistd.h> // Unsupported on MSVC.
#endif

using namespace mlir;

namespace onnx_mlir {

namespace {
void splitToSet(StringRef commaSeparated, llvm::StringSet<> &set) {
  SmallVector<StringRef> splits;
  commaSeparated.split(splits, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef s : splits)
    set.insert(s.trim());
}

#if defined(__APPLE__)
void logMessage(StringRef logFilename, StringRef msg,
    llvm::sys::fs::OpenFlags extraFlags = llvm::sys::fs::OF_None) {
  std::error_code EC;
  llvm::raw_fd_ostream os(logFilename, EC, llvm::sys::fs::OF_Text | extraFlags);
  if (EC) {
    llvm::errs() << "Error: '" << EC.message() << "' opening heap report file '"
                 << logFilename << "'\n";
    exit(1);
  }
  os << msg;
}
#endif
} // namespace

HeapReporter::HeapReporter(
    std::string logFilename, StringRef beforePasses, StringRef afterPasses)
    : logFilename(logFilename) {
  splitToSet(beforePasses, this->beforePassesSet);
  splitToSet(afterPasses, this->afterPassesSet);
  reportBegin("onnx-mlir heap report"
              "\n--report-heap-before='" +
              beforePasses.str() + "'\n--report-heap-after='" +
              afterPasses.str() + "'");
}

HeapReporter::~HeapReporter() {}

void HeapReporter::runBeforePass(mlir::Pass *pass, mlir::Operation *op) {
  StringRef name = pass->getArgument();
  if (beforePassesSet.contains(name))
    reportHeap("BEFORE PASS " + name.str());
}

void HeapReporter::runAfterPass(mlir::Pass *pass, mlir::Operation *op) {
  StringRef name = pass->getArgument();
  if (afterPassesSet.contains(name))
    reportHeap("AFTER PASS " + name.str());
}

#if defined(__APPLE__)
void HeapReporter::reportBegin(const std::string &heading) {
  if (!getenv("MallocStackLogging")) {
    llvm::errs() << "Error: Environment variable MallocStackLogging must be "
                    "set to report heap usage.\n"
                    "See utils/onnx-mlir-report-heap.sh\n";
    exit(1);
  }
  // Capture the first 40 lines of heap output, which include the top level
  // numbers and a handful of the largest allocation classes.
  command =
      "heap -s " + std::to_string(getpid()) + " | head -n40 >> " + logFilename;
  logMessage(logFilename,
      heading + "\nusing heap report command: '" + command + "'\n");
}

void HeapReporter::reportHeap(const std::string &heading) {
  logMessage(logFilename, "\n" + heading + ":\n", llvm::sys::fs::OF_Append);
  std::system(command.c_str());
}
#else
// TODO: Support heap reporting on more operating systems.

void HeapReporter::reportBegin(const std::string &heading) {
  llvm_unreachable("report-heap is not supported for this OS currently");
}

void HeapReporter::reportHeap(const std::string &heading) {
  llvm_unreachable("report-heap is not supported for this OS currently");
}
#endif

} // namespace onnx_mlir
