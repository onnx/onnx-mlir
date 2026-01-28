/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "include/onnx-mlir/Runtime/OMInstrument.h"
#include <chrono>
#include <thread>

// Keep this in sync with
// onnx-mlir/include/onnx-mlir/Compiler/OMCompilerRuntimeTypes.h
#define TAG(tag, op, node)                                                     \
  (((unsigned long long)(tag)) | (((unsigned long long)op.length()) << 8ull) | \
      (((unsigned long long)node.length()) << 24ull))
#define BEFORE 0x1ull
#define AFTER 0x2ull
#define TIME 0x4ull
#define MEM 0x8ull
#define INIT 0x10ull

int main(int argc, char *argv[]) {
  const std::string opstart = "TDStar.TOpStar";
  const std::string op2 = "TD2.TOp2";
  const std::string op3 = "TD3.TOp3";
  const std::string op4 = "TD4.TOp4";
  const std::string opfinal = "TDFin.TOpFin";
  const std::string nodeStart = "NodeSta";
  const std::string nodeOp2 = "Node2";
  const std::string nodeOp3 = "Node3";
  const std::string nodeOp4 = "Node4";
  const std::string nodeOpfinal = "NodeFin";
  // Time before / after
  OMInstrumentPoint(opstart.c_str(),
      TAG(INIT | TIME | BEFORE, opstart, nodeStart), nodeStart.c_str());
  OMInstrumentPoint(opstart.c_str(), TAG(TIME | AFTER, opstart, nodeStart),
      nodeStart.c_str());
  OMInstrumentPoint(
      op2.c_str(), TAG(MEM | BEFORE, op2, nodeOp2), nodeOp2.c_str());
  OMInstrumentPoint(
      op3.c_str(), TAG(MEM | AFTER, op3, nodeOp3), nodeOp3.c_str());
  OMInstrumentPoint(
      op4.c_str(), TAG(TIME | BEFORE, op4, nodeOp4), nodeOp4.c_str());
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  OMInstrumentPoint(
      op4.c_str(), TAG(TIME | AFTER, op4, nodeOp4), nodeOp4.c_str());
  OMInstrumentPoint(opfinal.c_str(), TAG(MEM | AFTER, opfinal, nodeOpfinal),
      nodeOpfinal.c_str());
  omInstrumentPrint();
  return 0;
}
