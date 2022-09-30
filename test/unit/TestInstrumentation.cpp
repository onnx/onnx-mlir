/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "include/onnx-mlir/Runtime/OMInstrument.h"
#include <chrono>
#include <thread>

int main(int argc, char *argv[]) {
  const char opstart[8] = "TOpStar";
  const char op2[8] = "TOp2";
  const char op3[8] = "TOp3";
  const char op4[8] = "TOp4";
  const char opfinal[8] = "TOpFin";
  const char nodeStart[8] = "NodeSta";
  const char nodeOp2[8] = "Node2";
  const char nodeOp3[8] = "Node3";
  const char nodeOp4[8] = "Node4";
  const char nodeOpfinal[8] = "NodeFin";
  OMInstrumentInit();
  OMInstrumentPoint(*(const int64_t *)opstart, 13, nodeStart);
  OMInstrumentPoint(*(const int64_t *)op2, 1, nodeOp2);
  OMInstrumentPoint(*(const int64_t *)op3, 4, nodeOp3);
  OMInstrumentPoint(*(const int64_t *)op4, 9, nodeOp4);
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  OMInstrumentPoint(*(const int64_t *)opfinal, 12, nodeOpfinal);
  return 0;
}
