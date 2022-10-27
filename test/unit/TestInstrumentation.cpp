/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "include/onnx-mlir/Runtime/OMInstrument.h"
#include <chrono>
#include <thread>

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
  OMInstrumentInit();
  OMInstrumentPoint(opstart.c_str(), 13, nodeStart.c_str());
  OMInstrumentPoint(op2.c_str(), 1, nodeOp2.c_str());
  OMInstrumentPoint(op3.c_str(), 4, nodeOp3.c_str());
  OMInstrumentPoint(op4.c_str(), 9, nodeOp4.c_str());
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  OMInstrumentPoint(opfinal.c_str(), 12, nodeOpfinal.c_str());
  return 0;
}
