/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "include/onnx-mlir/Runtime/OMInstrument.h"
#include <chrono>
#include <thread>

int main(int argc, char *argv[]) {
  const char opstart[8] = "TOpStar";
  const char op2[8]     = "TOp2";
  const char op3[8]     = "TOp3";
  const char op4[8]     = "TOp4";
  const char opfinal[8] = "TOpFin";
  OMInstrumentInit();
  OMInstrumentPoint(*(const int64_t*)opstart, 13);
  OMInstrumentPoint(*(const int64_t*)op2, 1);
  OMInstrumentPoint(*(const int64_t*)op3, 4);
  OMInstrumentPoint(*(const int64_t*)op4, 9);
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  OMInstrumentPoint(*(const int64_t*)opfinal, 12);
  return 0;
}
