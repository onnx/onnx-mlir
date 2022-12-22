/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- OMInstrument.h - OM Instrument Declaration header ------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of API functions for instrumentation.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_OMINSTRUMENT_H
#define ONNX_MLIR_OMINSTRUMENT_H

#ifdef __cplusplus
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#else
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#endif // #ifdef __cplusplus

#if defined(__APPLE__) || defined(__MVS__)
#include <stdlib.h>
#else
#include <malloc.h>
#endif // #ifdef __APPLE__

#include <onnx-mlir/Compiler/OMCompilerMacros.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize instrument.
 * Initialize counter and read env variables for control
 *
 */
OM_EXTERNAL_VISIBILITY void OMInstrumentInit();

/**
 * Create an instrument point.
 * Measurement of runtime behavior will be measured and output
 * In current implementation, the elapsed time from previous instrument point,
 * and virtual memory size will be reported.
 *
 * @param id for this point. op name is used now.
 * @param tag can used to give extra control of output. Used for begin/end mark
 * now
 * @param nodeName is an onnx_node_name attribute in the graph.
 * @return void
 *
 */
OM_EXTERNAL_VISIBILITY void OMInstrumentPoint(
    const char *opName, int64_t tag, const char *nodeName);

#ifdef __cplusplus
}
#endif

#endif // ONNX_MLIR_OMINSTRUMENT_H
