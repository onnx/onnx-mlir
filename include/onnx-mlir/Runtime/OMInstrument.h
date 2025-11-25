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
 * Print the instrumentation information gathered during a single inference run.
 * If no information was recorded (e.g. the -profile-ir or -profile-ir-and-sig
 * were not used), this call does nothing. Printing goes on standard out or in
 * the file pointed to by the ONNX_MLIR_INSTRUMENT_FILE variable. In rare cases,
 * the instrumentation buffer may overflow, in which case printing may occur
 * prior to this call; nevertheless a call to the function below will ensure
 * that the entire information is printed out.
 *
 */
OM_EXTERNAL_VISIBILITY void omInstrumentPrint();

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
