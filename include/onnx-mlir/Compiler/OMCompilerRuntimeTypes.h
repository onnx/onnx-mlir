/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- OMCompilerRuntimeTypes.h - C/C++ Neutral shared types ------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains types that are shared between the compiler and runtimes
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_OM_COMPILER_RUNTIME_TYPES_H
#define ONNX_MLIR_OM_COMPILER_RUNTIME_TYPES_H

#ifdef __cplusplus
namespace onnx_mlir {
#endif

/* Compiler options to describe instrumentation options */
/* Will be encoded as a 64 bit int */
/* 1st byte: InstrumentActions */
/* 2nd-3rd byte: length of op name */
/* 4rd-5th byte: length of node name */

/* Enum can use the first 8 bits. */
typedef enum {
  InstrumentBeforeOp = 0x0,
  InstrumentAfterOp = 0x1,
  InstrumentReportTime = 0x2,
  InstrumentReportMemory = 0x3,
  InstrumentInit = 0x4,
} InstrumentActions;

/* Definition of setter/getter from a 64 bit unsigned int. Use 64 bit only to
   avoid endianess issues.
   o  Setter assume that that the int has been set to 0
   o  Getter for Enum (Before/After/Time/Memory) return 0 / nonnull
   o  Getter for length return the length of the string
*/

#define INIT_INSTRUMENT(x) (x) = 0ull

/* first byte */
#define SET_INSTRUMENT_BEFORE_OP(x)                                            \
  (x) = (x) | (0x1ull << (unsigned int)InstrumentBeforeOp)
#define CLEAR_INSTRUMENT_BEFORE_OP(x)                                          \
  (x) = (x) & (~(0x1ull << (unsigned int)InstrumentBeforeOp))
#define IS_INSTRUMENT_BEFORE_OP(x)                                             \
  ((x) & (0x1ull << (unsigned int)InstrumentBeforeOp))

#define SET_INSTRUMENT_AFTER_OP(x)                                             \
  (x) = (x) | (0x1ull << (unsigned int)InstrumentAfterOp)
#define CLEAR_INSTRUMENT_AFTER_OP(x)                                           \
  (x) = (x) & (~(0x1ull << (unsigned int)InstrumentAfterOp))
#define IS_INSTRUMENT_AFTER_OP(x)                                              \
  ((x) & (0x1ll << (unsigned int)InstrumentAfterOp))

#define SET_INSTRUMENT_REPORT_TIME(x)                                          \
  (x) = (x) | (0x1ull << (unsigned int)InstrumentReportTime)
#define IS_INSTRUMENT_REPORT_TIME(x)                                           \
  ((x) & (0x1ull << (unsigned int)InstrumentReportTime))

#define SET_INSTRUMENT_REPORT_MEMORY(x)                                        \
  (x) = (x) | (0x1ull << (unsigned int)InstrumentReportMemory)
#define IS_INSTRUMENT_REPORT_MEMORY(x)                                         \
  ((x) & (0x1ull << (unsigned int)InstrumentReportMemory))

#define SET_INSTRUMENT_INIT(x)                                        \
  (x) = (x) | (0x1ull << (unsigned int)InstrumentInit)
#define IS_INSTRUMENT_INIT(x)                                         \
  ((x) & (0x1ull << (unsigned int)InstrumentInit))

/* Second - third byte. */
#define INSTRUMENT_OP_NAME_MASK 0x3Full /* Max 64 chars. */
#define SET_INSTRUMENT_OP_NAME_LEN(x, len)                                     \
  (x) = (x) | ((((long long unsigned int)(len)) & INSTRUMENT_OP_NAME_MASK) << 8)
#define GET_INSTRUMENT_OP_NAME_LEN(x)                                          \
  (((long long unsigned int)(x) >> 8) & INSTRUMENT_OP_NAME_MASK)

/* Forth - fifth byte */
#define INSTRUMENT_NODE_NAME_MASK 0x1FFull /* Max 512 chars. */
#define SET_INSTRUMENT_NODE_NAME_LEN(x, len)                                   \
  (x) = (x) |                                                                  \
        ((((long long unsigned int)(len)) & INSTRUMENT_NODE_NAME_MASK) << 24)
#define GET_INSTRUMENT_NODE_NAME_LEN(x)                                        \
  (((long long unsigned int)(x) >> 24) & INSTRUMENT_NODE_NAME_MASK)

#ifdef __cplusplus
} // namespace onnx_mlir
#endif

#endif /* ONNX_MLIR_OM_COMPILER_RUNTIME_TYPES_H */
