/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- BinaryDecoder.cpp - Decode binary files into typed arrays ------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementation of a utility called BinaryDecoder, which
// decodes a sequence of binary data within a binary file specified by an
// offset and a length into a typed array and print to stdout.
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <iostream>
#include <string>

#include "onnx/onnx_pb.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"

#if defined(_WIN32)

#include <stdint.h>

typedef uint8_t u_int8_t;
typedef uint16_t u_int16_t;
typedef uint32_t u_int32_t;
typedef uint64_t u_int64_t;

#endif

llvm::cl::opt<std::string> Filename(
    llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::Required);
llvm::cl::opt<int64_t> Start("s",
    llvm::cl::desc("Specify the index of the starting byte"),
    llvm::cl::value_desc("start"), llvm::cl::Required);
llvm::cl::opt<int64_t> Size("n",
    llvm::cl::desc("Specify the number of bytes of data to decode"),
    llvm::cl::value_desc("size"), llvm::cl::Required);
llvm::cl::opt<bool> Remove(
    "rm", llvm::cl::desc(
              "Whether to remove the file being decoded after inspection."));

llvm::cl::opt<onnx::TensorProto::DataType> DataType(
    llvm::cl::desc("Choose data type to decode:"),
    llvm::cl::values(clEnumVal(onnx::TensorProto::FLOAT, "FLOAT"),
        clEnumVal(onnx::TensorProto::UINT8, "UINT8"),
        clEnumVal(onnx::TensorProto::INT8, "INT8"),
        clEnumVal(onnx::TensorProto::UINT16, "UINT16"),
        clEnumVal(onnx::TensorProto::INT16, "INT16"),
        clEnumVal(onnx::TensorProto::INT32, "INT32"),
        clEnumVal(onnx::TensorProto::INT64, "INT64"),
        clEnumVal(onnx::TensorProto::STRING, "STRING"),
        clEnumVal(onnx::TensorProto::BOOL, "BOOL"),
        clEnumVal(onnx::TensorProto::FLOAT16, "FLOAT16"),
        clEnumVal(onnx::TensorProto::DOUBLE, "DOUBLE"),
        clEnumVal(onnx::TensorProto::UINT32, "UINT32"),
        clEnumVal(onnx::TensorProto::UINT64, "UINT64")));

template <typename T>
int printBuffer(std::vector<char> buffer) {
  auto *ptr = reinterpret_cast<T *>(&buffer[0]);
  auto data = std::vector<T>(ptr, ptr + buffer.size() / sizeof(T));
  for (const auto &elem : data)
    std::cout << elem << " ";
  std::cout << std::endl;
  return 0;
}

template <>
int printBuffer<bool>(std::vector<char> buffer) {
  const char *rawData = buffer.data();
  for (unsigned i = 0; i < buffer.size() * 8; i++) {
    bool b = (rawData[i / CHAR_BIT] & (1 << (i % CHAR_BIT))) != 0;
    printf("%d", b);
    if ((i + 1) % 8 == 0)
      std::cout << " ";
  }
  std::cout << std::endl;
  return 0;
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  std::vector<char> buffer(Size);
  std::ifstream file(Filename, std::ios::in | std::ios::binary);
  if (!file)
    return -1;
  file.seekg(Start, file.beg);
  file.read(&buffer[0], Size);
  file.close();

  if (Remove)
    llvm::sys::fs::remove(Filename);

#define PRINT_BUFFER_FOR_TYPE(ONNX_TYPE, CPP_TYPE)                             \
  if (DataType == (ONNX_TYPE))                                                 \
    return printBuffer<CPP_TYPE>(buffer);

  PRINT_BUFFER_FOR_TYPE(onnx::TensorProto::BOOL, bool);
  PRINT_BUFFER_FOR_TYPE(onnx::TensorProto::UINT8, u_int8_t);
  PRINT_BUFFER_FOR_TYPE(onnx::TensorProto::UINT16, u_int16_t);
  PRINT_BUFFER_FOR_TYPE(onnx::TensorProto::INT16, int16_t);
  PRINT_BUFFER_FOR_TYPE(onnx::TensorProto::INT32, int32_t);
  PRINT_BUFFER_FOR_TYPE(onnx::TensorProto::INT64, int64_t);

  PRINT_BUFFER_FOR_TYPE(onnx::TensorProto::FLOAT, float);
  PRINT_BUFFER_FOR_TYPE(onnx::TensorProto::DOUBLE, double);
  PRINT_BUFFER_FOR_TYPE(onnx::TensorProto::UINT32, u_int32_t);
  PRINT_BUFFER_FOR_TYPE(onnx::TensorProto::UINT64, u_int64_t);
}
