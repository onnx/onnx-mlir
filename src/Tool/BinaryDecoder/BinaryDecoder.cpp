#include <fstream>
#include <iostream>
#include <string>

#include <llvm/Support/CommandLine.h>

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

enum OnnxDataType {
  UNDEFINED = 0,
  // Basic types.
  FLOAT = 1,  // float
  UINT8 = 2,  // uint8_t
  INT8 = 3,   // int8_t
  UINT16 = 4, // uint16_t
  INT16 = 5,  // int16_t
  INT32 = 6,  // int32_t
  INT64 = 7,  // int64_t
  STRING = 8, // string
  BOOL = 9,   // bool

  // IEEE754 half-precision floating-point format (16 bits wide).
  // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
  FLOAT16 = 10,

  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  COMPLEX64 = 14,  // complex with float32 real and imaginary components
  COMPLEX128 = 15, // complex with float64 real and imaginary components

  // Non-IEEE floating-point format based on IEEE754 single-precision
  // floating-point number truncated to 16 bits.
  // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
  BFLOAT16 = 16,

  // Future extensions go here.
};

llvm::cl::opt<OnnxDataType> DataType(
    llvm::cl::desc("Choose data type to decode:"),
    llvm::cl::values(clEnumVal(FLOAT, "FLOAT"), clEnumVal(UINT8, "UINT8"),
        clEnumVal(INT8, "INT8"), clEnumVal(UINT16, "UINT16"),
        clEnumVal(INT16, "INT16"), clEnumVal(INT32, "INT32"),
        clEnumVal(INT64, "INT64"), clEnumVal(STRING, "STRING"),
        clEnumVal(BOOL, "BOOL"), clEnumVal(FLOAT16, "FLOAT16"),
        clEnumVal(DOUBLE, "DOUBLE"), clEnumVal(UINT32, "UINT32"),
        clEnumVal(UINT64, "UINT64")));

template <typename T>
int printBuffer(std::vector<char> buffer) {
  auto *ptr = (T *)&buffer[0];
  auto data = std::vector<T>(ptr, ptr + buffer.size() / sizeof(T));
  for (const auto &elem : data)
    std::cout << elem << " ";
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
  if (DataType == ONNX_TYPE)                                                   \
    return printBuffer<CPP_TYPE>(buffer);

  PRINT_BUFFER_FOR_TYPE(UINT8, u_int8_t);
  PRINT_BUFFER_FOR_TYPE(UINT16, u_int16_t);
  PRINT_BUFFER_FOR_TYPE(INT16, int16_t);
  PRINT_BUFFER_FOR_TYPE(INT32, int32_t);
  PRINT_BUFFER_FOR_TYPE(INT64, int64_t);

  PRINT_BUFFER_FOR_TYPE(FLOAT, float);
  PRINT_BUFFER_FOR_TYPE(DOUBLE, double);
  PRINT_BUFFER_FOR_TYPE(UINT32, u_int32_t);
  PRINT_BUFFER_FOR_TYPE(UINT64, u_int64_t);
}