cmake_minimum_required(VERSION 3.15)
project(MNIST_NPU_Program_TVP)

include(ApolloNPUProgram)

set_source_files_properties(${MNIST_NPU_BINARY_DIR}/tcp_driver.tvp.cpp PROPERTIES GENERATED true)
set_source_files_properties(${MNIST_NPU_BINARY_DIR}/mnist_padded_static_noconsts_2d_bias.onnx.s PROPERTIES GENERATED true)

add_tvp_firmware(MNIST_NPU_Program_TVP
  CPPFILES
  ${MNIST_NPU_BINARY_DIR}/tcp_driver.tvp.cpp
  ${MNIST_NPU_BINARY_DIR}/mnist_padded_static_noconsts_2d_bias.onnx.s
)
