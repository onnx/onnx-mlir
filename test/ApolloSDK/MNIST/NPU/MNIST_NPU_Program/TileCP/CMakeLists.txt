cmake_minimum_required(VERSION 3.15)
project(MNIST_NPU_Program_TileCP)

include(ApolloNPUProgram)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../TVP)

set_source_files_properties(${MNIST_NPU_BINARY_DIR}/tcp_driver.tcp.cpp PROPERTIES GENERATED true)

add_tilecp_firmware(MNIST_NPU_Program_TileCP
  CPPFILES
  ${MNIST_NPU_BINARY_DIR}/tcp_driver.tcp.cpp
)
