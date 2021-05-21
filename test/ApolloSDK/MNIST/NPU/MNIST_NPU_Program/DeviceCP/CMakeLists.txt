cmake_minimum_required(VERSION 3.15)
project(MNIST_NPU_Program_DeviceCP)

include(ApolloNPUProgram)

set_source_files_properties(${MNIST_NPU_BINARY_DIR}/tcp_driver.dcp.cpp PROPERTIES GENERATED true)

add_devicecp_firmware(MNIST_NPU_Program_DeviceCP
  CPPFILES
  ${MNIST_NPU_BINARY_DIR}/tcp_driver.dcp.cpp
)
