cmake_minimum_required(VERSION 3.15)
project(MNIST_NPU_Program_DeviceCP)

include(ApolloNPUProgram)

include_directories(${MNIST_NPU_BINARY_DIR})

add_devicecp_firmware(MNIST_NPU_Program_DeviceCP
  CPPFILES
  MnistClusterCommand.cpp
)
