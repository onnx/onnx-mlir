cmake_minimum_required(VERSION 3.15)
project(MNIST_NPU_Program)

include(ApolloNPUProgram)

add_devicecp_build(MNIST_NPU_Program_DeviceCP
  DIR DeviceCP
  FIRMWARE MNIST_NPU_Program_DeviceCP
  CMAKE_GENERATE_PARAMS -DMNIST_NPU_BINARY_DIR=${MNIST_BINARY_DIR}
)

add_dependencies(MNIST_NPU_Program_DeviceCP maia)

add_tilecp_build(MNIST_NPU_Program_TileCP
  DIR TileCP
  FIRMWARE MNIST_NPU_Program_TileCP
  CMAKE_GENERATE_PARAMS -DMNIST_NPU_BINARY_DIR=${MNIST_BINARY_DIR}
)

add_dependencies(MNIST_NPU_Program_TileCP maia)

add_tvp_build(MNIST_NPU_Program_TVP
  DIR TVP
  FIRMWARE MNIST_NPU_Program_TVP
  CMAKE_GENERATE_PARAMS -DMNIST_NPU_BINARY_DIR=${MNIST_BINARY_DIR}
)

add_dependencies(MNIST_NPU_Program_TVP maia)

add_npu_program(MNIST_NPU_Program
  MANIFEST
  ${CMAKE_CURRENT_SOURCE_DIR}/manifest.json
  FIRMWARE
  ${CMAKE_CURRENT_BINARY_DIR}/DeviceCP/MNIST_NPU_Program_DeviceCP
  ${CMAKE_CURRENT_BINARY_DIR}/TileCP/MNIST_NPU_Program_TileCP
  ${CMAKE_CURRENT_BINARY_DIR}/TVP/MNIST_NPU_Program_TVP
)

add_dependencies(MNIST_NPU_Program MNIST_NPU_Program_TVP MNIST_NPU_Program_TileCP MNIST_NPU_Program_DeviceCP)

add_custom_command(TARGET MNIST_NPU_Program POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/MNIST_NPU_Program.zip ${MNIST_BINARY_DIR}
  COMMENT "Copying MNIST_NPU_Program.zip"
)