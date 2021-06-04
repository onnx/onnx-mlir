cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_GENERATOR MATCHES "Visual Studio")
  message(WARNING "Apollo MNIST testing does not support Visual Studio; please use ninja.")
elseif(NOT WIN32)
  message(WARNING "Apollo MNIST testing does not support non-Windows platforms.")
else()
  if (NOT APOLLO_SDK_PATH)
    set(APOLLO_SDK_PATH "APOLLO_SDK_PATH-NOTFOUND" CACHE PATH "Path to Apollo SDK." FORCE)
  endif()
  message(STATUS "Using Apollo SDK: ${APOLLO_SDK_PATH}")

  message(STATUS "Using LLVM: ${LLVM_TOOLS_BINARY_DIR}")
  message(STATUS "Using ONNX MLIR: ${ONNX_MLIR_RUNTIME_PATH}")

  project(MNIST)

  file(TO_CMAKE_PATH "${APOLLO_SDK_PATH}" ApolloSDK_PATH)

  set(ApolloSDK_Lib_PATH ${ApolloSDK_PATH}/lib)
  set(ApolloSDK_Package_PATH ${ApolloSDK_PATH}/packages)

  set(ONNX_MODEL_NAME mnist_padded_static_noconsts_2d_bias.onnx)
  set(ONNX_MODEL_F32_NAME mnist_padded_static_noconsts_2d_bias.f32.onnx)
  set(MNIST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
  set(MNIST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  add_custom_command(OUTPUT ${MNIST_BINARY_DIR}/${ONNX_MODEL_NAME}.ll
                     COMMAND ${Python3_EXECUTABLE} ${ONNX_MLIR_RUNTIME_PATH}/maia.py --f ${CMAKE_CURRENT_SOURCE_DIR}/Model/${ONNX_MODEL_NAME} --omb ${ONNX_MLIR_RUNTIME_PATH} --mb ${LLVM_TOOLS_BINARY_DIR}
                     MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/Model/${ONNX_MODEL_NAME}
                     DEPENDS copy-maia-tools
                     BYPRODUCTS ${MNIST_BINARY_DIR}/tcp_driver.tcp.cpp
                                ${MNIST_BINARY_DIR}/tcp_driver.tcp.h
                                ${MNIST_BINARY_DIR}/tcp_driver.tvp.cpp
                                ${MNIST_BINARY_DIR}/tcp_driver.tvp.h
                                ${MNIST_BINARY_DIR}/tcp_driver.dcp.cpp
                                ${MNIST_BINARY_DIR}/tcp_driver.dcp.h
                     VERBATIM)

  add_custom_command(OUTPUT ${MNIST_BINARY_DIR}/${ONNX_MODEL_NAME}.nounwind.ll
                     COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Model/attach_nounwind.py --f ${MNIST_BINARY_DIR}/${ONNX_MODEL_NAME}.ll
                     MAIN_DEPENDENCY ${MNIST_BINARY_DIR}/${ONNX_MODEL_NAME}.ll VERBATIM)

  add_custom_command(OUTPUT ${MNIST_BINARY_DIR}/${ONNX_MODEL_NAME}.s
                     COMMAND $<TARGET_FILE:llc> -mtriple=apollo-none-none -max-jump-table-size=0 -filetype=asm -O2 ${MNIST_BINARY_DIR}/${ONNX_MODEL_NAME}.nounwind.ll -o=${MNIST_BINARY_DIR}/${ONNX_MODEL_NAME}.s
                     MAIN_DEPENDENCY ${MNIST_BINARY_DIR}/${ONNX_MODEL_NAME}.nounwind.ll VERBATIM)

  add_custom_target(maia
                    DEPENDS ${MNIST_BINARY_DIR}/${ONNX_MODEL_NAME}.s)

  add_dependencies(maia onnx-mlir)

  set(CMAKE_MODULE_PATH "${ApolloSDK_PATH}/cmake")

  add_subdirectory(NPU)
  add_subdirectory(Host)

  add_custom_target(MNIST)

  add_dependencies(MNIST MNIST_Host_Program)

  add_test(NAME MNIST_NPU_SDK
           WORKING_DIRECTORY ${MNIST_BINARY_DIR}
           COMMAND Host/MNIST_Host_Program.exe NPU/MNIST_NPU_Program/MNIST_NPU_Program.zip)
endif()
