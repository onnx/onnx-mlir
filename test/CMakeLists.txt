# SPDX-License-Identifier: Apache-2.0

add_subdirectory(mlir)
add_subdirectory(backend)
add_subdirectory(numerical)
add_subdirectory(unit)
add_subdirectory(onnx2mlir)
add_subdirectory(mnistmodels)
# ISSUE-TODO-ruzhang-2021/05/07: ApolloSDK is only included on Windows because there is issue building fork-Apollo on Ubuntu now.
# Once the build issue is addressed on Ubuntu, ApolloSDK can be enabled on Ubuntu as well.
if (WIN32)
  add_subdirectory(ApolloSDK)
endif()
