# SPDX-License-Identifier: Apache-2.0

add_executable(CustomFnTest CustomFnTest.cpp)

target_link_libraries(CustomFnTest
        ${OMLibs}
        ${MLIRLibs}
        onnx)
