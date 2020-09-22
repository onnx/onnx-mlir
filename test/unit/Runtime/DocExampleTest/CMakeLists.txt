find_package(PythonInterp 3 REQUIRED)

add_custom_target(OMInstallOnnx
        COMMAND ${PYTHON_EXECUTABLE} setup.py install --user
        WORKING_DIRECTORY ${ONNX_MLIR_SRC_ROOT}/third_party/onnx)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gen_add_onnx.py
        ${CMAKE_CURRENT_BINARY_DIR}/gen_add_onnx.py COPYONLY)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/add.onnx
        COMMAND ${PYTHON_EXECUTABLE} gen_add_onnx.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(OMGenerateAddModel
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/add.onnx)
add_dependencies(OMGenerateAddModel OMInstallOnnx)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/add.so
        COMMAND onnx-mlir ${CMAKE_CURRENT_BINARY_DIR}/add.onnx
        DEPENDS OMGenerateAddModel)
add_custom_target(OMGenerateAddLibrary
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/add.so)
add_custom_target(OMCopyAndRename
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_BINARY_DIR}/add.so
        ${CMAKE_CURRENT_BINARY_DIR}/library.so
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/add.so)

add_executable(OMRuntimeTest
        main.c)

add_library(OMRuntimeTestModel SHARED IMPORTED)
file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/library.so)
set_property(TARGET OMRuntimeTestModel
        PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/library.so)
target_link_libraries(OMRuntimeTest
        ${CMAKE_CURRENT_BINARY_DIR}/library.so)
message(STATUS ${ONNX_MLIR_SRC_ROOT}/src/Runtime/include)
target_include_directories(OMRuntimeTest
        PRIVATE ${ONNX_MLIR_SRC_ROOT}/src/Runtime/include)
add_dependencies(OMRuntimeTest OMCopyAndRename)
add_test(NAME OMRuntimeTest
         COMMAND OMRuntimeTest)