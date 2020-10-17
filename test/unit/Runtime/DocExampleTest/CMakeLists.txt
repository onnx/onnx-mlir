# Documentation example requires ONNX package installation, which has been
# flaky on non-x86 platforms, so only perform this test on x86 arch.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
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
    target_include_directories(OMRuntimeTest
            PRIVATE ${ONNX_MLIR_SRC_ROOT}/include)
    add_dependencies(OMRuntimeTest OMCopyAndRename)
    # ISSUE-TODO-namcivca-2020-10-16: Disable test until build issues can be resolved
    #add_test(NAME OMRuntimeTest
    #        COMMAND OMRuntimeTest)
endif ()