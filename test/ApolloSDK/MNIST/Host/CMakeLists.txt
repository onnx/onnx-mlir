set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${ApolloSDK_Lib_PATH}/Host"
)

execute_process(COMMAND ${Python3_EXECUTABLE} -m pip install onnxruntime)

add_custom_command(OUTPUT ${MNIST_BINARY_DIR}/mnist_main.cpp
                   COMMAND ${Python3_EXECUTABLE} ${ONNX_MLIR_RUNTIME_PATH}/maia_generate_host.py --input ${CMAKE_CURRENT_SOURCE_DIR}/../Model/${ONNX_MODEL_F32_NAME} --output ${MNIST_BINARY_DIR}/mnist_main.cpp
                   MAIN_DEPENDENCY ${MNIST_SOURCE_DIR}/Model/${ONNX_MODEL_F32_NAME}
                   DEPENDS copy-maia-tools
                   VERBATIM)

include(HostTargets)

include_directories(${MNIST_BINARY_DIR})

add_executable(MNIST_Host_Program ${MNIST_BINARY_DIR}/mnist_main.cpp)
target_link_libraries(MNIST_Host_Program PUBLIC Runtime)
add_dependencies(MNIST_Host_Program MNIST_NPU_Program)

set_target_properties(MNIST_Host_Program
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${MNIST_BINARY_DIR}/Host"
)

add_custom_command(TARGET MNIST_Host_Program POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${MNIST_SOURCE_DIR}/inputs ${MNIST_BINARY_DIR}/inputs
  COMMENT "Copying model inputs"
)

add_custom_command(TARGET MNIST_Host_Program POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${MNIST_SOURCE_DIR}/outputs ${MNIST_BINARY_DIR}/outputs
  COMMENT "Copying model outputs"
)

if (WIN32)
  target_compile_definitions(MNIST_Host_Program PUBLIC NOMINMAX)
  add_custom_command(TARGET MNIST_Host_Program POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ApolloSDK_Lib_PATH}/Host/Emulator.dll ${MNIST_BINARY_DIR}
    COMMENT "Copying Emulator.dll"
  )

  add_custom_command(TARGET MNIST_Host_Program POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ApolloSDK_Lib_PATH}/Host/VShellLib.dll ${MNIST_BINARY_DIR}
    COMMENT "Copying VShellLib.dll"
  )

  add_custom_command(TARGET MNIST_Host_Program POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ApolloSDK_Lib_PATH}/Host/acl.dll ${MNIST_BINARY_DIR}
    COMMENT "Copying acl.dll"
  )
else()
  add_custom_command(TARGET MNIST_Host_Program POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ApolloSDK_Lib_PATH}/Host/libEmulator.so ${MNIST_BINARY_DIR}
    COMMENT "Copying libEmulator.so"
  )

  add_custom_command(TARGET MNIST_Host_Program POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ApolloSDK_Lib_PATH}/Host/libacl.so ${MNIST_BINARY_DIR}
    COMMENT "Copying libacl.so"
  )
endif()  
