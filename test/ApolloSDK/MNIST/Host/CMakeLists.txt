set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${ApolloSDK_Lib_PATH}/Host"
)

include(EmulatorTargets)
include(RuntimeTargets)

include_directories(${MNIST_SOURCE_DIR}/NPU/MNIST_NPU_Program/DeviceCP)
add_executable(MNIST_Host_Program mnist_main.cpp)
target_link_libraries(MNIST_Host_Program PUBLIC Runtime)

set_target_properties(MNIST_Host_Program
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${MNIST_BINARY_DIR}/Host"
)

if (WIN32)
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
