
# Invoke the MAIA graph compiler targeting Apollo
function(apollo_compile_graph TEST ADDITIONAL_ARGS)
  set (MODEL ${TEST}.onnx)
  add_custom_command(OUTPUT ${TEST_DIR}/${MODEL}.ll
                     COMMAND ${Python3_EXECUTABLE} ${ONNX_MLIR_RUNTIME_PATH}/maia.py --f ${TEST_DIR}/${MODEL} --omb ${ONNX_MLIR_RUNTIME_PATH} --mb ${LLVM_TOOLS_BINARY_DIR} ${ADDITIONAL_ARGS}
                     WORKING_DIRECTORY ${TEST_DIR}
                     MAIN_DEPENDENCY ${TEST_DIR}/${MODEL}
                     DEPENDS
                       copy-maia-tools
                       $<TARGET_FILE:mlir-opt>
                       $<TARGET_FILE:mlir-translate>
                       $<TARGET_FILE:llc>
                     BYPRODUCTS
                       ${TEST_DIR}/tcp_driver.tcp.cpp
                       ${TEST_DIR}/tcp_driver.tcp.h
                       ${TEST_DIR}/tcp_driver.tvp.cpp
                       ${TEST_DIR}/tcp_driver.tvp.h
                       ${TEST_DIR}/tcp_driver.dcp.cpp
                       ${TEST_DIR}/tcp_driver.dcp.h
                     VERBATIM)
    
  add_custom_command(OUTPUT ${TEST_DIR}/${MODEL}.nounwind.ll
                     COMMAND ${Python3_EXECUTABLE} ${MAIA_SOURCE_DIR}/attach_nounwind.py --f ${TEST_DIR}/${MODEL}.ll
                     MAIN_DEPENDENCY ${TEST_DIR}/${MODEL}.ll VERBATIM)

  add_custom_command(OUTPUT ${TEST_DIR}/${TEST}.s
                     COMMAND $<TARGET_FILE:llc> -mtriple=apollo-none-none -max-jump-table-size=0 -filetype=asm -O2 ${TEST_DIR}/${MODEL}.nounwind.ll -o=${TEST_DIR}/${TEST}.s
                     MAIN_DEPENDENCY ${TEST_DIR}/${MODEL}.nounwind.ll VERBATIM)

  add_custom_target(${TEST}-graph-compile
                    DEPENDS ${TEST_DIR}/${TEST}.s)

  add_dependencies(${TEST}-graph-compile onnx-mlir)
  add_dependencies(MAIA_TEST ${TEST}-graph-compile)
endfunction()

# Compile the host portion of an Apollo executable test
function(apollo_compile_host TEST)

  set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    "${ApolloSDK_Lib_PATH}/Host"
  )

  include(HostTargets)
  include_directories(${MAIA_BINARY_DIR})
  include_directories(${MAIA_SOURCE_DIR}/include)

  add_executable(${TEST}.host ${MAIA_BINARY_DIR}/${TEST}/host.cpp)
  target_link_libraries(${TEST}.host PUBLIC Runtime)

  set_target_properties(${TEST}.host
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${TEST_DIR}"
  )

  if (WIN32)
    target_compile_definitions(${TEST}.host PUBLIC NOMINMAX)
    add_custom_command(TARGET ${TEST}.host POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ApolloSDK_Lib_PATH}/Host/Emulator.dll ${TEST_DIR}
      COMMENT "Copying Emulator.dll"
    )

    add_custom_command(TARGET ${TEST}.host POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ApolloSDK_Lib_PATH}/Host/VShellLib.dll ${TEST_DIR}
      COMMENT "Copying VShellLib.dll"
    )

    add_custom_command(TARGET ${TEST}.host POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ApolloSDK_Lib_PATH}/Host/acl.dll ${TEST_DIR}
      COMMENT "Copying acl.dll"
    )
  else()
    add_custom_command(TARGET ${TEST}.host POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ApolloSDK_Lib_PATH}/Host/libEmulator.so ${TEST_DIR}
      COMMENT "Copying libEmulator.so"
    )

    add_custom_command(TARGET ${TEST}.host POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ApolloSDK_Lib_PATH}/Host/libacl.so ${TEST_DIR}
      COMMENT "Copying libacl.so"
    )
  endif()  

  add_dependencies(${TEST}.host ${TEST}-graph-compile)
  add_dependencies(MAIA_TEST ${TEST}.host)

endfunction()

# Compile the NPU portion of an Apollo executable test
function(apollo_compile_npu TEST)

  include(ApolloNPUProgram)
  
  # This is required to make the APOLLO SDK functions work correctly
  #   - The Apollo SDK fuctions implicitly determine input and output locations
  #   - We genereate multiple tests from the same source location which breaks
  #     this implicit generation.
  #   - We workaround this by chagning CMAKE_CURRENT_BINARY_DIR for each test
  # If we change the Apollo SDK functions to explicitly specficy input and
  # output locations we can remove this code and also move the Apollo specific
  # files to their own directory insttead of using the `apollo.` prefix.
  set(CMAKE_CURRENT_BINARY_DIR ${TEST_DIR})
  
  add_devicecp_build(${TEST}_DeviceCP
                     DIR apollo.DeviceCP
                     FIRMWARE DeviceCP
                     CMAKE_GENERATE_PARAMS -DDCP_CODE_FILE=${TEST_DIR}/tcp_driver.dcp.cpp
  )

  add_tilecp_build(${TEST}_TileCP
                   DIR apollo.TileCP
                   FIRMWARE TileCP
                   CMAKE_GENERATE_PARAMS -DTCP_CODE_FILE=${TEST_DIR}/tcp_driver.tcp.cpp
  )

  add_tvp_build(${TEST}_TVP
                DIR apollo.TVP
                FIRMWARE TVP
                CMAKE_GENERATE_PARAMS -DTVP_CODE_FILE=${TEST_DIR}/tcp_driver.tvp.cpp -DTVP_ASM_FILE=${TEST_DIR}/${TEST}.s
  )

  add_npu_program(${TEST}_NPU
                  MANIFEST ${MAIA_SOURCE_DIR}/apollo.manifest.json
                  FIRMWARE
                    ${TEST_DIR}/apollo.DeviceCP/DeviceCP
                    ${TEST_DIR}/apollo.TileCP/TileCP
                    ${TEST_DIR}/apollo.TVP/TVP
  )

  add_dependencies(${TEST}_DeviceCP ${TEST}-graph-compile)
  add_dependencies(${TEST}_TileCP ${TEST}-graph-compile)
  add_dependencies(${TEST}_TVP ${TEST}-graph-compile)
  add_dependencies(${TEST}_NPU ${TEST}_TVP ${TEST}_TileCP ${TEST}_DeviceCP)
  add_dependencies(MAIA_TEST ${TEST}_NPU)

endfunction()
