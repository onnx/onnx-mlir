


#set(DLC_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
#set(DLC_BIN_ROOT "${CMAKE_CURRENT_BINARY_DIR}")

set(DLC_LIBRARY_PATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(DLC_RUNTIME_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

#set(DLC_ONNX_MLIR_SRC_ROOT ${ONNX_MLIR_SRC_ROOT})
# set(DLC_ONNX_MLIR_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
# set(DLC_ONNX_MLIR_SRC_ROOT "${ONNX_MLIR_SRC_ROOT}/Accelerators/NNPA/src/")
#set(DLC_ONNX_MLIR_BIN_ROOT ${ONNX_MLIR_BIN_ROOT})


message(DEBUG "in NNPA/src: DLC_ONNX_MLIR_SRC_ROOT=${DLC_ONNX_MLIR_SRC_ROOT}") 
  add_subdirectory(Dialect) 
  add_subdirectory(Conversion)
  add_subdirectory(Support)
  add_subdirectory(Transform)
  
  # add_subdirectory(Tool)
  
  # find_package(ZLIB REQUIRED)
  

  add_subdirectory(Compiler)
  



