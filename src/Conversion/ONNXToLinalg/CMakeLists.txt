add_onnx_mlir_library(OMONNXToLinalg
  ConvertONNXToLinalg.cpp
  )

target_link_libraries(OMONNXToLinalg
  onnx
  )

# Header dependencies
add_dependencies(OMONNXToLinalg OMONNXOpsInc)

# Linking dependencies
add_dependencies(OMONNXToLinalg OMONNXOps OMONNXToKrnl)
