get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)


add_llvm_tool(onnx-lsp-server
  onnx-lsp-server.cpp
)


target_link_libraries(onnx-lsp-server
  PRIVATE
  ${OMLibs}

  OMCompilerDialects
  MLIRLspServerLib
)