# SPDX-License-Identifier: Apache-2.0

# This will cause onnx to be built. More importantly, some variable definitions
# when building onnx such as -DONNX_ML=1 -DONNX_NAMESPACE=onnx will be carried over
# when compiling FrontendDialectHelper.cpp, etc. because they are properly exported
# as part of the onnx target and target_link_libraries interprets them correctly.
add_onnx_mlir_library(OMBuilder
  FrontendDialectHelper.cpp
  FrontendDialectTransformer.cpp
  ImportONNXUtils.cpp
  ModelInputShaper.cpp

  LINK_LIBS PUBLIC
  OMHasOnnxSubgraphOpInterface
  OMONNXOps
  OMResultTypeInferenceOpInterface
  MLIRFuncDialect
  onnx
  )

configure_file(OpBuildTable.inc.dc.in
  ${CMAKE_CURRENT_BINARY_DIR}/OpBuildTable.inc.dc
  @ONLY
  )
