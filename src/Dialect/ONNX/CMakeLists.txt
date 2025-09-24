# SPDX-License-Identifier: Apache-2.0

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "s390x")
  # currently needed for a bug on Z for the ONNXOps.cpp.inc file
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O2" "-O1")
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O3" "-O1")
endif()

add_subdirectory(ElementsAttr)
add_subdirectory(ONNXOps)
add_subdirectory(Transforms)

add_onnx_mlir_dialect(ONNX onnx)
add_onnx_mlir_dialect_doc(onnx ONNX.td)

add_onnx_mlir_library(OMONNXOps
  # Top files for ONNX dialect
  DialectBuilder.cpp
  OnnxElementsAttrBuilder.cpp
  ONNXAttributes.cpp
  ONNXDialect.cpp
  ONNXDimAnalysis.cpp
  ONNXOps.cpp
  ONNXOps/OpHelper.cpp
  ONNXOps/Canonicalize.cpp
  ONNXOps/ShapeHelper.cpp
  ONNXTypes.cpp

  # Support for shape inference and verifiers
  ONNXOps/Additional/ConcatShapeTranspose.cpp
  ONNXOps/Additional/Custom.cpp
  ONNXOps/Additional/Dim.cpp
  ONNXOps/Additional/EntryPoint.cpp
  ONNXOps/Additional/Return.cpp
  ONNXOps/Additional/LayoutTransform.cpp
  ONNXOps/Additional/None.cpp
  ONNXOps/Additional/ShapeTransform.cpp
  ONNXOps/ControlFlow/If.cpp
  ONNXOps/ControlFlow/Loop.cpp
  ONNXOps/ControlFlow/Scan.cpp
  ONNXOps/ML/CategoryMapper.cpp
  ONNXOps/ML/OneHotEncoder.cpp
  ONNXOps/Math/Bernoulli.cpp
  ONNXOps/Math/DFT.cpp
  ONNXOps/Math/Einsum.cpp
  ONNXOps/Math/EinsumHelper.cpp
  ONNXOps/Math/ElementwiseBroadcast.cpp
  ONNXOps/Math/ElementwiseUnary.cpp
  ONNXOps/Math/Gemm.cpp
  ONNXOps/Math/HardMax.cpp
  ONNXOps/Math/Window.cpp
  ONNXOps/Math/LRN.cpp
  ONNXOps/Math/MatMul.cpp
  ONNXOps/Math/RandomNormal.cpp
  ONNXOps/Math/RandomNormalLike.cpp
  ONNXOps/Math/RandomUniform.cpp
  ONNXOps/Math/Reduction.cpp
  ONNXOps/Math/Scatter.cpp
  ONNXOps/Math/TopK.cpp
  ONNXOps/NN/Conv.cpp
  ONNXOps/NN/Dropout.cpp
  ONNXOps/NN/Normalization.cpp
  ONNXOps/NN/Pooling.cpp
  ONNXOps/NN/RoiAlign.cpp
  ONNXOps/ObjectDetection/NonMaxSuppression.cpp
  ONNXOps/Quantize/DequantizeLinear.cpp
  ONNXOps/Quantize/DynamicQuantizeLinear.cpp
  ONNXOps/Quantize/QuantizeLinear.cpp
  ONNXOps/RNN/RNN.cpp
  ONNXOps/Sequence/Sequence.cpp
  ONNXOps/Sequence/SplitToSequence.cpp
  ONNXOps/Tensor/ArgMinMax.cpp
  ONNXOps/Tensor/Compress.cpp
  ONNXOps/Tensor/Concat.cpp
  ONNXOps/Tensor/ConcatFromSequence.cpp
  ONNXOps/Tensor/Constant.cpp
  ONNXOps/Tensor/ConstantOfShape.cpp
  ONNXOps/Tensor/DepthToSpace.cpp
  ONNXOps/Tensor/Expand.cpp
  ONNXOps/Tensor/EyeLike.cpp
  ONNXOps/Tensor/Flatten.cpp
  ONNXOps/Tensor/Gather.cpp
  ONNXOps/Tensor/GatherElements.cpp
  ONNXOps/Tensor/GatherND.cpp
  ONNXOps/Tensor/GridSample.cpp
  ONNXOps/Tensor/Identity.cpp
  ONNXOps/Tensor/NonZero.cpp
  ONNXOps/Tensor/OneHot.cpp
  ONNXOps/Tensor/Optional.cpp
  ONNXOps/Tensor/Pad.cpp
  ONNXOps/Tensor/Range.cpp
  ONNXOps/Tensor/Reshape.cpp
  ONNXOps/Tensor/Resize.cpp
  ONNXOps/Tensor/ReverseSequence.cpp
  ONNXOps/Tensor/Shape.cpp
  ONNXOps/Tensor/Size.cpp
  ONNXOps/Tensor/Slice.cpp
  ONNXOps/Tensor/SpaceToDepth.cpp
  ONNXOps/Tensor/Split.cpp
  ONNXOps/Tensor/Squeeze.cpp
  ONNXOps/Tensor/Tile.cpp
  ONNXOps/Tensor/Transpose.cpp
  ONNXOps/Tensor/Unique.cpp
  ONNXOps/Tensor/Unsqueeze.cpp
  ONNXOps/Tensor/Upsample.cpp
  
  DEPENDS
  OMHasOnnxSubgraphOpInterfaceIncGen
  OMONNXIncGen
  OMONNXCanonicalizeIncGen
  OMResultTypeInferenceOpInterfaceIncGen
  OMShapeInferenceOpInterfaceIncGen

  LINK_LIBS PRIVATE
  OMDiagnostic

  LINK_LIBS PUBLIC
  OMMlirDialects
  OMMlirUtilities
  OMShapeHelperOpInterface
  OMONNXElementsAttr
  onnx
  MLIRFuncDialect
  MLIRMemRefTransforms
  )

configure_file(ONNXOps.td.inc.dc.in
  ${CMAKE_CURRENT_BINARY_DIR}/ONNXOps.td.inc.dc
  @ONLY
  )
