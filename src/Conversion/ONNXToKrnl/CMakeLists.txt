# SPDX-License-Identifier: Apache-2.0

add_library(OMONNXToKrnl
        ONNXToKrnlCommon.cpp
        ONNXToKrnlCommon.hpp
        ControlFlow/Loop.cpp
        ControlFlow/Scan.cpp
        Math/Clip.cpp
        Math/Elementwise.cpp
        Math/Gemm.cpp
        Math/LRN.cpp
        Math/MatMul.cpp
        Math/Reduction.cpp
        Math/Softmax.cpp
        NN/Conv.cpp
        NN/Normalization.cpp
        NN/Pooling.cpp
        RNN/RNNBase.cpp
        RNN/RNNBase.hpp
        RNN/GRU.cpp
        RNN/LSTM.cpp
        RNN/RNN.cpp
        Tensor/ArgMax.cpp
        Tensor/Expand.cpp
        Tensor/Identity.cpp
        Tensor/Reshape.cpp
        Tensor/Pad.cpp
        Tensor/Transpose.cpp
        Tensor/Squeeze.cpp
        Tensor/Unsqueeze.cpp
        Tensor/Constant.cpp
        Tensor/ConstantOfShape.cpp
        Tensor/Concat.cpp
        Tensor/Split.cpp
        Tensor/Gather.cpp
        Tensor/Shape.cpp
        Tensor/Slice.cpp
        Tensor/Size.cpp
        Tensor/Flatten.cpp
        Tensor/Tile.cpp
        ConvertONNXToKrnl.cpp)
target_link_libraries(OMONNXToKrnl
        OMSupport
        )
target_include_directories(OMONNXToKrnl
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})

# Header dependencies
add_dependencies(OMONNXToKrnl OMKrnlOpsInc OMONNXOpsInc)
# Linking dependencies
add_dependencies(OMONNXToKrnl OMKrnlOps)

install(TARGETS OMONNXToKrnl DESTINATION lib)
