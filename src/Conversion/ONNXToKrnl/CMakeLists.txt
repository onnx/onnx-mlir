add_library(OMONNXToKrnl
        ONNXToKrnlCommon.cpp
        ONNXToKrnlCommon.hpp
        Math/Elementwise.cpp
        Math/Gemm.cpp
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
        Tensor/Identity.cpp
        Tensor/Reshape.cpp
        Tensor/PadConstantValuePad.cpp
        Tensor/Pad.cpp
        Tensor/Transpose.cpp
        Tensor/Squeeze.cpp
        Tensor/Unsqueeze.cpp
        Tensor/Constant.cpp
        Tensor/ConstantOfShape.cpp
        Tensor/Concat.cpp
        Tensor/Split.cpp
        Tensor/Gather.cpp
        Tensor/Size.cpp
        Tensor/Flatten.cpp
        Tensor/Tile.cpp
        ConvertONNXToKrnl.cpp)
target_link_libraries(OMONNXToKrnl
        OMSupport
        onnx)
target_include_directories(OMONNXToKrnl
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})

# Header dependencies
add_dependencies(OMONNXToKrnl OMKrnlOpsInc OMONNXOpsInc)
# Linking dependencies
add_dependencies(OMONNXToKrnl OMKrnlOps)
