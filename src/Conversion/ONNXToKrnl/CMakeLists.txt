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
        Tensor/Identity.cpp
        Tensor/Reshape.cpp
        Tensor/PadConstantValuePad.cpp
        Tensor/Transpose.cpp
        Tensor/Unsqueeze.cpp
        Tensor/Constant.cpp
        ConvertONNXToKrnl.cpp)
target_include_directories(OMONNXToKrnl
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})

# Linking dependencies:
add_dependencies(OMONNXToKrnl
        OMKrnlOps)