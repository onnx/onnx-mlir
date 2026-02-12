// RUN: onnx-mlir-opt --decompose-onnx=enable-matmulnbits-decompose --mlir-print-elementsattrs-with-hex-if-larger=128 --mlir-elide-elementsattrs-if-larger=128 %s -split-input-file | FileCheck %s

func.func @matmul_n_bits_no_zp(%a: tensor<1x2x32xf32>) -> tensor<1x2x4xf32> {
  %b = "onnx.Constant"() {value = dense<24> : tensor<4x2x8xui8>} : () -> tensor<4x2x8xui8>
  %scales = "onnx.Constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>} : () -> tensor<4x2xf32>
  %out = "onnx.Custom"(%a, %b, %scales) {
    domain_name = "com.microsoft",
    function_name = "MatmulNBits",
    K = 32 : si64,
    N = 4 : si64,
    accuracy_level = 4 : si64,
    bits = 4 : si64,
    block_size = 16 : si64
  }: (tensor<1x2x32xf32>, tensor<4x2x8xui8>, tensor<4x2xf32>) -> tensor<1x2x4xf32>
  return %out : tensor<1x2x4xf32>
}

// CHECK-LABEL:  func.func @matmul_n_bits_no_zp
// CHECK-SAME:       (%[[ARG0:.*]]: tensor<1x2x32xf32>) -> tensor<1x2x4xf32> {
// CHECK-DAG:      %[[B:.*]] = onnx.Constant {{.*}} : tensor<1x4x32xui8>
// CHECK-DAG:      %[[SCALES:.*]] = onnx.Constant {{.*}} : tensor<4x2xf32>
// CHECK-DAG:      %[[ZPS:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:          %[[SCALES_R:.*]] = "onnx.Reshape"(%[[SCALES]]
// CHECK:          %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[B]], %[[SCALES_R]], %[[ZPS]]) {axis = -1 : si64, block_size = 16 : si64} 
// CHECK-SAME:         (tensor<1x4x32xui8>, tensor<1x4x2xf32>, none) -> tensor<1x4x32xf32>
// CHECK:          %[[TRANSPOSE:.*]] = "onnx.Transpose"(%[[DQ]]) {perm = [0, 2, 1]} : (tensor<1x4x32xf32>) -> tensor<1x32x4xf32>
// CHECK:          %[[MM:.*]] = "onnx.MatMul"(%[[ARG0]], %[[TRANSPOSE]]) : (tensor<1x2x32xf32>, tensor<1x32x4xf32>) -> tensor<1x2x4xf32>
// CHECK:          return %[[MM]] : tensor<1x2x4xf32>

// ----- 

func.func @matmul_n_bits(%a: tensor<1x2x32xf32>) -> tensor<1x2x4xf32> {
  %b = "onnx.Constant"() {value = dense<24> : tensor<4x2x8xui8>} : () -> tensor<4x2x8xui8>
  %scales = "onnx.Constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>} : () -> tensor<4x2xf32>
  %zps = "onnx.Constant"() {value = dense<[[21], [22], [23], [24]]> : tensor<4x1xui8>} : () -> tensor<4x1xui8>
  %out = "onnx.Custom"(%a, %b, %scales, %zps) {
    domain_name = "com.microsoft",
    function_name = "MatmulNBits",
    K = 32 : si64,
    N = 4 : si64,
    accuracy_level = 4 : si64,
    bits = 4 : si64,
    block_size = 16 : si64
  }: (tensor<1x2x32xf32>, tensor<4x2x8xui8>, tensor<4x2xf32>, tensor<4x1xui8>) -> tensor<1x2x4xf32>
  return %out : tensor<1x2x4xf32>
}

// CHECK-LABEL:  func.func @matmul_n_bits
// CHECK-SAME:       (%[[ARG0:.*]]: tensor<1x2x32xf32>) -> tensor<1x2x4xf32> {
// CHECK-DAG:      %[[B:.*]] = onnx.Constant {{.*}} : tensor<1x4x32xui8>
// CHECK-DAG:      %[[SCALES:.*]] = onnx.Constant {{.*}} : tensor<4x2xf32>
// CHECK-DAG:      %[[ZPS:.*]] = onnx.Constant {{.*}} : tensor<1x4x2xui8>
// CHECK:          %[[SCALES_R:.*]] = "onnx.Reshape"(%[[SCALES]], {{.*}} : (tensor<4x2xf32>, tensor<3xi64>) -> tensor<1x4x2xf32>
// CHECK:          %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[B]], %[[SCALES_R]], %[[ZPS]]) {axis = -1 : si64, block_size = 16 : si64} 
// CHECK-SAME:         (tensor<1x4x32xui8>, tensor<1x4x2xf32>, tensor<1x4x2xui8>) -> tensor<1x4x32xf32>
// CHECK:          %[[TRANSPOSE:.*]] = "onnx.Transpose"(%[[DQ]]) {perm = [0, 2, 1]} : (tensor<1x4x32xf32>) -> tensor<1x32x4xf32>
// CHECK:          %[[MM:.*]] = "onnx.MatMul"(%[[ARG0]], %[[TRANSPOSE]]) : (tensor<1x2x32xf32>, tensor<1x32x4xf32>) -> tensor<1x2x4xf32>
// CHECK:          return %[[MM]] : tensor<1x2x4xf32>

// ----- 

func.func @matmul_n_bits_bias(%a: tensor<1x2x32xf32>) -> tensor<1x2x4xf32> {
  %b = "onnx.Constant"() {value = dense<24> : tensor<4x2x8xui8>} : () -> tensor<4x2x8xui8>
  %scales = "onnx.Constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>} : () -> tensor<4x2xf32>
  %zps = "onnx.Constant"() {value = dense<[[21], [22], [23], [24]]> : tensor<4x1xui8>} : () -> tensor<4x1xui8>
  %bias = "onnx.Constant"() {value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %out = "onnx.Custom"(%a, %b, %scales, %zps, %none, %bias) {
    domain_name = "com.microsoft",
    function_name = "MatmulNBits",
    K = 32 : si64,
    N = 4 : si64,
    accuracy_level = 4 : si64,
    bits = 4 : si64,
    block_size = 16 : si64
  }: (tensor<1x2x32xf32>, tensor<4x2x8xui8>, tensor<4x2xf32>, tensor<4x1xui8>, none, tensor<4xf32>) -> tensor<1x2x4xf32>
  return %out : tensor<1x2x4xf32>
}

// CHECK-LABEL:  func.func @matmul_n_bits_bias
// CHECK-SAME:       (%[[ARG0:.*]]: tensor<1x2x32xf32>) -> tensor<1x2x4xf32> {
// CHECK-DAG:      %[[B:.*]] = onnx.Constant {{.*}} : tensor<1x4x32xui8>
// CHECK-DAG:      %[[SCALES:.*]] = onnx.Constant {{.*}} : tensor<4x2xf32>
// CHECK-DAG:      %[[ZPS:.*]] = onnx.Constant {{.*}} : tensor<1x4x2xui8>
// CHECK:          %[[SCALES_R:.*]] = "onnx.Reshape"(%[[SCALES]]
// CHECK:          %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[B]], %[[SCALES_R]], %[[ZPS]]) {axis = -1 : si64, block_size = 16 : si64} 
// CHECK-SAME:         (tensor<1x4x32xui8>, tensor<1x4x2xf32>, tensor<1x4x2xui8>) -> tensor<1x4x32xf32>
// CHECK:          %[[TRANSPOSE:.*]] = "onnx.Transpose"(%[[DQ]]) {perm = [0, 2, 1]} : (tensor<1x4x32xf32>) -> tensor<1x32x4xf32>
// CHECK:          %[[MM:.*]] = "onnx.MatMul"(%[[ARG0]], %[[TRANSPOSE]]) : (tensor<1x2x32xf32>, tensor<1x32x4xf32>) -> tensor<1x2x4xf32>
// CHECK:          %[[BIAS:.*]] = "onnx.Add"(%[[MM]]
// CHECK:          return %[[BIAS]] : tensor<1x2x4xf32>

// ----- 

func.func @matmul_n_bits_bits_2(%a: tensor<1x2x64xf32>) -> tensor<1x2x4xf32> {
  %b = "onnx.Constant"() {value = dense<24> : tensor<4x4x4xui8>} : () -> tensor<4x4x4xui8>
  %scales = "onnx.Constant"() {value = dense<2.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
  %zps = "onnx.Constant"() {value = dense<[[21], [22], [23], [24]]> : tensor<4x1xui8>} : () -> tensor<4x1xui8>
  %out = "onnx.Custom"(%a, %b, %scales, %zps) {
    domain_name = "com.microsoft",
    function_name = "MatmulNBits",
    K = 64 : si64,
    N = 4 : si64,
    accuracy_level = 4 : si64,
    bits = 2 : si64,
    block_size = 16 : si64
  }: (tensor<1x2x64xf32>, tensor<4x4x4xui8>, tensor<4x4xf32>, tensor<4x1xui8>) -> tensor<1x2x4xf32>
  return %out : tensor<1x2x4xf32>
}

// CHECK-LABEL:  func.func @matmul_n_bits_bits_2
// CHECK-SAME:       (%[[ARG0:.*]]: tensor<1x2x64xf32>) -> tensor<1x2x4xf32> {
// CHECK-DAG:      %[[B:.*]] = onnx.Constant {{.*}} : tensor<1x4x64xui8>
// CHECK-DAG:      %[[SCALES:.*]] = onnx.Constant {{.*}} : tensor<4x4xf32>
// CHECK-DAG:      %[[ZPS:.*]] = onnx.Constant {{.*}} : tensor<1x4x4xui8>
// CHECK:          %[[SCALES_R:.*]] = "onnx.Reshape"(%[[SCALES]]
// CHECK:          %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[B]], %[[SCALES_R]], %[[ZPS]]) {axis = -1 : si64, block_size = 16 : si64} 
// CHECK-SAME:         (tensor<1x4x64xui8>, tensor<1x4x4xf32>, tensor<1x4x4xui8>) -> tensor<1x4x64xf32>
// CHECK:          %[[TRANSPOSE:.*]] = "onnx.Transpose"(%[[DQ]]) {perm = [0, 2, 1]} : (tensor<1x4x64xf32>) -> tensor<1x64x4xf32>
// CHECK:          %[[MM:.*]] = "onnx.MatMul"(%[[ARG0]], %[[TRANSPOSE]]) : (tensor<1x2x64xf32>, tensor<1x64x4xf32>) -> tensor<1x2x4xf32>
// CHECK:          return %[[MM]] : tensor<1x2x4xf32>

// ----- 

func.func @matmul_n_bits_bits_8(%a: tensor<1x2x64xf32>) -> tensor<1x2x4xf32> {
  %b = "onnx.Constant"() {value = dense<24> : tensor<4x4x16xui8>} : () -> tensor<4x4x16xui8>
  %scales = "onnx.Constant"() {value = dense<2.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
  %zps = "onnx.Constant"() {value = dense<20> : tensor<4x4xui8>} : () -> tensor<4x4xui8>
  %out = "onnx.Custom"(%a, %b, %scales, %zps) {
    domain_name = "com.microsoft",
    function_name = "MatmulNBits",
    K = 64 : si64,
    N = 4 : si64,
    accuracy_level = 4 : si64,
    bits = 8 : si64,
    block_size = 16 : si64
  }: (tensor<1x2x64xf32>, tensor<4x4x16xui8>, tensor<4x4xf32>, tensor<4x4xui8>) -> tensor<1x2x4xf32>
  return %out : tensor<1x2x4xf32>
}


// CHECK-LABEL:  func.func @matmul_n_bits_bits_8
// CHECK-SAME:       (%[[ARG0:.*]]: tensor<1x2x64xf32>) -> tensor<1x2x4xf32> {
// CHECK-DAG:      %[[B:.*]] = onnx.Constant {{.*}} : tensor<1x4x64xui8>
// CHECK-DAG:      %[[SCALES:.*]] = onnx.Constant {{.*}} : tensor<4x4xf32>
// CHECK-DAG:      %[[ZPS:.*]] = onnx.Constant {{.*}} : tensor<1x4x4xui8>
// CHECK:          %[[SCALES_R:.*]] = "onnx.Reshape"(%[[SCALES]]
// CHECK:          %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[B]], %[[SCALES_R]], %[[ZPS]]) {axis = -1 : si64, block_size = 16 : si64} 
// CHECK-SAME:         (tensor<1x4x64xui8>, tensor<1x4x4xf32>, tensor<1x4x4xui8>) -> tensor<1x4x64xf32>
// CHECK:          %[[TRANSPOSE:.*]] = "onnx.Transpose"(%[[DQ]]) {perm = [0, 2, 1]} : (tensor<1x4x64xf32>) -> tensor<1x64x4xf32>
// CHECK:          %[[MM:.*]] = "onnx.MatMul"(%[[ARG0]], %[[TRANSPOSE]]) : (tensor<1x2x64xf32>, tensor<1x64x4xf32>) -> tensor<1x2x4xf32>
// CHECK:          return %[[MM]] : tensor<1x2x4xf32>

// ----- 

func.func @matmul_n_bits_block_32(%a: tensor<1x2x64xf32>) -> tensor<1x2x4xf32> {
  %b = "onnx.Constant"() {value = dense<24> : tensor<4x2x16xui8>} : () -> tensor<4x2x16xui8>
  %scales = "onnx.Constant"() {value = dense<2.0> : tensor<4x2xf32>} : () -> tensor<4x2xf32>
  %zps = "onnx.Constant"() {value = dense<20> : tensor<4x1xui8>} : () -> tensor<4x1xui8>
  %out = "onnx.Custom"(%a, %b, %scales, %zps) {
    domain_name = "com.microsoft",
    function_name = "MatmulNBits",
    K = 64 : si64,
    N = 4 : si64,
    accuracy_level = 4 : si64,
    bits = 4 : si64,
    block_size = 32 : si64
  }: (tensor<1x2x64xf32>, tensor<4x2x16xui8>, tensor<4x2xf32>, tensor<4x1xui8>) -> tensor<1x2x4xf32>
  return %out : tensor<1x2x4xf32>
}

// CHECK-LABEL:  func.func @matmul_n_bits_block_32
// CHECK-SAME:       (%[[ARG0:.*]]: tensor<1x2x64xf32>) -> tensor<1x2x4xf32> {
// CHECK-DAG:      %[[B:.*]] = onnx.Constant {{.*}} : tensor<1x4x64xui8>
// CHECK-DAG:      %[[SCALES:.*]] = onnx.Constant {{.*}} : tensor<4x2xf32>
// CHECK-DAG:      %[[ZPS:.*]] = onnx.Constant {{.*}} : tensor<1x4x2xui8>
// CHECK:          %[[SCALES_R:.*]] = "onnx.Reshape"(%[[SCALES]]
// CHECK:          %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[B]], %[[SCALES_R]], %[[ZPS]]) {axis = -1 : si64, block_size = 32 : si64} 
// CHECK-SAME:         (tensor<1x4x64xui8>, tensor<1x4x2xf32>, tensor<1x4x2xui8>) -> tensor<1x4x64xf32>
// CHECK:          %[[TRANSPOSE:.*]] = "onnx.Transpose"(%[[DQ]]) {perm = [0, 2, 1]} : (tensor<1x4x64xf32>) -> tensor<1x64x4xf32>
// CHECK:          %[[MM:.*]] = "onnx.MatMul"(%[[ARG0]], %[[TRANSPOSE]]) : (tensor<1x2x64xf32>, tensor<1x64x4xf32>) -> tensor<1x2x4xf32>
// CHECK:          return %[[MM]] : tensor<1x2x4xf32>

// ----- 

func.func @matmul_n_bits_1d_scales_zp(%a: tensor<1x2x32xf32>) -> tensor<1x2x4xf32> {
  %b = "onnx.Constant"() {value = dense<24> : tensor<4x2x8xui8>} : () -> tensor<4x2x8xui8>
  %scales = "onnx.Constant"() {value = dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]> : tensor<8xf32>} : () -> tensor<8xf32>
  %zps = "onnx.Constant"() {value = dense<[21, 22, 23, 24]> : tensor<4xui8>} : () -> tensor<4xui8>
  %out = "onnx.Custom"(%a, %b, %scales, %zps) {
    domain_name = "com.microsoft",
    function_name = "MatmulNBits",
    K = 32 : si64,
    N = 4 : si64,
    accuracy_level = 4 : si64,
    bits = 4 : si64,
    block_size = 16 : si64
  }: (tensor<1x2x32xf32>, tensor<4x2x8xui8>, tensor<8xf32>, tensor<4xui8>) -> tensor<1x2x4xf32>
  return %out : tensor<1x2x4xf32>
}

// CHECK-LABEL:  func.func @matmul_n_bits_1d_scales_zp
// CHECK-SAME:       (%[[ARG0:.*]]: tensor<1x2x32xf32>) -> tensor<1x2x4xf32> {
// CHECK-DAG:      %[[B:.*]] = onnx.Constant {{.*}} : tensor<1x4x32xui8>
// CHECK-DAG:      %[[SCALES:.*]] = onnx.Constant {{.*}} : tensor<8xf32>
// CHECK-DAG:      %[[ZPS:.*]] = onnx.Constant {{.*}} : tensor<1x4x2xui8>
// CHECK:          %[[SCALES_R:.*]] = "onnx.Reshape"(%[[SCALES]]
// CHECK:          %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[B]], %[[SCALES_R]], %[[ZPS]]) {axis = -1 : si64, block_size = 16 : si64} 
// CHECK-SAME:         (tensor<1x4x32xui8>, tensor<1x4x2xf32>, tensor<1x4x2xui8>) -> tensor<1x4x32xf32>
// CHECK:          %[[TRANSPOSE:.*]] = "onnx.Transpose"(%[[DQ]]) {perm = [0, 2, 1]} : (tensor<1x4x32xf32>) -> tensor<1x32x4xf32>
// CHECK:          %[[MM:.*]] = "onnx.MatMul"(%[[ARG0]], %[[TRANSPOSE]]) : (tensor<1x2x32xf32>, tensor<1x32x4xf32>) -> tensor<1x2x4xf32>
// CHECK:          return %[[MM]] : tensor<1x2x4xf32>

// -----

// scales expect shape N x ceil(K / block_size)
//   => ceil(56 / 16) = ceil(3.5) = 4
// zero_points expect shape N x ceil((K * bits) / (8 * block_size))
//   => ceil((56 * 4) / (8 * 16)) = ceil(1.75) = 2 
func.func @matmul_n_bits_slice_scales(%a: tensor<1x3x56xf32>) -> tensor<1x3x2xf32> {
  %b = "onnx.Constant"() {value = dense<24> : tensor<2x4x8xui8>} : () -> tensor<2x4x8xui8>
  %scales = "onnx.Constant"() {value = dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %zps = "onnx.Constant"() {value = dense<[[21, 22], [23, 24]]> : tensor<2x2xui8>} : () -> tensor<2x2xui8>
  %out = "onnx.Custom"(%a, %b, %scales, %zps) {
    domain_name = "com.microsoft",
    function_name = "MatmulNBits",
    K = 56 : si64,
    N = 2 : si64,
    accuracy_level = 4 : si64,
    bits = 4 : si64,
    block_size = 16 : si64
  }: (tensor<1x3x56xf32>, tensor<2x4x8xui8>, tensor<2x4xf32>, tensor<2x2xui8>) -> tensor<1x3x2xf32>
  return %out : tensor<1x3x2xf32>
}

// CHECK-LABEL:  func.func @matmul_n_bits_slice_scales
// CHECK-SAME:       (%[[ARG0:.*]]: tensor<1x3x56xf32>) -> tensor<1x3x2xf32> {

// CHECK-DAG:      %[[B:.*]] = onnx.Constant {{.*}} : tensor<1x2x64xui8>
// CHECK-DAG:      %[[START:.*]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:      %[[END:.*]] = onnx.Constant dense<56> : tensor<1xi64>
// CHECK-DAG:      %[[AXIS:.*]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:      %[[STEP:.*]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:      %[[B_SLICE:.*]] = "onnx.Slice"(%[[B]], %[[START]], %[[END]], %[[AXIS]], %[[STEP]])

// CHECK-DAG:      %[[SCALES:.*]] = onnx.Constant {{.*}} : tensor<2x4xf32>
// CHECK-DAG:      %[[ZPS:.*]] = onnx.Constant {{.*}} : tensor<1x2x4xui8>

// CHECK:          %[[SCALES_R:.*]] = "onnx.Reshape"(%[[SCALES]]
// CHECK:          %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[B_SLICE]], %[[SCALES_R]], %[[ZPS]]) {axis = -1 : si64, block_size = 16 : si64} 
// CHECK-SAME:         (tensor<1x2x56xui8>, tensor<1x2x4xf32>, tensor<1x2x4xui8>) -> tensor<1x2x56xf32>
// CHECK:          %[[TRANSPOSE:.*]] = "onnx.Transpose"(%[[DQ]]) {perm = [0, 2, 1]} : (tensor<1x2x56xf32>) -> tensor<1x56x2xf32>
// CHECK:          %[[MM:.*]] = "onnx.MatMul"(%[[ARG0]], %[[TRANSPOSE]]) : (tensor<1x3x56xf32>, tensor<1x56x2xf32>) -> tensor<1x3x2xf32>
// CHECK:          return %[[MM]] : tensor<1x3x2xf32>

// -----

// zero_points expect shape N x ceil((K * bits) / (8 * block_size))
//   => ceil((896 * 4) / (8 * 128)) = ceil(3.5) = 4 
func.func @matmul_n_bits_slice_zps(%arg0: tensor<1x256x896xf32> ) -> (tensor<1x256x1152xf32> ) {
  %0 = onnx.Constant {value = dense<24> : tensor<1152x7x64xui8>} : tensor<1152x7x64xui8> 
  %1 = onnx.Constant {value = dense<1.5> : tensor<1152x7xf32>} : tensor<1152x7xf32> 
  %2 = onnx.Constant {value = dense<1> : tensor<1152x4xui8>} : tensor<1152x4xui8> 
  %3 = "onnx.Custom"(%arg0, %0, %1, %2) {
    K = 896 : si64,
    N = 1152 : si64,
    bits = 4 : si64,
    block_size = 128 : si64,
    domain_name = "com.microsoft",
    function_name = "MatMulNBits"} : (tensor<1x256x896xf32>, tensor<1152x7x64xui8>, tensor<1152x7xf32>, tensor<1152x4xui8>) -> tensor<1x256x1152xf32> 
  return %3 : tensor<1x256x1152xf32> 
} 

// CHECK-LABEL:  func.func @matmul_n_bits_slice_zps
// CHECK-SAME:       (%[[ARG0:.*]]: tensor<1x256x896xf32>) -> tensor<1x256x1152xf32> {

// CHECK-DAG:      %[[B:.*]] = onnx.Constant {{.*}} : tensor<1x1152x896xui8>
// CHECK-DAG:      %[[SCALES:.*]] = onnx.Constant {{.*}} : tensor<1152x7xf32>

// CHECK-DAG:      %[[ZPS:.*]] = onnx.Constant {{.*}} : tensor<1x1152x8xui8>
// CHECK-DAG:      %[[START:.*]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:      %[[END:.*]] = onnx.Constant dense<7> : tensor<1xi64>
// CHECK-DAG:      %[[AXIS:.*]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:      %[[STEP:.*]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:      %[[ZPS_SLICE:.*]] = "onnx.Slice"(%[[ZPS]], %[[START]], %[[END]], %[[AXIS]], %[[STEP]])

// CHECK:          %[[SCALES_R:.*]] = "onnx.Reshape"(%[[SCALES]]
// CHECK:          %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[B]], %[[SCALES_R]], %[[ZPS_SLICE]]) {axis = -1 : si64, block_size = 128 : si64} 
// CHECK-SAME:         (tensor<1x1152x896xui8>, tensor<1x1152x7xf32>, tensor<1x1152x7xui8>) -> tensor<1x1152x896xf32>
// CHECK:          %[[TRANSPOSE:.*]] = "onnx.Transpose"(%[[DQ]]) {perm = [0, 2, 1]} : (tensor<1x1152x896xf32>) -> tensor<1x896x1152xf32>
// CHECK:          %[[MM:.*]] = "onnx.MatMul"(%[[ARG0]], %[[TRANSPOSE]]) : (tensor<1x256x896xf32>, tensor<1x896x1152xf32>) -> tensor<1x256x1152xf32>
// CHECK:          return %[[MM]] : tensor<1x256x1152xf32>

// ----- 

func.func @matmul_n_bits_check_vals(%a: tensor<1x2x32xf32>) -> tensor<1x2x4xf32> {
  %b = "onnx.Constant"() {value = dense<[[[199, 1, 246, 96, 155, 126, 139, 83], [110, 31, 77, 191, 86, 178, 184, 102]], [[182, 152, 221, 199, 136, 3, 33, 146], [254, 140, 149, 31, 70, 105, 162, 128]], [[17, 172, 53, 103, 26, 58, 111, 143], [4, 178, 107, 147, 14, 6, 214, 144]], [[13, 11, 106, 145, 217, 24, 197, 222], [59, 109, 156, 84, 130, 77, 31, 21]]]> : tensor<4x2x8xui8>} : () -> tensor<4x2x8xui8>
  %scales = "onnx.Constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>} : () -> tensor<4x2xf32>
  %zps = "onnx.Constant"() {value = dense<[[65], [45], [111], [182]]> : tensor<4x1xui8>} : () -> tensor<4x1xui8>
  %out = "onnx.Custom"(%a, %b, %scales, %zps) {
    domain_name = "com.microsoft",
    function_name = "MatmulNBits",
    K = 32 : si64,
    N = 4 : si64,
    accuracy_level = 4 : si64,
    bits = 4 : si64,
    block_size = 16 : si64
  }: (tensor<1x2x32xf32>, tensor<4x2x8xui8>, tensor<4x2xf32>, tensor<4x1xui8>) -> tensor<1x2x4xf32>
  return %out : tensor<1x2x4xf32>
}

// These values have been verified to be correct using the python onnx-runtime libraries.
// CHECK-LABEL:  func.func @matmul_n_bits_check_vals

// CHECK:          %[[B:.*]] = onnx.Constant dense<{{\[\[}}[7, 12, 1, 0, 6, 15, 0, 6, 11, 9, 14, 7, 11, 8, 3, 5, 14, 6, 15, 1, 13, 4, 15, 11, 6, 5, 2, 11, 8, 11, 6, 6], 
// CHECK-SAME:       [6, 11, 8, 9, 13, 13, 7, 12, 8, 8, 3, 0, 1, 2, 2, 9, 14, 15, 12, 8, 5, 9, 15, 1, 6, 4, 9, 6, 2, 10, 0, 8],
// CHECK-SAME:       [1, 1, 12, 10, 5, 3, 7, 6, 10, 1, 10, 3, 15, 6, 15, 8, 4, 0, 2, 11, 11, 6, 3, 9, 14, 0, 6, 0, 6, 13, 0, 9], 
// CHECK-SAME:       [13, 0, 11, 0, 10, 6, 1, 9, 9, 13, 8, 1, 5, 12, 14, 13, 11, 3, 13, 6, 12, 9, 4, 5, 2, 8, 13, 4, 15, 1, 5, 1]
// CHECK-SAME:     ]]> : tensor<1x4x32xui8>

// CHECK:          %[[ZPS:.*]] = onnx.Constant dense<{{\[\[}}[1, 4], [13, 2], [15, 6], [6, 11]]]> : tensor<1x4x2xui8>
// CHECK:          "onnx.DequantizeLinear"(%[[B]], %{{.*}}, %[[ZPS]]) {axis = -1 : si64, block_size = 16 : si64} 

// -----

func.func @matmul_n_bits_not_divisible_check_vals(%a: tensor<1x3x56xf32>) -> tensor<1x3x2xf32> {
  %b = "onnx.Constant"() {value = dense<[[[199, 1, 246, 96, 155, 126, 139, 83], [110, 31, 77, 191, 86, 178, 184, 102], [182, 152, 221, 199, 136, 3, 33, 146], [254, 140, 149, 31, 70, 105, 162, 128]], [[17, 172, 53, 103, 26, 58, 111, 143], [4, 178, 107, 147, 14, 6, 214, 144], [13, 11, 106, 145, 217, 24, 197, 222], [59, 109, 156, 84, 130, 77, 31, 21]]]> : tensor<2x4x8xui8>} : () -> tensor<2x4x8xui8>
  %scales = "onnx.Constant"() {value = dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %zps = "onnx.Constant"() {value = dense<[[65, 45], [111, 182]]> : tensor<2x2xui8>} : () -> tensor<2x2xui8>
  %out = "onnx.Custom"(%a, %b, %scales, %zps) {
    domain_name = "com.microsoft",
    function_name = "MatmulNBits",
    K = 56 : si64,
    N = 2 : si64,
    accuracy_level = 4 : si64,
    bits = 4 : si64,
    block_size = 16 : si64
  }: (tensor<1x3x56xf32>, tensor<2x4x8xui8>, tensor<2x4xf32>, tensor<2x2xui8>) -> tensor<1x3x2xf32>
  return %out : tensor<1x3x2xf32>
}

// These values have been verified to be correct using the python onnx-runtime libraries.
// CHECK-LABEL:  func.func @matmul_n_bits_not_divisible_check_vals

// CHECK:          %[[ZPS:.*]] = onnx.Constant dense<{{\[\[}}[1, 4, 13, 2], [15, 6, 6, 11]]]> : tensor<1x2x4xui8>

// CHECK:          %[[B:.*]] = onnx.Constant dense<{{\[\[}}[7, 12, 1, 0, 6, 15, 0, 6, 11, 9, 14, 7, 11, 8, 3, 5, 14, 6, 15, 1, 13, 4, 15, 11, 6, 5, 2, 11, 8, 11, 6, 6, 6, 11, 8, 9, 13, 13, 7, 12, 8, 8, 3, 0, 1, 2, 2, 9, 14, 15, 12, 8, 5, 9, 15, 1, 6, 4, 9, 6, 2, 10, 0, 8],
// CHECK-SAME:       [1, 1, 12, 10, 5, 3, 7, 6, 10, 1, 10, 3, 15, 6, 15, 8, 4, 0, 2, 11, 11, 6, 3, 9, 14, 0, 6, 0, 6, 13, 0, 9, 13, 0, 11, 0, 10, 6, 1, 9, 9, 13, 8, 1, 5, 12, 14, 13, 11, 3, 13, 6, 12, 9, 4, 5, 2, 8, 13, 4, 15, 1, 5, 1] 
// CHECK-SAME:     ]]> : tensor<1x2x64xui8>

// CHECK:          %[[START:.*]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:      %[[END:.*]] = onnx.Constant dense<56> : tensor<1xi64>
// CHECK-DAG:      %[[AXIS:.*]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:      %[[STEP:.*]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:      %[[B_SLICE:.*]] = "onnx.Slice"(%[[B]], %[[START]], %[[END]], %[[AXIS]], %[[STEP]])

// CHECK:          "onnx.DequantizeLinear"(%[[B_SLICE]], %{{.*}}, %[[ZPS]]) {axis = -1 : si64, block_size = 16 : si64} 