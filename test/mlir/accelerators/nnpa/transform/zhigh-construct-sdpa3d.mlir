// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --zhigh-construct-sdpa3d %s -split-input-file | FileCheck %s

// Test: Fuse MatMul(Softmax(MatMul(Q, KT) + Mask), V) into zhigh.SDPA3D
// when inner MatMul has transposeB=0 (KT already transposed).
func.func @test_sdpa3d_no_transpose(
    %Q: tensor<2x64x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>,
    %KT: tensor<2x32x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>,
    %Mask: tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>,
    %V: tensor<2x128x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>) ->
    tensor<*xf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "zhigh.MatMul"(%Q, %KT, %none) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<2x64x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x32x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %1 = "zhigh.Add"(%0, %Mask) : (tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %2 = "zhigh.Softmax"(%1) {act_func = "ACT_NONE"} : (tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %3 = "zhigh.MatMul"(%2, %V, %none) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x128x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<*xf16>
  return %3 : tensor<*xf16>

  // CHECK-LABEL: test_sdpa3d_no_transpose
  // CHECK: "zhigh.SDPA3D"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{transposeKT = 0 : si64}>
  // CHECK-NOT: zhigh.MatMul
}

// -----

// Test: Fuse MatMul(Softmax(MatMul(Q, K) + Mask), V) into zhigh.SDPA3D
// when inner MatMul has transposeB=1 (K is not transposed, hardware transposes it).
func.func @test_sdpa3d_transpose_kt(
    %Q: tensor<2x64x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>,
    %K: tensor<2x128x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>,
    %Mask: tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>,
    %V: tensor<2x128x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>) ->
    tensor<*xf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "zhigh.MatMul"(%Q, %K, %none) {transposeA = 0 : si64, transposeB = 1 : si64} : (tensor<2x64x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x128x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %1 = "zhigh.Add"(%0, %Mask) : (tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %2 = "zhigh.Softmax"(%1) {act_func = "ACT_NONE"} : (tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %3 = "zhigh.MatMul"(%2, %V, %none) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x128x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<*xf16>
  return %3 : tensor<*xf16>

  // CHECK-LABEL: test_sdpa3d_transpose_kt
  // CHECK: "zhigh.SDPA3D"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{transposeKT = 1 : si64}>
  // CHECK-NOT: zhigh.MatMul
}

// -----

// Test: Do NOT fuse when inner MatMul has transposeA=1 (not supported).
func.func @test_sdpa3d_no_fuse_transpose_a(
    %Q: tensor<2x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>,
    %KT: tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>,
    %Mask: tensor<2x32x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>,
    %V: tensor<2x128x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>) ->
    tensor<*xf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "zhigh.MatMul"(%Q, %KT, %none) {transposeA = 1 : si64, transposeB = 0 : si64} : (tensor<2x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<2x32x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %1 = "zhigh.Add"(%0, %Mask) : (tensor<2x32x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x32x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x32x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %2 = "zhigh.Softmax"(%1) {act_func = "ACT_NONE"} : (tensor<2x32x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x32x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %3 = "zhigh.MatMul"(%2, %V, %none) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<2x32x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x128x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<*xf16>
  return %3 : tensor<*xf16>

  // CHECK-LABEL: test_sdpa3d_no_fuse_transpose_a
  // CHECK-NOT: zhigh.SDPA3D
  // CHECK: zhigh.MatMul
  // CHECK: zhigh.Add
  // CHECK: zhigh.Softmax
  // CHECK: zhigh.MatMul
}

// -----

// Test: Dynamic dimensions, transposeB=0.
func.func @test_sdpa3d_dynamic_no_transpose(
    %Q: tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>,
    %KT: tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>,
    %Mask: tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>,
    %V: tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) ->
    tensor<*xf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "zhigh.MatMul"(%Q, %KT, %none) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %1 = "zhigh.Add"(%0, %Mask) : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %2 = "zhigh.Softmax"(%1) {act_func = "ACT_NONE"} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %3 = "zhigh.MatMul"(%2, %V, %none) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<*xf16>
  return %3 : tensor<*xf16>

  // CHECK-LABEL: test_sdpa3d_dynamic_no_transpose
  // CHECK: "zhigh.SDPA3D"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{transposeKT = 0 : si64}>
  // CHECK-NOT: zhigh.MatMul
}

// -----

// Test: Dynamic dimensions, transposeB=1.
func.func @test_sdpa3d_dynamic_transpose_kt(
    %Q: tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>,
    %K: tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>,
    %Mask: tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>,
    %V: tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) ->
    tensor<*xf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "zhigh.MatMul"(%Q, %K, %none) {transposeA = 0 : si64, transposeB = 1 : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %1 = "zhigh.Add"(%0, %Mask) : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %2 = "zhigh.Softmax"(%1) {act_func = "ACT_NONE"} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %3 = "zhigh.MatMul"(%2, %V, %none) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<*xf16>
  return %3 : tensor<*xf16>

  // CHECK-LABEL: test_sdpa3d_dynamic_transpose_kt
  // CHECK: "zhigh.SDPA3D"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{transposeKT = 1 : si64}>
  // CHECK-NOT: zhigh.MatMul
}
