// RUN: onnx-mlir-opt --recompose-onnx %s -split-input-file | FileCheck %s

// PyTorch `unbind` of a fused QKV projection: Reshape, then per output a
// unit Slice, a Squeeze and a head Transpose. The Slices become one Split and
// the Transpose moves above it, giving Reshape -> Transpose -> Split ->
// Squeeze, the chain the NNPA unstick-split-heads fusion recognizes.
func.func @unbind_qkv(%arg0: tensor<1x?x1152xf32>) -> (tensor<1x12x?x32xf32>, tensor<1x12x?x32xf32>, tensor<1x12x?x32xf32>) {
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %c2 = onnx.Constant dense<2> : tensor<1xi64>
  %c3 = onnx.Constant dense<3> : tensor<1xi64>
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %cm3 = onnx.Constant dense<-3> : tensor<1xi64>
  %shape = onnx.Constant dense<[1, -1, 3, 12, 32]> : tensor<5xi64>
  %r = "onnx.Reshape"(%arg0, %shape) <{allowzero = 0 : si64}> : (tensor<1x?x1152xf32>, tensor<5xi64>) -> tensor<1x?x3x12x32xf32>
  %s0 = "onnx.Slice"(%r, %c0, %c1, %cm3, %c1) : (tensor<1x?x3x12x32xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x?x1x12x32xf32>
  %q0 = "onnx.Squeeze"(%s0, %c2) : (tensor<1x?x1x12x32xf32>, tensor<1xi64>) -> tensor<1x?x12x32xf32>
  %s1 = "onnx.Slice"(%r, %c1, %c2, %cm3, %c1) : (tensor<1x?x3x12x32xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x?x1x12x32xf32>
  %q1 = "onnx.Squeeze"(%s1, %c2) : (tensor<1x?x1x12x32xf32>, tensor<1xi64>) -> tensor<1x?x12x32xf32>
  %s2 = "onnx.Slice"(%r, %c2, %c3, %cm3, %c1) : (tensor<1x?x3x12x32xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x?x1x12x32xf32>
  %q2 = "onnx.Squeeze"(%s2, %c2) : (tensor<1x?x1x12x32xf32>, tensor<1xi64>) -> tensor<1x?x12x32xf32>
  %t0 = "onnx.Transpose"(%q0) <{perm = [0, 2, 1, 3]}> : (tensor<1x?x12x32xf32>) -> tensor<1x12x?x32xf32>
  %t1 = "onnx.Transpose"(%q1) <{perm = [0, 2, 1, 3]}> : (tensor<1x?x12x32xf32>) -> tensor<1x12x?x32xf32>
  %t2 = "onnx.Transpose"(%q2) <{perm = [0, 2, 1, 3]}> : (tensor<1x?x12x32xf32>) -> tensor<1x12x?x32xf32>
  return %t0, %t1, %t2 : tensor<1x12x?x32xf32>, tensor<1x12x?x32xf32>, tensor<1x12x?x32xf32>

// CHECK-LABEL:  func.func @unbind_qkv
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x1152xf32>) -> (tensor<1x12x?x32xf32>, tensor<1x12x?x32xf32>, tensor<1x12x?x32xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, -1, 3, 12, 32]> : tensor<5xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_2_]]) <{allowzero = 0 : si64}> : (tensor<1x?x1152xf32>, tensor<5xi64>) -> tensor<1x?x3x12x32xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Transpose"([[VAR_3_]]) <{perm = [0, 3, 2, 1, 4]}> : (tensor<1x?x3x12x32xf32>) -> tensor<1x12x3x?x32xf32>
// CHECK:           [[VAR_5_:%.+]]:3 = "onnx.Split"([[VAR_4_]], [[VAR_1_]]) <{axis = 2 : si64}> : (tensor<1x12x3x?x32xf32>, tensor<3xi64>) -> (tensor<1x12x1x?x32xf32>, tensor<1x12x1x?x32xf32>, tensor<1x12x1x?x32xf32>)
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Squeeze"([[VAR_5_]]#0, [[VAR_0_]]) : (tensor<1x12x1x?x32xf32>, tensor<1xi64>) -> tensor<1x12x?x32xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Squeeze"([[VAR_5_]]#1, [[VAR_0_]]) : (tensor<1x12x1x?x32xf32>, tensor<1xi64>) -> tensor<1x12x?x32xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Squeeze"([[VAR_5_]]#2, [[VAR_0_]]) : (tensor<1x12x1x?x32xf32>, tensor<1xi64>) -> tensor<1x12x?x32xf32>
// CHECK:           return [[VAR_6_]], [[VAR_7_]], [[VAR_8_]] : tensor<1x12x?x32xf32>, tensor<1x12x?x32xf32>, tensor<1x12x?x32xf32>
}

// -----

// The Slices are not an unbind (the last one is missing), so nothing changes.
func.func @slices_do_not_tile(%arg0: tensor<2x3x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %c2 = onnx.Constant dense<2> : tensor<1xi64>
  %c1d = onnx.Constant dense<1> : tensor<1xi64>
  %s0 = "onnx.Slice"(%arg0, %c0, %c1, %c1d, %c1) : (tensor<2x3x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x1x4xf32>
  %s1 = "onnx.Slice"(%arg0, %c1, %c2, %c1d, %c1) : (tensor<2x3x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x1x4xf32>
  %q0 = "onnx.Squeeze"(%s0, %c1d) : (tensor<2x1x4xf32>, tensor<1xi64>) -> tensor<2x4xf32>
  %q1 = "onnx.Squeeze"(%s1, %c1d) : (tensor<2x1x4xf32>, tensor<1xi64>) -> tensor<2x4xf32>
  return %q0, %q1 : tensor<2x4xf32>, tensor<2x4xf32>

// CHECK-LABEL:  func.func @slices_do_not_tile
// CHECK-NOT:       onnx.Split
// CHECK:           "onnx.Slice"
// CHECK:           "onnx.Slice"
}

// -----

// Different Transpose perms after the Split: the Transpose is not hoisted, the
// Split itself (from the unbind) stays.
func.func @different_perms(%arg0: tensor<2x3x4x5xf32>) -> (tensor<2x5x4xf32>, tensor<2x4x5xf32>, tensor<2x5x4xf32>) {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %c2 = onnx.Constant dense<2> : tensor<1xi64>
  %c3 = onnx.Constant dense<3> : tensor<1xi64>
  %s0 = "onnx.Slice"(%arg0, %c0, %c1, %c1, %c1) : (tensor<2x3x4x5xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x1x4x5xf32>
  %s1 = "onnx.Slice"(%arg0, %c1, %c2, %c1, %c1) : (tensor<2x3x4x5xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x1x4x5xf32>
  %s2 = "onnx.Slice"(%arg0, %c2, %c3, %c1, %c1) : (tensor<2x3x4x5xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x1x4x5xf32>
  %q0 = "onnx.Squeeze"(%s0, %c1) : (tensor<2x1x4x5xf32>, tensor<1xi64>) -> tensor<2x4x5xf32>
  %q1 = "onnx.Squeeze"(%s1, %c1) : (tensor<2x1x4x5xf32>, tensor<1xi64>) -> tensor<2x4x5xf32>
  %q2 = "onnx.Squeeze"(%s2, %c1) : (tensor<2x1x4x5xf32>, tensor<1xi64>) -> tensor<2x4x5xf32>
  %t0 = "onnx.Transpose"(%q0) <{perm = [0, 2, 1]}> : (tensor<2x4x5xf32>) -> tensor<2x5x4xf32>
  %t2 = "onnx.Transpose"(%q2) <{perm = [0, 2, 1]}> : (tensor<2x4x5xf32>) -> tensor<2x5x4xf32>
  return %t0, %q1, %t2 : tensor<2x5x4xf32>, tensor<2x4x5xf32>, tensor<2x5x4xf32>

// CHECK-LABEL:  func.func @different_perms
// CHECK:           [[SPLIT_:%.+]]:3 = "onnx.Split"(%arg0, {{.*}}) <{axis = 1 : si64}>
// CHECK-NOT:       "onnx.Slice"
// CHECK:           "onnx.Transpose"
// CHECK:           "onnx.Transpose"
}

// -----

// Other users of the data do not prevent the rewrite: the extra Slice (a
// different step) and the Neg keep reading the Reshape result.
func.func @other_users(%arg0: tensor<2x3x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x3x4xf32>, tensor<2x2x4xf32>) {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %c2 = onnx.Constant dense<2> : tensor<1xi64>
  %c3 = onnx.Constant dense<3> : tensor<1xi64>
  %s0 = "onnx.Slice"(%arg0, %c0, %c1, %c1, %c1) : (tensor<2x3x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x1x4xf32>
  %s1 = "onnx.Slice"(%arg0, %c1, %c2, %c1, %c1) : (tensor<2x3x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x1x4xf32>
  %s2 = "onnx.Slice"(%arg0, %c2, %c3, %c1, %c1) : (tensor<2x3x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x1x4xf32>
  %q0 = "onnx.Squeeze"(%s0, %c1) : (tensor<2x1x4xf32>, tensor<1xi64>) -> tensor<2x4xf32>
  %q1 = "onnx.Squeeze"(%s1, %c1) : (tensor<2x1x4xf32>, tensor<1xi64>) -> tensor<2x4xf32>
  %q2 = "onnx.Squeeze"(%s2, %c1) : (tensor<2x1x4xf32>, tensor<1xi64>) -> tensor<2x4xf32>
  %neg = "onnx.Neg"(%arg0) : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %every2 = "onnx.Slice"(%arg0, %c0, %c3, %c1, %c2) : (tensor<2x3x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x2x4xf32>
  return %q0, %q1, %q2, %neg, %every2 : tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x3x4xf32>, tensor<2x2x4xf32>

// CHECK-LABEL:  func.func @other_users
// CHECK:           [[SPLIT_:%.+]]:3 = "onnx.Split"(%arg0, {{.*}}) <{axis = 1 : si64}>
// CHECK-DAG:       "onnx.Neg"(%arg0)
// CHECK-DAG:       "onnx.Slice"(%arg0, {{.*}}) : {{.*}} -> tensor<2x2x4xf32>
}
