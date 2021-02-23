// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// -----

func private @test_geluf32(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Gelu"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_geluf32
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ONE:%.+]] = constant 1.000000e+00 : f32
  // CHECK: [[MINUSBETA:%.+]] = constant -1.702000e+00 : f32
  // CHECK: [[EXP_INPUT:%.+]] = mulf [[LOAD]], [[MINUSBETA]] : f32
  // CHECK: [[EXP_RES:%.+]] = math.exp [[EXP_INPUT]] : f32
  // CHECK: [[DENOMINATOR:%.+]] = addf [[EXP_RES]], [[ONE]] : f32
  // CHECK: [[GELU_RES:%.+]] = divf [[LOAD]], [[DENOMINATOR]] : f32
  // CHECK: krnl.store [[GELU_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_gelubf16(%arg0 : tensor<?x10xbf16>) -> tensor<*xbf16> {
  %0 = "onnx.Gelu"(%arg0) : (tensor<?x10xbf16>) -> tensor<*xbf16>
  "std.return"(%0) : (tensor<*xbf16>) -> ()

  // CHECK-LABEL: test_gelubf16
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xbf16>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xbf16>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xbf16>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xbf16>
  // CHECK: [[ONE:%.+]] = constant 1.000000e+00 : bf16
  // CHECK: [[MINUSBETA:%.+]] = constant -1.703130e+00 : bf16
  // CHECK: [[EXP_INPUT:%.+]] = mulf [[LOAD]], [[MINUSBETA]] : bf16
  // CHECK: [[EXP_RES:%.+]] = math.exp [[EXP_INPUT]] : bf16
  // CHECK: [[DENOMINATOR:%.+]] = addf [[EXP_RES]], [[ONE]] : bf16
  // CHECK: [[GELU_RES:%.+]] = divf [[LOAD]], [[DENOMINATOR]] : bf16
  // CHECK: krnl.store [[GELU_RES]], [[RES]][%arg1, %arg2] : memref<?x10xbf16>
  // CHECK: return [[RES]] : memref<?x10xbf16>
}

// -----

// CHECK: #set = affine_set<()[s0] : (s0 - 1 == 0)>
// CHECK: test_expandf32_2d_to_2d([[ARG:.*]]: {{.*}})
func @test_expandf32_2d_to_2d(%arg0: tensor<3x1xf32>) -> tensor<3x4xf32> {
    %0 = "onnx.Constant"() {value = dense<[3,4]> : tensor<2xi64>} : () -> tensor<2xi64>
    %1 = "onnx.Expand"(%arg0, %0) : (tensor<3x1xf32>, tensor<2xi64>) -> tensor<3x4xf32>
    return %1 : tensor<3x4xf32>
    // CHECK: %[[NEW_SHAPE:.*]] = "krnl.global"()
    // CHECK: %[[NEW_DIM0:.*]] = krnl.load %[[NEW_SHAPE]][%c0]
    // CHECK: %[[NEW_DIM0_INDEX:.*]] = index_cast %[[NEW_DIM0]]
    // CHECK: %[[DIM0:.*]] = dim [[ARG]], %c0{{.*}}
    // CHECK: {{.*}} = affine.if #set()[%[[DIM0]]] -> index {
    // CHECK:           affine.yield %[[NEW_DIM0_INDEX]]
    // CHECK: } else {
    // CHECK:           affine.yield %[[DIM0]]
    // CHECK: }

    // CHECK: %[[NEW_DIM1:.*]] = krnl.load %[[NEW_SHAPE]][%c1{{.*}}]
    // CHECK: %[[NEW_DIM1_INDEX:.*]] = index_cast %[[NEW_DIM1]]
    // CHECK: %[[DIM1:.*]] = dim [[ARG]], %c1{{.*}}
    // CHECK: {{.*}} = affine.if #set()[%[[DIM1]]] -> index {
    // CHECK:           affine.yield %[[NEW_DIM1_INDEX]]
    // CHECK: } else {
    // CHECK:           affine.yield %[[DIM1]]
    // CHECK: }

    // CHECK: %[[DST:.*]] = alloc()
    // CHECK: krnl.iterate({{.*}}) with ({{.*}} -> %[[INDEX0:.*]] = 0 to 3, {{.*}} -> %[[INDEX1:.*]] = 0 to 4) {
    // CHECK: %[[i:.*]] = affine.if #set()[%[[DIM0]]]
    // CHECK:    affine.yield %c0{{.*}}
    // CHECK:  } else {
    // CHECK:    affine.yield %[[INDEX0]]
    // CHECK:  }
    
    // CHECK: %[[j:.*]] = affine.if #set()[%[[DIM1]]]
    // CHECK:    affine.yield %c0{{.*}}
    // CHECK:  } else {
    // CHECK:    affine.yield %[[INDEX1]]
    // CHECK:  }

    // CHECK: %[[TMP:.*]] = krnl.load [[ARG]][%[[i]], %[[j]]]
    // CHECK: krnl.store %[[TMP]], %[[DST]][%[[INDEX0]], %[[INDEX1]]]
    // CHECK: }
    // CHECK: return %[[DST]]
}

// -----

// CHECK: test_expandbf16_2d_to_2d([[ARG:.*]]: {{.*}})
func @test_expandbf16_2d_to_2d(%arg0: tensor<3x1xbf16>) -> tensor<3x4xbf16> {
    %0 = "onnx.Constant"() {value = dense<[3,4]> : tensor<2xi64>} : () -> tensor<2xi64>
    %1 = "onnx.Expand"(%arg0, %0) : (tensor<3x1xbf16>, tensor<2xi64>) -> tensor<3x4xbf16>
    return %1 : tensor<3x4xbf16>
    // CHECK: %[[NEW_SHAPE:.*]] = "krnl.global"()
    // CHECK: %[[NEW_DIM0:.*]] = krnl.load %[[NEW_SHAPE]][%c0]
    // CHECK: %[[NEW_DIM0_INDEX:.*]] = index_cast %[[NEW_DIM0]]
    // CHECK: %[[DIM0:.*]] = dim [[ARG]], %c0{{.*}}
    // CHECK: {{.*}} = affine.if #set()[%[[DIM0]]] -> index {
    // CHECK:           affine.yield %[[NEW_DIM0_INDEX]]
    // CHECK: } else {
    // CHECK:           affine.yield %[[DIM0]]
    // CHECK: }

    // CHECK: %[[NEW_DIM1:.*]] = krnl.load %[[NEW_SHAPE]][%c1{{.*}}]
    // CHECK: %[[NEW_DIM1_INDEX:.*]] = index_cast %[[NEW_DIM1]]
    // CHECK: %[[DIM1:.*]] = dim [[ARG]], %c1{{.*}}
    // CHECK: {{.*}} = affine.if #set()[%[[DIM1]]] -> index {
    // CHECK:           affine.yield %[[NEW_DIM1_INDEX]]
    // CHECK: } else {
    // CHECK:           affine.yield %[[DIM1]]
    // CHECK: }

    // CHECK: %[[DST:.*]] = alloc()
    // CHECK: krnl.iterate({{.*}}) with ({{.*}} -> %[[INDEX0:.*]] = 0 to 3, {{.*}} -> %[[INDEX1:.*]] = 0 to 4) {
    // CHECK: %[[i:.*]] = affine.if #set()[%[[DIM0]]]
    // CHECK:    affine.yield %c0{{.*}}
    // CHECK:  } else {
    // CHECK:    affine.yield %[[INDEX0]]
    // CHECK:  }
    
    // CHECK: %[[j:.*]] = affine.if #set()[%[[DIM1]]]
    // CHECK:    affine.yield %c0{{.*}}
    // CHECK:  } else {
    // CHECK:    affine.yield %[[INDEX1]]
    // CHECK:  }

    // CHECK: %[[TMP:.*]] = krnl.load [[ARG]][%[[i]], %[[j]]]
    // CHECK: krnl.store %[[TMP]], %[[DST]][%[[INDEX0]], %[[INDEX1]]]
    // CHECK: }
    // CHECK: return %[[DST]]
}

// -----

// CHECK: test_expandbf16_2d_to_3d([[ARG:.*]]: {{.*}})
func @test_expandbf16_2d_to_3d(%arg0: tensor<3x1xbf16>) -> tensor<?x?x?xbf16> {
    %0 = "onnx.Constant"() {value = dense<[2, 1, 6]> : tensor<3xi64>} : () -> tensor<3xi64>
    %1 = "onnx.Expand"(%arg0, %0) : (tensor<3x1xbf16>, tensor<3xi64>) -> tensor<?x?x?xbf16>
    return %1 : tensor<?x?x?xbf16>
    // CHECK: %[[NEW_SHAPE:.*]] = "krnl.global"()
    // CHECK: %[[NEW_DIM1:.*]] = krnl.load %[[NEW_SHAPE]][%c1{{.*}}]
    // CHECK: %[[NEW_DIM1_INDEX:.*]] = index_cast %[[NEW_DIM1]]
    // CHECK: %[[DIM0:.*]] = dim [[ARG]], %c0{{.*}}
    // CHECK: {{.*}} = affine.if #set()[%[[DIM0]]] -> index {
    // CHECK:           affine.yield %[[NEW_DIM1_INDEX]]
    // CHECK: } else {
    // CHECK:           affine.yield %[[DIM0]]
    // CHECK: }

    // CHECK: %[[NEW_DIM2:.*]] = krnl.load %[[NEW_SHAPE]][%c2{{.*}}]
    // CHECK: %[[NEW_DIM2_INDEX:.*]] = index_cast %[[NEW_DIM2]]
    // CHECK: %[[DIM1:.*]] = dim [[ARG]], %c1{{.*}}
    // CHECK: {{.*}} = affine.if #set()[%[[DIM1]]] -> index {
    // CHECK:           affine.yield %[[NEW_DIM2_INDEX]]
    // CHECK: } else {
    // CHECK:           affine.yield %[[DIM1]]
    // CHECK: }

    // CHECK: %[[DST:.*]] = alloc()
    // CHECK: krnl.iterate({{.*}}) with ({{.*}} -> %[[INDEX0:.*]] = 0 to 2, {{.*}} -> %[[INDEX1:.*]] = 0 to 3, {{.*}} -> %[[INDEX2:.*]] = 0 to 6) {
    // CHECK: %[[i:.*]] = affine.if #set()[%[[DIM0]]]
    // CHECK:    affine.yield %c0{{.*}}
    // CHECK:  } else {
    // CHECK:    affine.yield %[[INDEX1]]
    // CHECK:  }

    // CHECK: %[[j:.*]] = affine.if #set()[%[[DIM1]]]
    // CHECK:    affine.yield %c0{{.*}}
    // CHECK:  } else {
    // CHECK:    affine.yield %[[INDEX2]]
    // CHECK:  }

    // CHECK: %[[TMP:.*]] = krnl.load [[ARG]][%[[i]], %[[j]]]
    // CHECK: krnl.store %[[TMP]], %[[DST]][%[[INDEX0]], %[[INDEX1]], %[[INDEX2]]]
    // CHECK: }
    // CHECK: return %[[DST]]
}

// -----

// CHECK: test_expandbf16_unknown_sized_input([[ARG:.*]]: {{.*}})
func @test_expandbf16_unknown_sized_input(%arg0: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
    %0 = "onnx.Constant"() {value = dense<[1, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
    %1 = "onnx.Expand"(%arg0, %0) : (tensor<?x?xbf16>, tensor<2xi64>) -> tensor<?x?xbf16>
    return %1 : tensor<?x?xbf16>

    // CHECK: %[[NEW_SHAPE:.*]] = "krnl.global"()
    // CHECK: %[[NEW_DIM0:.*]] = krnl.load %[[NEW_SHAPE]][%c0]
    // CHECK: %[[NEW_DIM0_INDEX:.*]] = index_cast %[[NEW_DIM0]]
    // CHECK: %[[DIM0:.*]] = dim [[ARG]], %c0{{.*}}
    // CHECK: %[[DYN_SIZE:.*]] = affine.if #set()[%[[DIM0]]] -> index {
    // CHECK:           affine.yield %[[NEW_DIM0_INDEX]]
    // CHECK: } else {
    // CHECK:           affine.yield %[[DIM0]]
    // CHECK: }

    // CHECK: %[[NEW_DIM1:.*]] = krnl.load %[[NEW_SHAPE]][%c1{{.*}}]
    // CHECK: %[[NEW_DIM1_INDEX:.*]] = index_cast %[[NEW_DIM1]]
    // CHECK: %[[DIM1:.*]] = dim [[ARG]], %c1{{.*}}
    // CHECK: {{.*}} = affine.if #set()[%[[DIM1]]] -> index {
    // CHECK:           affine.yield %[[NEW_DIM1_INDEX]]
    // CHECK: } else {
    // CHECK:           affine.yield %[[DIM1]]
    // CHECK: }

    // CHECK: %[[DST:.*]] = alloc(%[[DYN_SIZE]])
    // CHECK: %[[DYN_SIZE1:.*]] = dim %[[DST]], %c0{{.*}}
    // CHECK: krnl.iterate({{.*}}) with ({{.*}} -> %[[INDEX0:.*]] = 0 to %[[DYN_SIZE1]], {{.*}} -> %[[INDEX1:.*]] = 0 to 3) {
    // CHECK: %[[i:.*]] = affine.if #set()[%[[DIM0]]]
    // CHECK:    affine.yield %c0{{.*}}
    // CHECK:  } else {
    // CHECK:    affine.yield %[[INDEX0]]
    // CHECK:  }
    
    // CHECK: %[[j:.*]] = affine.if #set()[%[[DIM1]]]
    // CHECK:    affine.yield %c0{{.*}}
    // CHECK:  } else {
    // CHECK:    affine.yield %[[INDEX1]]
    // CHECK:  }

    // CHECK: %[[TMP:.*]] = krnl.load [[ARG]][%[[i]], %[[j]]]
    // CHECK: krnl.store %[[TMP]], %[[DST]][%[[INDEX0]], %[[INDEX1]]]
    // CHECK: }
    // CHECK: return %[[DST]]
}
