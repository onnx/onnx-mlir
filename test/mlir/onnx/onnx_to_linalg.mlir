// RUN: onnx-mlir-opt --convert-onnx-to-linalg -verify-diagnostics -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func @test_lowering(
// CHECK-SAME:                        %[[VAL_0:.*]]: tensor<16x128xf32>,
// CHECK-SAME:                        %[[VAL_1:.*]]: tensor<128x32xf32>) {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() : memref<16x32xf32>
// CHECK:           linalg.matmul ins(%[[VAL_0]], %[[VAL_1]] : tensor<16x128xf32>, tensor<128x32xf32>) outs(%[[VAL_2]] : memref<16x32xf32>)
// CHECK:           memref.dealloc %[[VAL_2]] : memref<16x32xf32>
// CHECK:           return
// CHECK:         }
func @test_lowering(%arg0: tensor<16x128xf32>, %arg1: tensor<128x32xf32>) -> () {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<16x128xf32>, tensor<128x32xf32>) -> tensor<16x32xf32>
  return
}

// -----

func @test_invalid(%arg0: tensor<16x?xf32>, %arg1: tensor<?x32xf32>) -> () {
// expected-warning@below {{This operation takes tensors with unsupported by current target sizes}}
// expected-error@below {{failed to legalize operation 'onnx.MatMul' that was explicitly marked illegal}}
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<16x?xf32>, tensor<?x32xf32>) -> tensor<16x32xf32>
  return
}

// -----

func @test_invalid2(%arg0: tensor<16x33xf32>, %arg1: tensor<33x32xf32>) -> () {
// expected-warning@below {{This operation takes tensors with unsupported by current target sizes}}
// expected-error@below {{failed to legalize operation 'onnx.MatMul' that was explicitly marked illegal}}
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<16x33xf32>, tensor<33x32xf32>) -> tensor<16x32xf32>
  return
}

// -----

func @test_invalid3(%arg0: tensor<?x32xf32>, %arg1: tensor<32x32xf32>) -> () {
// expected-warning@below {{This operation produces unsupported by current target dynamically sized tensor}}
// expected-error@below {{failed to legalize operation 'onnx.MatMul' that was explicitly marked illegal}}
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x32xf32>, tensor<32x32xf32>) -> tensor<?x32xf32>
  return
}

// -----

// CHECK-LABEL: test_3Dx3D_matmul
func @test_3Dx3D_matmul(%arg0: tensor<10x512x3072xbf16>, %arg1: tensor<10x3072x3072xbf16>) -> () {
  %8 = "onnx.MatMul"(%arg0, %arg1) {onnx_node_name = "MatMul0"} : (tensor<10x512x3072xbf16>, tensor<10x3072x3072xbf16>) -> tensor<10x512x3072xbf16>
  return
  // CHECK:   %[[VAL_0:.*]] = memref.alloc() : memref<10x512x3072xbf16>
  // CHECK:   %[[LOOP:.*]] = krnl.define_loops 1
  // CHECK:   krnl.iterate(%[[LOOP]]) with (%[[LOOP]] -> %[[I:.*]] = 0 to 10) {
  // CHECK:     %[[MEMREF_0:.*]] = krnl.dummy_cast %arg0 : (tensor<10x512x3072xbf16>) -> memref<10x512x3072xbf16>
  // CHECK:     %[[A:.*]] = memref.subview %[[MEMREF_0]][%[[I]], 0, 0] [1, 512, 3072] [1, 1, 1] : memref<10x512x3072xbf16> to memref<512x3072xbf16, #map>
  // CHECK:     %[[MEMREF_1:.*]] = krnl.dummy_cast %arg1 : (tensor<10x3072x3072xbf16>) -> memref<10x3072x3072xbf16>
  // CHECK:     %[[B:.*]] = memref.subview %[[MEMREF_1]][%[[I]], 0, 0] [1, 3072, 3072] [1, 1, 1] : memref<10x3072x3072xbf16> to memref<3072x3072xbf16, #map>
  // CHECK:     %[[RESULT:.*]] = memref.subview %[[VAL_0]][%[[I]], 0, 0] [1, 512, 3072] [1, 1, 1] : memref<10x512x3072xbf16> to memref<512x3072xbf16, #map>
  // CHECK:     linalg.matmul ins(%[[A]], %[[B]] : memref<512x3072xbf16, #map>, memref<3072x3072xbf16, #map>) outs(%[[RESULT]] : memref<512x3072xbf16, #map>)
  // CHECK:   }
  // CHECK:   memref.dealloc %[[VAL_0]] : memref<10x512x3072xbf16>
  // CHECK:   return
  // CHECK: }
}

// -----

// CHECK-LABEL: test_3Dx2D_matmul
func @test_3Dx2D_matmul(%arg0: tensor<10x512x3072xbf16>, %arg1: tensor<3072x3072xbf16>) -> () {
  %8 = "onnx.MatMul"(%arg0, %arg1) {onnx_node_name = "MatMul0"} : (tensor<10x512x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<10x512x3072xbf16>
  return
  // CHECK:   %[[VAL_0:.*]] = memref.alloc() : memref<10x512x3072xbf16>
  // CHECK:   %[[LOOP:.*]] = krnl.define_loops 1
  // CHECK:   krnl.iterate(%[[LOOP]]) with (%[[LOOP]] -> %[[I:.*]] = 0 to 10) {
  // CHECK:     %[[MEMREF_0:.*]] = krnl.dummy_cast %arg0 : (tensor<10x512x3072xbf16>) -> memref<10x512x3072xbf16>
  // CHECK:     %[[A:.*]] = memref.subview %[[MEMREF_0]][%[[I]], 0, 0] [1, 512, 3072] [1, 1, 1] : memref<10x512x3072xbf16> to memref<512x3072xbf16, #map>
  // CHECK:     %[[RESULT:.*]] = memref.subview %[[VAL_0]][%[[I]], 0, 0] [1, 512, 3072] [1, 1, 1] : memref<10x512x3072xbf16> to memref<512x3072xbf16, #map>
  // CHECK:     linalg.matmul ins(%[[A]], %arg1 : memref<512x3072xbf16, #map>, tensor<3072x3072xbf16>) outs(%[[RESULT]] : memref<512x3072xbf16, #map>)
  // CHECK:   }
  // CHECK:   memref.dealloc %[[VAL_0]] : memref<10x512x3072xbf16>
  // CHECK:   return
  // CHECK: }
}

// -----

// CHECK-LABEL: test_4Dx4D_matmul
func @test_4Dx4D_matmul(%arg0: tensor<10x10x512x3072xbf16>, %arg1: tensor<10x10x3072x3072xbf16>) -> () {
  %8 = "onnx.MatMul"(%arg0, %arg1) {onnx_node_name = "MatMul0"} : (tensor<10x10x512x3072xbf16>, tensor<10x10x3072x3072xbf16>) -> tensor<10x10x512x3072xbf16>
  return
  // CHECK:   %[[VAL_0:.*]] = memref.alloc() : memref<10x10x512x3072xbf16>
  // CHECK:   %[[LOOP:.*]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate(%[[LOOP]]#0, %[[LOOP]]#1) with (%[[LOOP]]#0 -> %[[I:.*]] = 0 to 10, %[[LOOP]]#1 -> %[[J:.*]] = 0 to 10) {
  // CHECK:     %[[MEMREF_0:.*]] = krnl.dummy_cast %arg0 : (tensor<10x10x512x3072xbf16>) -> memref<10x10x512x3072xbf16>
  // CHECK:     %[[A:.*]] = memref.subview %[[MEMREF_0]][%[[I]], %[[J]], 0, 0] [1, 1, 512, 3072] [1, 1, 1, 1] : memref<10x10x512x3072xbf16> to memref<512x3072xbf16, #map>
  // CHECK:     %[[MEMREF_1:.*]] = krnl.dummy_cast %arg1 : (tensor<10x10x3072x3072xbf16>) -> memref<10x10x3072x3072xbf16>
  // CHECK:     %[[B:.*]] = memref.subview %[[MEMREF_1]][%[[I]], %[[J]], 0, 0] [1, 1, 3072, 3072] [1, 1, 1, 1] : memref<10x10x3072x3072xbf16> to memref<3072x3072xbf16, #map>
  // CHECK:     %[[RESULT:.*]] = memref.subview %[[VAL_0]][%[[I]], %[[J]], 0, 0] [1, 1, 512, 3072] [1, 1, 1, 1] : memref<10x10x512x3072xbf16> to memref<512x3072xbf16, #map>
  // CHECK:   linalg.matmul ins(%[[A]], %[[B]] : memref<512x3072xbf16, #map>, memref<3072x3072xbf16, #map>) outs(%[[RESULT]] : memref<512x3072xbf16, #map>)
  // CHECK:   }
  // CHECK:   memref.dealloc %[[VAL_0]] : memref<10x10x512x3072xbf16>
  // CHECK:   return
  // CHECK: }
}

// -----

// CHECK-LABEL: test_4Dx2D_matmul
func @test_4Dx2D_matmul(%arg0: tensor<10x10x512x3072xbf16>, %arg1: tensor<3072x3072xbf16>) -> () {
  %8 = "onnx.MatMul"(%arg0, %arg1) {onnx_node_name = "MatMul0"} : (tensor<10x10x512x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<10x10x512x3072xbf16>
  return
  // CHECK:   %[[VAL_0:.*]] = memref.alloc() : memref<10x10x512x3072xbf16>
  // CHECK:   %[[LOOP:.*]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate(%[[LOOP]]#0, %[[LOOP]]#1) with (%[[LOOP]]#0 -> %[[I:.*]] = 0 to 10, %[[LOOP]]#1 -> %[[J:.*]] = 0 to 10) {
  // CHECK:     %[[MEMREF_0:.*]] = krnl.dummy_cast %arg0 : (tensor<10x10x512x3072xbf16>) -> memref<10x10x512x3072xbf16>
  // CHECK:     %[[A:.*]] = memref.subview %[[MEMREF_0]][%[[I]], %[[J]], 0, 0] [1, 1, 512, 3072] [1, 1, 1, 1] : memref<10x10x512x3072xbf16> to memref<512x3072xbf16, #map>
  // CHECK:     %[[RESULT:.*]] = memref.subview %[[VAL_0]][%[[I]], %[[J]], 0, 0] [1, 1, 512, 3072] [1, 1, 1, 1] : memref<10x10x512x3072xbf16> to memref<512x3072xbf16, #map>
  // CHECK:     linalg.matmul ins(%[[A]], %arg1 : memref<512x3072xbf16, #map>, tensor<3072x3072xbf16>) outs(%[[RESULT]] : memref<512x3072xbf16, #map>)
  // CHECK:   }
  // CHECK:   memref.dealloc %[[VAL_0]] : memref<10x10x512x3072xbf16>
  // CHECK:   return
  // CHECK: }
}

// -----

// CHECK-LABEL: test_3Dx4D_matmul
func @test_3Dx4D_matmul(%arg0: tensor<10x3072x3072xbf16>, %arg1: tensor<10x10x3072x512xbf16>) -> () {
  %8 = "onnx.MatMul"(%arg0, %arg1) {onnx_node_name = "MatMul0"} : (tensor<10x3072x3072xbf16>, tensor<10x10x3072x512xbf16>) -> tensor<10x10x3072x512xbf16>
  return
  // CHECK:   %[[VAL_0:.*]] = memref.alloc() : memref<10x10x3072x512xbf16>
  // CHECK:   %[[LOOP:.*]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate(%[[LOOP]]#0, %[[LOOP]]#1) with (%[[LOOP]]#0 -> %[[I:.*]] = 0 to 10, %[[LOOP]]#1 -> %[[J:.*]] = 0 to 10) {
  // CHECK:     %[[MEMREF_0:.*]] = krnl.dummy_cast %arg0 : (tensor<10x3072x3072xbf16>) -> memref<10x3072x3072xbf16>
  // CHECK:     %[[A:.*]] = memref.subview %[[MEMREF_0]][%[[J]], 0, 0] [1, 3072, 3072] [1, 1, 1] : memref<10x3072x3072xbf16> to memref<3072x3072xbf16, #map0>
  // CHECK:     %[[MEMREF_1:.*]] = krnl.dummy_cast %arg1 : (tensor<10x10x3072x512xbf16>) -> memref<10x10x3072x512xbf16>
  // CHECK:     %[[B:.*]] = memref.subview %[[MEMREF_1]][%[[I]], %[[J]], 0, 0] [1, 1, 3072, 512] [1, 1, 1, 1] : memref<10x10x3072x512xbf16> to memref<3072x512xbf16, #map1>
  // CHECK:     %[[RESULT:.*]] = memref.subview %[[VAL_0]][%[[I]], %[[J]], 0, 0] [1, 1, 3072, 512] [1, 1, 1, 1] : memref<10x10x3072x512xbf16> to memref<3072x512xbf16, #map1>
  // CHECK:     linalg.matmul ins(%[[A]], %[[B]] : memref<3072x3072xbf16, #map0>, memref<3072x512xbf16, #map1>) outs(%[[RESULT]] : memref<3072x512xbf16, #map1>)
  // CHECK:   }
  // CHECK:   memref.dealloc %[[VAL_0]] : memref<10x10x3072x512xbf16>
  // CHECK:   return
  // CHECK: }
}

// -----

// CHECK-LABEL: test_4Dx4D_matmul_broadcast
func @test_4Dx4D_matmul_broadcast(%arg0: tensor<10x10x512x3072xbf16>, %arg1: tensor<1x1x3072x3072xbf16>) -> () {
  %8 = "onnx.MatMul"(%arg0, %arg1) {onnx_node_name = "MatMul0"} : (tensor<10x10x512x3072xbf16>, tensor<1x1x3072x3072xbf16>) -> tensor<10x10x512x3072xbf16>
  return
  // CHECK:   %[[VAL_0:.*]] = memref.alloc() : memref<10x10x512x3072xbf16>
  // CHECK:   %[[LOOP:.*]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate(%[[LOOP]]#0, %[[LOOP]]#1) with (%[[LOOP]]#0 -> %[[I:.*]] = 0 to 1, %[[LOOP]]#1 -> %[[J:.*]] = 0 to 1) {
  // CHECK:     %[[MEMREF_0:.*]] = krnl.dummy_cast %arg0 : (tensor<10x10x512x3072xbf16>) -> memref<10x10x512x3072xbf16>
  // CHECK:     %[[A:.*]] = memref.subview %[[MEMREF_0]][%[[I]], %[[J]], 0, 0] [1, 1, 512, 3072] [1, 1, 1, 1] : memref<10x10x512x3072xbf16> to memref<512x3072xbf16, #map0>
  // CHECK:     %[[MEMREF_1:.*]] = krnl.dummy_cast %arg1 : (tensor<1x1x3072x3072xbf16>) -> memref<1x1x3072x3072xbf16>
  // CHECK:     %[[B:.*]] = memref.subview %[[MEMREF_1]][0, 0, 0, 0] [1, 1, 3072, 3072] [1, 1, 1, 1] : memref<1x1x3072x3072xbf16> to memref<3072x3072xbf16, #map1>
  // CHECK:     %[[RESULT:.*]] = memref.subview %[[VAL_0]][%[[I]], %[[J]], 0, 0] [1, 1, 512, 3072] [1, 1, 1, 1] : memref<10x10x512x3072xbf16> to memref<512x3072xbf16, #map0>
  // CHECK:     linalg.matmul ins(%[[A]], %[[B]] : memref<512x3072xbf16, #map0>, memref<3072x3072xbf16, #map1>) outs(%[[RESULT]] : memref<512x3072xbf16, #map0>)
  // CHECK:   }
  // CHECK:   memref.dealloc %[[VAL_0]] : memref<10x10x512x3072xbf16>
  // CHECK:   return
  // CHECK: }
}

// -----

// CHECK-LABEL: test_4Dx4D_matmul_broadcast2
func @test_4Dx4D_matmul_broadcast2(%arg0: tensor<10x10x512x3072xbf16>, %arg1: tensor<10x1x3072x3072xbf16>) -> () {
  %8 = "onnx.MatMul"(%arg0, %arg1) {onnx_node_name = "MatMul0"} : (tensor<10x10x512x3072xbf16>, tensor<10x1x3072x3072xbf16>) -> tensor<10x10x512x3072xbf16>
  return
  // CHECK:   %[[VAL_0:.*]] = memref.alloc() : memref<10x10x512x3072xbf16>
  // CHECK:   %[[LOOP:.*]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate(%[[LOOP]]#0, %[[LOOP]]#1) with (%[[LOOP]]#0 -> %[[I:.*]] = 0 to 10, %[[LOOP]]#1 -> %[[J:.*]] = 0 to 1) {
  // CHECK:     %[[MEMREF_0:.*]] = krnl.dummy_cast %arg0 : (tensor<10x10x512x3072xbf16>) -> memref<10x10x512x3072xbf16>
  // CHECK:     %[[A:.*]] = memref.subview %[[MEMREF_0]][%[[I]], %[[J]], 0, 0] [1, 1, 512, 3072] [1, 1, 1, 1] : memref<10x10x512x3072xbf16> to memref<512x3072xbf16, #map>
  // CHECK:     %[[MEMREF_1:.*]] = krnl.dummy_cast %arg1 : (tensor<10x1x3072x3072xbf16>) -> memref<10x1x3072x3072xbf16>
  // CHECK:     %[[B:.*]] = memref.subview %[[MEMREF_1]][%[[I]], 0, 0, 0] [1, 1, 3072, 3072] [1, 1, 1, 1] : memref<10x1x3072x3072xbf16> to memref<3072x3072xbf16, #map>
  // CHECK:     %[[RESULT:.*]] = memref.subview %[[VAL_0]][%[[I]], %[[J]], 0, 0] [1, 1, 512, 3072] [1, 1, 1, 1] : memref<10x10x512x3072xbf16> to memref<512x3072xbf16, #map>
  // CHECK:     linalg.matmul ins(%[[A]], %[[B]] : memref<512x3072xbf16, #map>, memref<3072x3072xbf16, #map>) outs(%[[RESULT]] : memref<512x3072xbf16, #map>)
  // CHECK:   }
  // CHECK:   memref.dealloc %[[VAL_0]] : memref<10x10x512x3072xbf16>
  // CHECK:   return
  // CHECK: }
}
