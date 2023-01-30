// RUN: onnx-mlir-opt --onnx-layer-name-debug-info --mlir-print-debuginfo %s -split-input-file | FileCheck %s

func.func @test_debug_info(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: none) ->  tensor<5x2x965x967xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {onnx_node_name = "foobar", pads = [1, 2, 3, 4], dilations = [1, 1]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x64x64xf32>, none) ->  tensor<5x2x965x967xf32> loc(#convLoc)
  return %0 : tensor<5x2x965x967xf32> loc(#returnLoc)
} loc(#funcLoc)

#funcLoc = loc("foo.mlir":1:1)
#convLoc = loc("foo.mlir":2:1)
#returnLoc = loc("foo.mlir":3:1)

// CHECK: func.func @test_debug_info
// CHECK-NEXT: %{{.*}} = "onnx.Conv"
// CHECK-SAME: onnx_node_name = "[[LAYER_NAME:[a-z]+]]"
// CHECK-SAME: loc(#[[CONV_LOC:[a-z0-9]+]])
// CHECK-NEXT: return {{.*}} loc(#[[RET_LOC:[a-z0-9]+]])
// CHECK-NEXT: } loc(#[[FUNC_LOC:[a-z0-9]+]])

// CHECK-DAG: #[[FUNC_LOC]] = loc("foo.mlir":1:1)
// CHECK-DAG: #[[CONV_LOC]] = loc("[[LAYER_NAME]]"("foo.mlir":2:1))
// CHECK-DAG: #[[RET_LOC]] = loc("foo.mlir":3:1)

// -----
func.func @test_missing_layer_name(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: none) ->  tensor<5x2x965x967xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {pads = [1, 2, 3, 4], dilations = [1, 1]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x64x64xf32>, none) ->  tensor<5x2x965x967xf32> loc(#convLoc)
  return %0 : tensor<5x2x965x967xf32> loc(#returnLoc)
} loc(#funcLoc)

#funcLoc = loc("foo.mlir":1:1)
#convLoc = loc("foo.mlir":2:1)
#returnLoc = loc("foo.mlir":3:1)

// CHECK: func.func @test_missing_layer_name
// CHECK-NEXT: %{{.*}} = "onnx.Conv"
// CHECK-NOT: onnx_node_name
// CHECK-SAME: loc(#[[CONV_LOC:[a-z0-9]+]])
// CHECK-NEXT: return {{.*}} loc(#[[RET_LOC:[a-z0-9]+]])
// CHECK-NEXT: } loc(#[[FUNC_LOC:[a-z0-9]+]])

// CHECK-DAG: #[[FUNC_LOC]] = loc("foo.mlir":1:1)
// CHECK-DAG: #[[CONV_LOC]] = loc("foo.mlir":2:1)
// CHECK-DAG: #[[RET_LOC]] = loc("foo.mlir":3:1)


// -----

// Only the location of the convolution is changed, so convLoc should become a 
// NameLoc and the return should use the original location
func.func @test_duplicate_location(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: none) ->  tensor<5x2x965x967xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {onnx_node_name = "foobar", pads = [1, 2, 3, 4], dilations = [1, 1]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x64x64xf32>, none) ->  tensor<5x2x965x967xf32> loc(#convLoc)
  return %0 : tensor<5x2x965x967xf32> loc(#convLoc)
} loc(#funcLoc)

#funcLoc = loc("foo.mlir":1:1)
#convLoc = loc("foo.mlir":2:1)

// CHECK: #[[RET_LOC:[a-z0-9]+]] = loc("foo.mlir":2:1)

// CHECK: func.func @test_duplicate_location
// CHECK-NEXT: %{{.*}} = "onnx.Conv"
// CHECK-SAME: onnx_node_name = "[[LAYER_NAME:[a-z]+]]"
// CHECK-SAME: loc(#[[CONV_LOC:[a-z0-9]+]])
// CHECK-NEXT: return {{.*}} loc(#[[RET_LOC]])
// CHECK-NEXT: } loc(#[[FUNC_LOC:[a-z0-9]+]])

// CHECK-DAG: #[[FUNC_LOC]] = loc("foo.mlir":1:1)
// CHECK-DAG: #[[CONV_LOC]] = loc("[[LAYER_NAME]]"("foo.mlir":2:1))
