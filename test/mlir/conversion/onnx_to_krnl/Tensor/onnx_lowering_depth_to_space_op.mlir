module {
}


// -----
Warning: [Shape inference, dim 5] the inferred dim (1) is different from the existing dim (4). Use the existing dim instead.
#map = affine_map<()[s0] -> (s0 floordiv 16)>
#map1 = affine_map<()[s0] -> (s0 * 4)>
module {
  func.func private @test_depth_to_space_dynamic_dims(%arg0: memref<1x?x8x?xf32>) -> memref<1x?x32x?xf32> {
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c1 : memref<1x?x8x?xf32>
    %dim_0 = memref.dim %arg0, %c3 : memref<1x?x8x?xf32>
    %0 = affine.apply #map()[%dim]
    %1 = affine.apply #map1()[%dim_0]
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<6xindex>
    krnl.store %c1, %alloc[%c0] : memref<6xindex>
    krnl.store %c4, %alloc[%c1] : memref<6xindex>
    krnl.store %c4, %alloc[%c2] : memref<6xindex>
    krnl.store %0, %alloc[%c3] : memref<6xindex>
    krnl.store %c8, %alloc[%c4] : memref<6xindex>
    krnl.store %dim_0, %alloc[%c5] : memref<6xindex>
    %2 = builtin.unrealized_conversion_cast %alloc : memref<6xindex> to tensor<6xi64>
    %3 = builtin.unrealized_conversion_cast %arg0 : memref<1x?x8x?xf32> to tensor<1x?x8x?xf32>
    %4 = "onnx.Reshape"(%3, %2) <{allowzero = 0 : si64}> : (tensor<1x?x8x?xf32>, tensor<6xi64>) -> tensor<?x?x?x?x?x?xf32>
    %5 = builtin.unrealized_conversion_cast %4 : tensor<?x?x?x?x?x?xf32> to memref<?x?x?x?x?x?xf32>
    %cast = memref.cast %5 : memref<?x?x?x?x?x?xf32> to memref<1x4x4x?x8x?xf32>
    %6 = builtin.unrealized_conversion_cast %cast : memref<1x4x4x?x8x?xf32> to tensor<1x4x4x?x8x?xf32>
    %7 = "onnx.Transpose"(%6) <{perm = [5, 4, 3, 2, 1, 0]}> : (tensor<1x4x4x?x8x?xf32>) -> tensor<1x8x8x4x4x4xf32>
    %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<4xindex>
    krnl.store %c1, %alloc_1[%c0] : memref<4xindex>
    krnl.store %0, %alloc_1[%c1] : memref<4xindex>
    krnl.store %c32, %alloc_1[%c2] : memref<4xindex>
    krnl.store %1, %alloc_1[%c3] : memref<4xindex>
    %8 = builtin.unrealized_conversion_cast %alloc_1 : memref<4xindex> to tensor<4xi64>
    %9 = "onnx.Reshape"(%7, %8) <{allowzero = 0 : si64}> : (tensor<1x8x8x4x4x4xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
    %10 = builtin.unrealized_conversion_cast %9 : tensor<?x?x?x?xf32> to memref<?x?x?x?xf32>
    %cast_2 = memref.cast %10 : memref<?x?x?x?xf32> to memref<1x?x32x?xf32>
    return %cast_2 : memref<1x?x32x?xf32>
  }
}

