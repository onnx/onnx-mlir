#map = affine_map<(d0) -> (d0)>
module {
  func.func @test_transpose_lowered_to_a_view_op(%arg0: memref<?x1x1x384xf32>) -> memref<384x1x1x?xf32> {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x1x1x384xf32>
    %alloc = memref.alloc(%dim) {alignment = 16 : i64} : memref<384x1x1x?xf32>
    %0:4 = krnl.define_loops 4
    %dim_0 = memref.dim %arg0, %c0 : memref<?x1x1x384xf32>
    krnl.iterate(%0#0, %0#1, %0#2, %0#3) with (%0#0 -> %arg1 = 0 to #map(%dim_0), %0#1 -> %arg2 = 0 to 1, %0#2 -> %arg3 = 0 to 1, %0#3 -> %arg4 = 0 to 384){
      %1:4 = krnl.get_induction_var_value(%0#0, %0#1, %0#2, %0#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
      %2 = krnl.load %arg0[%1#0, %1#1, %1#2, %1#3] : memref<?x1x1x384xf32>
      krnl.store %2, %alloc[%1#3, %1#2, %1#1, %1#0] : memref<384x1x1x?xf32>
    }
    return %alloc : memref<384x1x1x?xf32>
  }
}


// -----
#map = affine_map<(d0) -> (d0)>
module {
  func.func @test_transpose_lowered_to_a_view_op_inv(%arg0: memref<?x1x1x384xf32>) -> memref<384x1x1x?xf32> {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x1x1x384xf32>
    %alloc = memref.alloc(%dim) {alignment = 16 : i64} : memref<384x1x1x?xf32>
    %0:4 = krnl.define_loops 4
    %dim_0 = memref.dim %arg0, %c0 : memref<?x1x1x384xf32>
    krnl.iterate(%0#0, %0#1, %0#2, %0#3) with (%0#0 -> %arg1 = 0 to #map(%dim_0), %0#1 -> %arg2 = 0 to 1, %0#2 -> %arg3 = 0 to 1, %0#3 -> %arg4 = 0 to 384){
      %1:4 = krnl.get_induction_var_value(%0#0, %0#1, %0#2, %0#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
      %2 = krnl.load %arg0[%1#0, %1#1, %1#2, %1#3] : memref<?x1x1x384xf32>
      krnl.store %2, %alloc[%1#3, %1#2, %1#1, %1#0] : memref<384x1x1x?xf32>
    }
    return %alloc : memref<384x1x1x?xf32>
  }
}


// -----
#map = affine_map<(d0) -> (d0)>
module {
  func.func @test_transpose_block_1_last_dim(%arg0: memref<?x256x12x64xf32>) -> memref<64x12x256x64xf32> {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<64x12x256x64xf32>
    %0:4 = krnl.define_loops 4
    %dim = memref.dim %arg0, %c0 : memref<?x256x12x64xf32>
    krnl.iterate(%0#0, %0#1, %0#2, %0#3) with (%0#0 -> %arg1 = 0 to #map(%dim), %0#1 -> %arg2 = 0 to 256, %0#2 -> %arg3 = 0 to 12, %0#3 -> %arg4 = 0 to 64){
      %1:4 = krnl.get_induction_var_value(%0#0, %0#1, %0#2, %0#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
      %2 = krnl.load %arg0[%1#0, %1#1, %1#2, %1#3] : memref<?x256x12x64xf32>
      krnl.store %2, %alloc[%1#3, %1#2, %1#1, %1#0] : memref<64x12x256x64xf32>
    }
    return %alloc : memref<64x12x256x64xf32>
  }
}


// -----
Warning: [Shape inference, dim 0] the inferred dim (64) is different from the existing dim (2). Use the existing dim instead.
Warning: [Shape inference, dim 1] the inferred dim (32) is different from the existing dim (12). Use the existing dim instead.
Warning: [Shape inference, dim 2] the inferred dim (12) is different from the existing dim (256). Use the existing dim instead.
Warning: [Shape inference, dim 3] the inferred dim (256) is different from the existing dim (32). Use the existing dim instead.
Warning: [Shape inference, dim 4] the inferred dim (2) is different from the existing dim (64). Use the existing dim instead.
Warning: [Shape inference, dim 0] the inferred dim (64) is different from the existing dim (2). Use the existing dim instead.
Warning: [Shape inference, dim 1] the inferred dim (32) is different from the existing dim (12). Use the existing dim instead.
Warning: [Shape inference, dim 2] the inferred dim (12) is different from the existing dim (256). Use the existing dim instead.
Warning: [Shape inference, dim 3] the inferred dim (256) is different from the existing dim (32). Use the existing dim instead.
Warning: [Shape inference, dim 4] the inferred dim (2) is different from the existing dim (64). Use the existing dim instead.
Warning: [Shape inference, dim 0] the inferred dim (64) is different from the existing dim (2). Use the existing dim instead.
Warning: [Shape inference, dim 1] the inferred dim (32) is different from the existing dim (12). Use the existing dim instead.
Warning: [Shape inference, dim 2] the inferred dim (12) is different from the existing dim (256). Use the existing dim instead.
Warning: [Shape inference, dim 3] the inferred dim (256) is different from the existing dim (32). Use the existing dim instead.
Warning: [Shape inference, dim 4] the inferred dim (2) is different from the existing dim (64). Use the existing dim instead.
Warning: [Shape inference, dim 0] the inferred dim (64) is different from the existing dim (2). Use the existing dim instead.
Warning: [Shape inference, dim 1] the inferred dim (32) is different from the existing dim (12). Use the existing dim instead.
Warning: [Shape inference, dim 2] the inferred dim (12) is different from the existing dim (256). Use the existing dim instead.
Warning: [Shape inference, dim 3] the inferred dim (256) is different from the existing dim (32). Use the existing dim instead.
Warning: [Shape inference, dim 4] the inferred dim (2) is different from the existing dim (64). Use the existing dim instead.
module {
  func.func @test_transpose_block_2_last_dims(%arg0: memref<2x256x12x32x64xf32>) -> memref<2x12x256x32x64xf32> {
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<2x12x256x32x64xf32>
    %0:5 = krnl.define_loops 5
    krnl.iterate(%0#0, %0#1, %0#2, %0#3, %0#4) with (%0#0 -> %arg1 = 0 to 2, %0#1 -> %arg2 = 0 to 256, %0#2 -> %arg3 = 0 to 12, %0#3 -> %arg4 = 0 to 32, %0#4 -> %arg5 = 0 to 64){
      %1:5 = krnl.get_induction_var_value(%0#0, %0#1, %0#2, %0#3, %0#4) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index)
      %2 = krnl.load %arg0[%1#0, %1#1, %1#2, %1#3, %1#4] : memref<2x256x12x32x64xf32>
      krnl.store %2, %alloc[%1#4, %1#3, %1#2, %1#1, %1#0] : memref<2x12x256x32x64xf32>
    }
    return %alloc : memref<2x12x256x32x64xf32>
  }
}

