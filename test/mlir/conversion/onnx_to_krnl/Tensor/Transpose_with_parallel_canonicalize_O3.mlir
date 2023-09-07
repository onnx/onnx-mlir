#map = affine_map<(d0) -> (d0)>
module {
  func.func @test_transpose_block_1_last_dim_parallel(%arg0: memref<?x256x12x64xf32>) -> memref<64x12x256x64xf32> {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<64x12x256x64xf32>
    %0:4 = krnl.define_loops 4
    %dim = memref.dim %arg0, %c0 : memref<?x256x12x64xf32>
    krnl.parallel %0#0 : !krnl.loop
    krnl.iterate(%0#0, %0#1, %0#2, %0#3) with (%0#0 -> %arg1 = 0 to #map(%dim), %0#1 -> %arg2 = 0 to 256, %0#2 -> %arg3 = 0 to 12, %0#3 -> %arg4 = 0 to 64){
      %1:4 = krnl.get_induction_var_value(%0#0, %0#1, %0#2, %0#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
      %2 = krnl.load %arg0[%1#0, %1#1, %1#2, %1#3] : memref<?x256x12x64xf32>
      krnl.store %2, %alloc[%1#3, %1#2, %1#1, %1#0] : memref<64x12x256x64xf32>
    }
    return %alloc : memref<64x12x256x64xf32>
  }
}

