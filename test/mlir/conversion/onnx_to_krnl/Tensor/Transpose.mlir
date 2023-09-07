module {
  func.func private @test_transpose(%arg0: memref<10x20x30x40xf32>) -> memref<10x20x30x40xf32> {
    %c40 = arith.constant 40 : index
    %c30 = arith.constant 30 : index
    %c20 = arith.constant 20 : index
    %c10 = arith.constant 10 : index
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<40x30x20x10xf32>
    %0:4 = krnl.define_loops 4
    %c0 = arith.constant 0 : index
    %c10_0 = arith.constant 10 : index
    %c20_1 = arith.constant 20 : index
    %c30_2 = arith.constant 30 : index
    %c40_3 = arith.constant 40 : index
    krnl.iterate(%0#0, %0#1, %0#2, %0#3) with (%0#0 -> %arg1 = 0 to 10, %0#1 -> %arg2 = 0 to 20, %0#2 -> %arg3 = 0 to 30, %0#3 -> %arg4 = 0 to 40){
      %2:4 = krnl.get_induction_var_value(%0#0, %0#1, %0#2, %0#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
      %3 = krnl.load %arg0[%2#0, %2#1, %2#2, %2#3] : memref<10x20x30x40xf32>
      krnl.store %3, %alloc[%2#3, %2#2, %2#1, %2#0] : memref<40x30x20x10xf32>
    }
    %c10_4 = arith.constant 10 : index
    %c20_5 = arith.constant 20 : index
    %c30_6 = arith.constant 30 : index
    %c40_7 = arith.constant 40 : index
    %alloc_8 = memref.alloc() {alignment = 16 : i64} : memref<10x20x30x40xf32>
    %1:4 = krnl.define_loops 4
    %c0_9 = arith.constant 0 : index
    %c40_10 = arith.constant 40 : index
    %c30_11 = arith.constant 30 : index
    %c20_12 = arith.constant 20 : index
    %c10_13 = arith.constant 10 : index
    krnl.iterate(%1#0, %1#1, %1#2, %1#3) with (%1#0 -> %arg1 = 0 to 40, %1#1 -> %arg2 = 0 to 30, %1#2 -> %arg3 = 0 to 20, %1#3 -> %arg4 = 0 to 10){
      %2:4 = krnl.get_induction_var_value(%1#0, %1#1, %1#2, %1#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
      %3 = krnl.load %alloc[%2#0, %2#1, %2#2, %2#3] : memref<40x30x20x10xf32>
      krnl.store %3, %alloc_8[%2#3, %2#2, %2#1, %2#0] : memref<10x20x30x40xf32>
    }
    return %alloc_8 : memref<10x20x30x40xf32>
  }
}


// -----
#map = affine_map<(d0) -> (d0)>
module {
  func.func private @test_transpose_dynamic_dims(%arg0: memref<10x?x30x40xf32>) -> memref<40x30x?x10xf32> {
    %c40 = arith.constant 40 : index
    %c30 = arith.constant 30 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c1 : memref<10x?x30x40xf32>
    %c10 = arith.constant 10 : index
    %alloc = memref.alloc(%dim) {alignment = 16 : i64} : memref<40x30x?x10xf32>
    %0:4 = krnl.define_loops 4
    %c0 = arith.constant 0 : index
    %c10_0 = arith.constant 10 : index
    %c1_1 = arith.constant 1 : index
    %dim_2 = memref.dim %arg0, %c1_1 : memref<10x?x30x40xf32>
    %c30_3 = arith.constant 30 : index
    %c40_4 = arith.constant 40 : index
    krnl.iterate(%0#0, %0#1, %0#2, %0#3) with (%0#0 -> %arg1 = 0 to 10, %0#1 -> %arg2 = 0 to #map(%dim_2), %0#2 -> %arg3 = 0 to 30, %0#3 -> %arg4 = 0 to 40){
      %1:4 = krnl.get_induction_var_value(%0#0, %0#1, %0#2, %0#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
      %2 = krnl.load %arg0[%1#0, %1#1, %1#2, %1#3] : memref<10x?x30x40xf32>
      krnl.store %2, %alloc[%1#3, %1#2, %1#1, %1#0] : memref<40x30x?x10xf32>
    }
    return %alloc : memref<40x30x?x10xf32>
  }
}

