module {
  func.func private @test_loop_simple_main_graph(%arg0: memref<i64>, %arg1: memref<i1>, %arg2: memref<1xi64>) -> memref<1xi64> {
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1xi64>
    %0 = krnl.define_loops 1
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    krnl.iterate(%0) with (%0 -> %arg3 = 0 to 1){
      %5 = krnl.get_induction_var_value(%0) : (!krnl.loop) -> index
      %6 = krnl.load %arg2[%5] : memref<1xi64>
      krnl.store %6, %alloc[%5] : memref<1xi64>
    }
    %alloc_0 = memref.alloc() : memref<i1>
    %1 = krnl.load %arg1[] : memref<i1>
    krnl.store %1, %alloc_0[] : memref<i1>
    %2 = krnl.load %arg0[] : memref<i64>
    %3 = arith.index_cast %2 : i64 to index
    %4 = krnl.define_loops 1
    %c0_1 = arith.constant 0 : index
    krnl.iterate(%4) with (%4 -> %arg3 = %c0_1 to %3){
      %5 = krnl.get_induction_var_value(%4) : (!krnl.loop) -> index
      %6 = krnl.load %alloc_0[] : memref<i1>
      scf.if %6 {
        "krnl.region"() ({
          %7 = arith.index_cast %5 : index to i64
          %alloc_2 = memref.alloc() : memref<i64>
          krnl.store %7, %alloc_2[] : memref<i64>
          %c1_3 = arith.constant 1 : index
          %c1_4 = arith.constant 1 : index
          %c1_5 = arith.constant 1 : index
          %alloc_6 = memref.alloc() {alignment = 16 : i64} : memref<1xi64>
          %8 = krnl.define_loops 1
          %c0_7 = arith.constant 0 : index
          %c1_8 = arith.constant 1 : index
          krnl.iterate(%8) with (%8 -> %arg4 = 0 to 1){
            %14 = krnl.get_induction_var_value(%8) : (!krnl.loop) -> index
            %c1_11 = arith.constant 1 : index
            %c0_12 = arith.constant 0 : index
            %15 = krnl.load %alloc[%c0_12] : memref<1xi64>
            %16 = krnl.load %alloc_2[] : memref<i64>
            %17 = arith.addi %15, %16 : i64
            krnl.store %17, %alloc_6[%14] : memref<1xi64>
          }
          %9 = builtin.unrealized_conversion_cast %alloc_6 : memref<1xi64> to tensor<1xi64>
          %10 = builtin.unrealized_conversion_cast %arg1 : memref<i1> to memref<i1>
          %11 = builtin.unrealized_conversion_cast %9 : tensor<1xi64> to memref<1xi64>
          %12 = krnl.load %10[] : memref<i1>
          krnl.store %12, %alloc_0[] : memref<i1>
          %13 = krnl.define_loops 1
          %c0_9 = arith.constant 0 : index
          %c1_10 = arith.constant 1 : index
          krnl.iterate(%13) with (%13 -> %arg4 = 0 to 1){
            %14 = krnl.get_induction_var_value(%13) : (!krnl.loop) -> index
            %15 = krnl.load %11[%14] : memref<1xi64>
            krnl.store %15, %alloc[%14] : memref<1xi64>
          }
        }) : () -> ()
      }
    }
    return %alloc : memref<1xi64>
  }
}


// -----
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d2)>
#map2 = affine_map<(d0) -> (d0)>
module {
  func.func @test_loop(%arg0: memref<i64>, %arg1: memref<i1>, %arg2: memref<?xf32>) -> memref<?x?xf32> {
    %0 = krnl.load %arg0[] : memref<i64>
    %1 = arith.index_cast %0 : i64 to index
    %alloc = memref.alloc(%1) {alignment = 16 : i64} : memref<?xmemref<?xf32>>
    %alloc_0 = memref.alloc() : memref<i1>
    %2 = krnl.load %arg1[] : memref<i1>
    krnl.store %2, %alloc_0[] : memref<i1>
    %3 = krnl.load %arg0[] : memref<i64>
    %4 = arith.index_cast %3 : i64 to index
    %5 = krnl.define_loops 1
    %c0 = arith.constant 0 : index
    krnl.iterate(%5) with (%5 -> %arg3 = %c0 to %4){
      %8 = krnl.get_induction_var_value(%5) : (!krnl.loop) -> index
      %9 = krnl.load %alloc_0[] : memref<i1>
      scf.if %9 {
        "krnl.region"() ({
          %10 = arith.index_cast %8 : index to i64
          %alloc_5 = memref.alloc() : memref<i64>
          krnl.store %10, %alloc_5[] : memref<i64>
          %c1 = arith.constant 1 : index
          %c0_6 = arith.constant 0 : index
          %dim_7 = memref.dim %arg2, %c0_6 : memref<?xf32>
          %c0_8 = arith.constant 0 : index
          %dim_9 = memref.dim %arg2, %c0_8 : memref<?xf32>
          %11 = affine.max #map(%dim_7, %dim_9)
          %c1_10 = arith.constant 1 : index
          %alloc_11 = memref.alloc(%11) {alignment = 16 : i64} : memref<?xf32>
          %12 = krnl.define_loops 1
          %c0_12 = arith.constant 0 : index
          %c0_13 = arith.constant 0 : index
          krnl.iterate(%12) with (%12 -> %arg4 = 0 to #map1(%dim_7, %dim_9, %11)){
            %17 = krnl.get_induction_var_value(%12) : (!krnl.loop) -> index
            %18 = krnl.load %arg2[%17] : memref<?xf32>
            %19 = krnl.load %arg2[%17] : memref<?xf32>
            %20 = arith.addf %18, %19 : f32
            krnl.store %20, %alloc_11[%17] : memref<?xf32>
          }
          %13 = builtin.unrealized_conversion_cast %alloc_11 : memref<?xf32> to tensor<?xf32>
          %14 = builtin.unrealized_conversion_cast %arg1 : memref<i1> to memref<i1>
          %15 = builtin.unrealized_conversion_cast %13 : tensor<?xf32> to memref<?xf32>
          %16 = krnl.load %14[] : memref<i1>
          krnl.store %16, %alloc_0[] : memref<i1>
          "krnl.seqstore"(%15, %alloc, %8) : (memref<?xf32>, memref<?xmemref<?xf32>>, index) -> ()
        }) : () -> ()
      }
    }
    %c0_1 = arith.constant 0 : index
    %6 = krnl.load %alloc[%c0_1] : memref<?xmemref<?xf32>>
    %c0_2 = arith.constant 0 : index
    %c0_3 = arith.constant 0 : index
    %dim = memref.dim %6, %c0_3 : memref<?xf32>
    %alloc_4 = memref.alloc(%1, %dim) {alignment = 16 : i64} : memref<?x?xf32>
    %7 = krnl.define_loops 1
    krnl.iterate(%7) with (%7 -> %arg3 = %c0 to %4){
      %8 = krnl.get_induction_var_value(%7) : (!krnl.loop) -> index
      "krnl.region"() ({
        %9 = "krnl.seqextract"(%alloc, %8) <{copy = 0 : ui1}> : (memref<?xmemref<?xf32>>, index) -> memref<?xf32>
        %10 = krnl.define_loops 1
        %c0_5 = arith.constant 0 : index
        %c0_6 = arith.constant 0 : index
        %dim_7 = memref.dim %9, %c0_6 : memref<?xf32>
        krnl.iterate(%10) with (%10 -> %arg4 = 0 to #map2(%dim_7)){
          %11 = krnl.get_induction_var_value(%10) : (!krnl.loop) -> index
          %12 = krnl.load %9[%11] : memref<?xf32>
          krnl.store %12, %alloc_4[%8, %11] : memref<?x?xf32>
        }
      }) : () -> ()
    }
    return %alloc_4 : memref<?x?xf32>
  }
}

