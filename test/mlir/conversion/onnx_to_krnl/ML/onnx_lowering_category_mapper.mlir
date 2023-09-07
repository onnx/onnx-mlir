module {
}


// -----
module {
  func.func private @test_category_mapper_string_to_int64(%arg0: memref<2x2x!krnl.string>) -> memref<2x2xi64> {
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c3_i32 = arith.constant 3 : i32
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<2x2xi64>
    %0 = "krnl.global"() <{name = "G0", shape = [3], value = dense<[1, 0, -3]> : tensor<3xi32>}> : () -> memref<3xi32>
    %1 = "krnl.global"() <{name = "V1", shape = [3], value = dense<[1, 2, 0]> : tensor<3xi32>}> : () -> memref<3xi32>
    %2 = "krnl.global"() <{name = "cats_int64s2", shape = [3], value = dense<[1, 2, 3]> : tensor<3xi64>}> : () -> memref<3xi64>
    %3 = "krnl.global"() <{name = "cats_strings3", shape = [3], value = dense<["cat", "dog", "cow"]> : tensor<3x!krnl.string>}> : () -> memref<3x!krnl.string>
    %4:2 = krnl.define_loops 2
    krnl.iterate(%4#0, %4#1) with (%4#0 -> %arg1 = 0 to 2, %4#1 -> %arg2 = 0 to 2){
      %5:2 = krnl.get_induction_var_value(%4#0, %4#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %6 = "krnl.getref"(%arg0, %c0_i64) : (memref<2x2x!krnl.string>, i64) -> memref<2x2x!krnl.string>
      %7 = krnl.load %6[%5#0, %5#1] : memref<2x2x!krnl.string>
      %8 = "krnl.find_index"(%7, %0, %1, %c3_i32) : (!krnl.string, memref<3xi32>, memref<3xi32>, i32) -> index
      %9 = krnl.load %3[%8] : memref<3x!krnl.string>
      %10 = "krnl.strlen"(%9) : (!krnl.string) -> i64
      %11 = "krnl.strncmp"(%7, %9, %10) : (!krnl.string, !krnl.string, i64) -> i32
      %12 = arith.cmpi eq, %11, %c0_i32 : i32
      scf.if %12 {
        %13 = krnl.load %2[%8] : memref<3xi64>
        krnl.store %13, %alloc[%5#0, %5#1] : memref<2x2xi64>
      } else {
        krnl.store %c-1_i64, %alloc[%5#0, %5#1] : memref<2x2xi64>
      }
    }
    return %alloc : memref<2x2xi64>
  }
}


// -----
module {
  func.func private @test_category_mapper_int64_to_string(%arg0: memref<2x2xi64>) -> memref<2x2x!krnl.string> {
    %c3_i32 = arith.constant 3 : i32
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<2x2x!krnl.string>
    %0 = "krnl.global"() <{name = "G5", shape = [3], value = dense<[-1, 1, 0]> : tensor<3xi32>}> : () -> memref<3xi32>
    %1 = "krnl.global"() <{name = "V6", shape = [3], value = dense<[2, 1, 0]> : tensor<3xi32>}> : () -> memref<3xi32>
    %2 = "krnl.global"() <{name = "cats_int64s7", shape = [3], value = dense<[1, 2, 3]> : tensor<3xi64>}> : () -> memref<3xi64>
    %3 = "krnl.global"() <{name = "cats_strings8", shape = [3], value = dense<["cat", "dog", "cow"]> : tensor<3x!krnl.string>}> : () -> memref<3x!krnl.string>
    %4 = "krnl.global"() <{name = "default_string9", shape = [], value = dense<"none"> : tensor<!krnl.string>}> : () -> memref<!krnl.string>
    %5:2 = krnl.define_loops 2
    krnl.iterate(%5#0, %5#1) with (%5#0 -> %arg1 = 0 to 2, %5#1 -> %arg2 = 0 to 2){
      %6:2 = krnl.get_induction_var_value(%5#0, %5#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %7 = krnl.load %arg0[%6#0, %6#1] : memref<2x2xi64>
      %8 = "krnl.find_index"(%7, %0, %1, %c3_i32) : (i64, memref<3xi32>, memref<3xi32>, i32) -> index
      %9 = krnl.load %2[%8] : memref<3xi64>
      %10 = arith.cmpi eq, %7, %9 : i64
      scf.if %10 {
        %11 = krnl.load %3[%8] : memref<3x!krnl.string>
        krnl.store %11, %alloc[%6#0, %6#1] : memref<2x2x!krnl.string>
      } else {
        %11 = krnl.load %4[] : memref<!krnl.string>
        krnl.store %11, %alloc[%6#0, %6#1] : memref<2x2x!krnl.string>
      }
    }
    return %alloc : memref<2x2x!krnl.string>
  }
}


// -----
module {
  func.func private @test_rank3_category_mapper_string_to_int64(%arg0: memref<2x2x2x!krnl.string>) -> memref<2x2x2xi64> {
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c3_i32 = arith.constant 3 : i32
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<2x2x2xi64>
    %0 = "krnl.global"() <{name = "G10", shape = [3], value = dense<[1, 0, -3]> : tensor<3xi32>}> : () -> memref<3xi32>
    %1 = "krnl.global"() <{name = "V11", shape = [3], value = dense<[1, 2, 0]> : tensor<3xi32>}> : () -> memref<3xi32>
    %2 = "krnl.global"() <{name = "cats_int64s12", shape = [3], value = dense<[1, 2, 3]> : tensor<3xi64>}> : () -> memref<3xi64>
    %3 = "krnl.global"() <{name = "cats_strings13", shape = [3], value = dense<["cat", "dog", "cow"]> : tensor<3x!krnl.string>}> : () -> memref<3x!krnl.string>
    %4:3 = krnl.define_loops 3
    krnl.iterate(%4#0, %4#1, %4#2) with (%4#0 -> %arg1 = 0 to 2, %4#1 -> %arg2 = 0 to 2, %4#2 -> %arg3 = 0 to 2){
      %5:3 = krnl.get_induction_var_value(%4#0, %4#1, %4#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
      %6 = "krnl.getref"(%arg0, %c0_i64) : (memref<2x2x2x!krnl.string>, i64) -> memref<2x2x2x!krnl.string>
      %7 = krnl.load %6[%5#0, %5#1, %5#2] : memref<2x2x2x!krnl.string>
      %8 = "krnl.find_index"(%7, %0, %1, %c3_i32) : (!krnl.string, memref<3xi32>, memref<3xi32>, i32) -> index
      %9 = krnl.load %3[%8] : memref<3x!krnl.string>
      %10 = "krnl.strlen"(%9) : (!krnl.string) -> i64
      %11 = "krnl.strncmp"(%7, %9, %10) : (!krnl.string, !krnl.string, i64) -> i32
      %12 = arith.cmpi eq, %11, %c0_i32 : i32
      scf.if %12 {
        %13 = krnl.load %2[%8] : memref<3xi64>
        krnl.store %13, %alloc[%5#0, %5#1, %5#2] : memref<2x2x2xi64>
      } else {
        krnl.store %c-1_i64, %alloc[%5#0, %5#1, %5#2] : memref<2x2x2xi64>
      }
    }
    return %alloc : memref<2x2x2xi64>
  }
}


// -----
module {
  func.func private @test_rank3_category_mapper_int64_to_string(%arg0: memref<2x2x2xi64>) -> memref<2x2x2x!krnl.string> {
    %c3_i32 = arith.constant 3 : i32
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<2x2x2x!krnl.string>
    %0 = "krnl.global"() <{name = "G15", shape = [3], value = dense<[-1, 1, 0]> : tensor<3xi32>}> : () -> memref<3xi32>
    %1 = "krnl.global"() <{name = "V16", shape = [3], value = dense<[2, 1, 0]> : tensor<3xi32>}> : () -> memref<3xi32>
    %2 = "krnl.global"() <{name = "cats_int64s17", shape = [3], value = dense<[1, 2, 3]> : tensor<3xi64>}> : () -> memref<3xi64>
    %3 = "krnl.global"() <{name = "cats_strings18", shape = [3], value = dense<["cat", "dog", "cow"]> : tensor<3x!krnl.string>}> : () -> memref<3x!krnl.string>
    %4 = "krnl.global"() <{name = "default_string19", shape = [], value = dense<"none"> : tensor<!krnl.string>}> : () -> memref<!krnl.string>
    %5:3 = krnl.define_loops 3
    krnl.iterate(%5#0, %5#1, %5#2) with (%5#0 -> %arg1 = 0 to 2, %5#1 -> %arg2 = 0 to 2, %5#2 -> %arg3 = 0 to 2){
      %6:3 = krnl.get_induction_var_value(%5#0, %5#1, %5#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
      %7 = krnl.load %arg0[%6#0, %6#1, %6#2] : memref<2x2x2xi64>
      %8 = "krnl.find_index"(%7, %0, %1, %c3_i32) : (i64, memref<3xi32>, memref<3xi32>, i32) -> index
      %9 = krnl.load %2[%8] : memref<3xi64>
      %10 = arith.cmpi eq, %7, %9 : i64
      scf.if %10 {
        %11 = krnl.load %3[%8] : memref<3x!krnl.string>
        krnl.store %11, %alloc[%6#0, %6#1, %6#2] : memref<2x2x2x!krnl.string>
      } else {
        %11 = krnl.load %4[] : memref<!krnl.string>
        krnl.store %11, %alloc[%6#0, %6#1, %6#2] : memref<2x2x2x!krnl.string>
      }
    }
    return %alloc : memref<2x2x2x!krnl.string>
  }
}

