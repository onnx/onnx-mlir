module {
  func.func @test_elide_krnl_global_constant(%arg0: memref<1xf32>) -> memref<1x70xf32> {
    %0 = "krnl.global"() <{name = "constant_00", shape = [1, 70]}> : () -> memref<1x70xf32>
    return %0 : memref<1x70xf32>
  }
}


// -----
module {
  func.func @test_elide_krnl_global_constant() -> memref<1x80xf32> {
    %0 = "krnl.global"() <{name = "constant_01", shape = [1, 80]}> : () -> memref<1x80xf32>
    return %0 : memref<1x80xf32>
  }
}

