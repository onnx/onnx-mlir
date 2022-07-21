// RUN: onnx-mlir-opt -O3 -allow-unregistered-dialect --optimize-memory-pools --canonicalize %s -split-input-file | FileCheck %s

#map0 = affine_map<(d0) -> (d0 + 4)>
#map1 = affine_map<(d0) -> (d0 + 8)>
#map2 = affine_map<(d0) -> (d0 + 12)>
#map3 = affine_map<(d0) -> (d0 + 16)>
#map4 = affine_map<(d0) -> (d0 + 20)>
#map5 = affine_map<(d0) -> (d0 + 24)>
#map6 = affine_map<(d0) -> (d0 + 28)>
func.func @main_graph(%arg0: memref<1x3x3xf32>, %arg1: memref<1x16x3xf32>, %arg2: memref<1x16x4xf32>, %arg3: memref<1x32xf32>) -> memref<1x3x4xf32> {
    %c0_i64 = arith.constant 0 : i64
    %c163840_i64 = arith.constant 163840 : i64
    %c155648_i64 = arith.constant 155648 : i64
    %c122880_i64 = arith.constant 122880 : i64
    %c90112_i64 = arith.constant 90112 : i64
    %c57344_i64 = arith.constant 57344 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c960_i64 = arith.constant 960 : i64
    %c896_i64 = arith.constant 896 : i64
    %c832_i64 = arith.constant 832 : i64
    %c768_i64 = arith.constant 768 : i64
    %c512_i64 = arith.constant 512 : i64
    %c496_i64 = arith.constant 496 : i64
    %c480_i64 = arith.constant 480 : i64
    %c464_i64 = arith.constant 464 : i64
    %c448_i64 = arith.constant 448 : i64
    %c432_i64 = arith.constant 432 : i64
    %c416_i64 = arith.constant 416 : i64
    %c400_i64 = arith.constant 400 : i64
    %c384_i64 = arith.constant 384 : i64
    %c336_i64 = arith.constant 336 : i64
    %c288_i64 = arith.constant 288 : i64
    %c240_i64 = arith.constant 240 : i64
    %c192_i64 = arith.constant 192 : i64
    %1 = memref.alloc() : memref<1x3x4xf32>
    %2 = memref.alloc() {alignment = 4096 : i64} : memref<172032xi8>
    %6 = "krnl.getref"(%2, %c90112_i64) : (memref<172032xi8>, i64) -> memref<1x4x1x1x32x64xf32>
    %7 = "krnl.getref"(%2, %c57344_i64) : (memref<172032xi8>, i64) -> memref<1x4x1x1x32x64xf32>
    %8 = "krnl.getref"(%2, %c24576_i64) : (memref<172032xi8>, i64) -> memref<1x4x1x1x32x64xf32>
    %9 = "krnl.getref"(%2, %c16384_i64) : (memref<172032xi8>, i64) -> memref<1x1x1x1x32x64xf32>
    %10 = "krnl.getref"(%2, %c8192_i64) : (memref<172032xi8>, i64) -> memref<1x1x1x1x32x64xf32>
    %11 = "krnl.getref"(%2, %c0_i64) : (memref<172032xi8>, i64) -> memref<1x1x1x1x32x64xf32>
    %12 = memref.alloc() : memref<1024xi8>
    %18 = "krnl.getref"(%12, %c496_i64) : (memref<1024xi8>, i64) -> memref<1x4xf32>
    %19 = "krnl.getref"(%12, %c480_i64) : (memref<1024xi8>, i64) -> memref<1x4xf32>
    %20 = "krnl.getref"(%12, %c464_i64) : (memref<1024xi8>, i64) -> memref<1x4xf32>
    %21 = "krnl.getref"(%12, %c448_i64) : (memref<1024xi8>, i64) -> memref<1x4xf32>
    %22 = "krnl.getref"(%12, %c432_i64) : (memref<1024xi8>, i64) -> memref<1x4xf32>
    %23 = "krnl.getref"(%12, %c416_i64) : (memref<1024xi8>, i64) -> memref<1x4xf32>
    %24 = "krnl.getref"(%12, %c400_i64) : (memref<1024xi8>, i64) -> memref<1x4xf32>
    %25 = "krnl.getref"(%12, %c384_i64) : (memref<1024xi8>, i64) -> memref<1x4xf32>
    %26 = "krnl.getref"(%12, %c336_i64) : (memref<1024xi8>, i64) -> memref<1x3x4xf32>
    %27 = "krnl.getref"(%12, %c288_i64) : (memref<1024xi8>, i64) -> memref<1x3x4xf32>
    %28 = "krnl.getref"(%12, %c240_i64) : (memref<1024xi8>, i64) -> memref<1x3x4xf32>
    %29 = "krnl.getref"(%12, %c192_i64) : (memref<1024xi8>, i64) -> memref<1x3x4xf32>
    %30 = "krnl.getref"(%12, %c0_i64) : (memref<1024xi8>, i64) -> memref<1x3x16xf32>
    "unknown.foo"(%arg0, %11) : (memref<1x3x3xf32>, memref<1x1x1x1x32x64xf32>) -> ()
    %31 = "krnl.global"() {name = "constant_0", offset = 0 : i64, shape = [1, 3, 4], value = dense<0.000000e+00> : tensor<1x3x4xf32>} : () -> memref<1x3x4xf32>
    "unknown.foo"(%31, %10) : (memref<1x3x4xf32>, memref<1x1x1x1x32x64xf32>) -> ()
    "unknown.foo"(%31, %9) : (memref<1x3x4xf32>, memref<1x1x1x1x32x64xf32>) -> ()
    %32:3 = krnl.define_loops 3
    krnl.iterate(%32#0, %32#1, %32#2) with (%32#0 -> %arg4 = 0 to 1, %32#1 -> %arg5 = 0 to 16, %32#2 -> %arg6 = 0 to 3) {
      %45 = krnl.load %arg1[%arg4, %arg5, %arg6] : memref<1x16x3xf32>
      krnl.store %45, %30[%arg4, %arg6, %arg5] : memref<1x3x16xf32>
    }
    %33:3 = krnl.define_loops 3
    krnl.iterate(%33#0, %33#1, %33#2) with (%33#0 -> %arg4 = 0 to 1, %33#1 -> %arg5 = 0 to 3, %33#2 -> %arg6 = 0 to 4) {
      %45 = krnl.load %30[%arg4, %arg5, %arg6] : memref<1x3x16xf32>
      krnl.store %45, %29[%arg4, %arg5, %arg6] : memref<1x3x4xf32>
    }
    %34:3 = krnl.define_loops 3
    krnl.iterate(%34#0, %34#1, %34#2) with (%34#0 -> %arg4 = 0 to 1, %34#1 -> %arg5 = 0 to 3, %34#2 -> %arg6 = 0 to 4) {
      %45 = affine.apply #map0(%arg6)
      %46 = krnl.load %30[%arg4, %arg5, %45] : memref<1x3x16xf32>
      krnl.store %46, %28[%arg4, %arg5, %arg6] : memref<1x3x4xf32>
    }
    %35:3 = krnl.define_loops 3
    krnl.iterate(%35#0, %35#1, %35#2) with (%35#0 -> %arg4 = 0 to 1, %35#1 -> %arg5 = 0 to 3, %35#2 -> %arg6 = 0 to 4) {
      %45 = affine.apply #map1(%arg6)
      %46 = krnl.load %30[%arg4, %arg5, %45] : memref<1x3x16xf32>
      krnl.store %46, %27[%arg4, %arg5, %arg6] : memref<1x3x4xf32>
    }
    %36:3 = krnl.define_loops 3
    krnl.iterate(%36#0, %36#1, %36#2) with (%36#0 -> %arg4 = 0 to 1, %36#1 -> %arg5 = 0 to 3, %36#2 -> %arg6 = 0 to 4) {
      %45 = affine.apply #map2(%arg6)
      %46 = krnl.load %30[%arg4, %arg5, %45] : memref<1x3x16xf32>
      krnl.store %46, %26[%arg4, %arg5, %arg6] : memref<1x3x4xf32>
    }
    "unknown.bar"(%27, %29, %26, %28, %8) : (memref<1x3x4xf32>, memref<1x3x4xf32>, memref<1x3x4xf32>, memref<1x3x4xf32>, memref<1x4x1x1x32x64xf32>) -> ()
    %37:2 = krnl.define_loops 2
    krnl.iterate(%37#0, %37#1) with (%37#0 -> %arg4 = 0 to 1, %37#1 -> %arg5 = 0 to 4) {
      %45 = krnl.load %arg3[%arg4, %arg5] : memref<1x32xf32>
      krnl.store %45, %25[%arg4, %arg5] : memref<1x4xf32>
    }
    %38:2 = krnl.define_loops 2
    krnl.iterate(%38#0, %38#1) with (%38#0 -> %arg4 = 0 to 1, %38#1 -> %arg5 = 0 to 4) {
      %45 = affine.apply #map0(%arg5)
      %46 = krnl.load %arg3[%arg4, %45] : memref<1x32xf32>
      krnl.store %46, %24[%arg4, %arg5] : memref<1x4xf32>
    }
    %39:2 = krnl.define_loops 2
    krnl.iterate(%39#0, %39#1) with (%39#0 -> %arg4 = 0 to 1, %39#1 -> %arg5 = 0 to 4) {
      %45 = affine.apply #map1(%arg5)
      %46 = krnl.load %arg3[%arg4, %45] : memref<1x32xf32>
      krnl.store %46, %23[%arg4, %arg5] : memref<1x4xf32>
    }
    %40:2 = krnl.define_loops 2
    krnl.iterate(%40#0, %40#1) with (%40#0 -> %arg4 = 0 to 1, %40#1 -> %arg5 = 0 to 4) {
      %45 = affine.apply #map2(%arg5)
      %46 = krnl.load %arg3[%arg4, %45] : memref<1x32xf32>
      krnl.store %46, %22[%arg4, %arg5] : memref<1x4xf32>
    }
    %41:2 = krnl.define_loops 2
    krnl.iterate(%41#0, %41#1) with (%41#0 -> %arg4 = 0 to 1, %41#1 -> %arg5 = 0 to 4) {
      %45 = affine.apply #map3(%arg5)
      %46 = krnl.load %arg3[%arg4, %45] : memref<1x32xf32>
      krnl.store %46, %21[%arg4, %arg5] : memref<1x4xf32>
    }
    %42:2 = krnl.define_loops 2
    krnl.iterate(%42#0, %42#1) with (%42#0 -> %arg4 = 0 to 1, %42#1 -> %arg5 = 0 to 4) {
      %45 = affine.apply #map4(%arg5)
      %46 = krnl.load %arg3[%arg4, %45] : memref<1x32xf32>
      krnl.store %46, %20[%arg4, %arg5] : memref<1x4xf32>
    }
    %43:2 = krnl.define_loops 2
    krnl.iterate(%43#0, %43#1) with (%43#0 -> %arg4 = 0 to 1, %43#1 -> %arg5 = 0 to 4) {
      %45 = affine.apply #map5(%arg5)
      %46 = krnl.load %arg3[%arg4, %45] : memref<1x32xf32>
      krnl.store %46, %19[%arg4, %arg5] : memref<1x4xf32>
    }
    %44:2 = krnl.define_loops 2
    krnl.iterate(%44#0, %44#1) with (%44#0 -> %arg4 = 0 to 1, %44#1 -> %arg5 = 0 to 4) {
      %45 = affine.apply #map6(%arg5)
      %46 = krnl.load %arg3[%arg4, %45] : memref<1x32xf32>
      krnl.store %46, %18[%arg4, %arg5] : memref<1x4xf32>
    }
    "unknown.bar"(%23, %25, %22, %24, %7) : (memref<1x4xf32>, memref<1x4xf32>, memref<1x4xf32>, memref<1x4xf32>, memref<1x4x1x1x32x64xf32>) -> ()
    "unknown.bar"(%19, %21, %18, %20, %6) : (memref<1x4xf32>, memref<1x4xf32>, memref<1x4xf32>, memref<1x4xf32>, memref<1x4x1x1x32x64xf32>) -> ()
    memref.dealloc %12 : memref<1024xi8>
    memref.dealloc %2 : memref<172032xi8>
    return %1 : memref<1x3x4xf32>
}

// CHECK: [[RES:%.+]] = memref.alloc() : memref<1x3x4xf32>
// CHECK: [[ALIGNED_MEM_POOL:%.+]] = memref.alloc() {alignment = 4096 : i64} : memref<40960xi8>
// CHECK: "krnl.getref"([[ALIGNED_MEM_POOL]], %c0_i64) : (memref<40960xi8>, i64) -> memref<1x4x1x1x32x64xf32>
// CHECK: "krnl.getref"([[ALIGNED_MEM_POOL]], %c0_i64) : (memref<40960xi8>, i64) -> memref<1x4x1x1x32x64xf32>
// CHECK: "krnl.getref"([[ALIGNED_MEM_POOL]], %c0_i64) : (memref<40960xi8>, i64) -> memref<1x4x1x1x32x64xf32>
// CHECK: "krnl.getref"([[ALIGNED_MEM_POOL]], %c32768_i64) : (memref<40960xi8>, i64) -> memref<1x1x1x1x32x64xf32>
// CHECK: "krnl.getref"([[ALIGNED_MEM_POOL]], %c32768_i64) : (memref<40960xi8>, i64) -> memref<1x1x1x1x32x64xf32>
// CHECK: "krnl.getref"([[ALIGNED_MEM_POOL]], %c32768_i64) : (memref<40960xi8>, i64) -> memref<1x1x1x1x32x64xf32>
// CHECK: [[MEM_POOL:%.+]] = memref.alloc() : memref<512xi8>
// CHECK: "krnl.getref"([[MEM_POOL]], %c0_i64) : (memref<512xi8>, i64) -> memref<1x4xf32>
// CHECK: "krnl.getref"([[MEM_POOL]], %c16_i64) : (memref<512xi8>, i64) -> memref<1x4xf32>
// CHECK: "krnl.getref"([[MEM_POOL]], %c32_i64) : (memref<512xi8>, i64) -> memref<1x4xf32>
// CHECK: "krnl.getref"([[MEM_POOL]], %c48_i64) : (memref<512xi8>, i64) -> memref<1x4xf32>
// CHECK: "krnl.getref"([[MEM_POOL]], %c64_i64) : (memref<512xi8>, i64) -> memref<1x4xf32>
// CHECK: "krnl.getref"([[MEM_POOL]], %c80_i64) : (memref<512xi8>, i64) -> memref<1x4xf32>
// CHECK: "krnl.getref"([[MEM_POOL]], %c96_i64) : (memref<512xi8>, i64) -> memref<1x4xf32>
// CHECK: "krnl.getref"([[MEM_POOL]], %c112_i64) : (memref<512xi8>, i64) -> memref<1x4xf32>
// CHECK: "krnl.getref"([[MEM_POOL]], %c128_i64) : (memref<512xi8>, i64) -> memref<1x3x4xf32>
// CHECK: "krnl.getref"([[MEM_POOL]], %c176_i64) : (memref<512xi8>, i64) -> memref<1x3x4xf32>
// CHECK: "krnl.getref"([[MEM_POOL]], %c224_i64) : (memref<512xi8>, i64) -> memref<1x3x4xf32>
// CHECK: "krnl.getref"([[MEM_POOL]], %c272_i64) : (memref<512xi8>, i64) -> memref<1x3x4xf32>
// CHECK: "krnl.getref"([[MEM_POOL]], %c320_i64) : (memref<512xi8>, i64) -> memref<1x3x16xf32>



