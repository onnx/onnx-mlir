// RUN: onnx-mlir-opt --bundle-memory-pools --canonicalize %s -split-input-file | FileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (1)>
#map3 = affine_map<()[s0] -> (s0)>
#map4 = affine_map<() -> (3)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0)[s0, s1] -> (d0 + s0 * s1)>
#map7 = affine_map<() -> (2)>
#map8 = affine_map<() -> ()>
module {
  %0 = "krnl.packed_const"() {file_name = "/tmp/packed_const-77d95f.tmp", is_le = true, size_in_bytes = 0 : i64} : () -> i64
  func @bundle_lstm_dyn_mem_pools(%arg0: memref<?x?x2xf32>, %arg1: memref<1x12x2xf32>, %arg2: memref<1x12x3xf32>) -> memref<1x?x3xf32> attributes {input_names = ["X", "W", "R"], output_names = ["Y"]} {
    %c0 = constant 0 : index
    %c2 = constant 2 : index
    %c1 = constant 1 : index
    %cst = constant 1.000000e+00 : f32
    %cst_0 = constant 2.000000e+00 : f32
    %cst_1 = constant 0.000000e+00 : f32
    %c4 = constant 4 : index
    %c3 = constant 3 : index
    %c0_i64 = constant 0 : i64
    %1 = dim %arg0, %c1 : memref<?x?x2xf32>
    %2 = alloc(%1) : memref<1x?x3xf32>
    %3 = dim %arg0, %c1 : memref<?x?x2xf32>
    %4 = muli %3, %c4 : index
    %5 = muli %4, %c3 : index
    %6 = alloc(%5) : memref<?xi8>
    %7 = "krnl.getref"(%6, %c0_i64, %3) : (memref<?xi8>, i64, index) -> memref<1x?x3xf32>
    %8:3 = krnl.define_loops 3
    krnl.iterate(%8#0, %8#1, %8#2) with (%8#0 -> %arg3 = 0 to 1, %8#1 -> %arg4 = 0 to %1, %8#2 -> %arg5 = 0 to 3) {
      affine.store %cst_1, %2[%arg3, %arg4, %arg5] : memref<1x?x3xf32>
      affine.store %cst_1, %7[%arg3, %arg4, %arg5] : memref<1x?x3xf32>
    }
    %9 = krnl.define_loops 1
    %10 = dim %arg0, %c0 : memref<?x?x2xf32>
    krnl.iterate(%9) with (%9 -> %arg3 = 0 to %10) {
      %11 = dim %arg0, %c1 : memref<?x?x2xf32>
      %12 = muli %11, %c4 : index
      %13 = muli %12, %c3 : index
      %14 = alloc(%13) : memref<?xi8>
      %15 = "krnl.getref"(%14, %c0_i64, %11) : (memref<?xi8>, i64, index) -> memref<?x3xf32>
      %16 = muli %11, %c4 : index
      %17 = muli %16, %c3 : index
      %18 = alloc(%17) : memref<?xi8>
      %19 = "krnl.getref"(%18, %c0_i64, %11) : (memref<?xi8>, i64, index) -> memref<?x3xf32>
      %20 = dim %arg0, %c1 : memref<?x?x2xf32>
      %21 = muli %20, %c4 : index
      %22 = muli %21, %c3 : index
      %23 = alloc(%22) : memref<?xi8>
      %24 = "krnl.getref"(%23, %c0_i64, %20) : (memref<?xi8>, i64, index) -> memref<?x3xf32>
      %25 = muli %20, %c4 : index
      %26 = muli %25, %c3 : index
      %27 = alloc(%26) : memref<?xi8>
      %28 = "krnl.getref"(%27, %c0_i64, %20) : (memref<?xi8>, i64, index) -> memref<?x3xf32>
      %29 = dim %arg0, %c1 : memref<?x?x2xf32>
      %30 = muli %29, %c4 : index
      %31 = muli %30, %c3 : index
      %32 = alloc(%31) : memref<?xi8>
      %33 = "krnl.getref"(%32, %c0_i64, %29) : (memref<?xi8>, i64, index) -> memref<?x3xf32>
      %34 = muli %29, %c4 : index
      %35 = muli %34, %c3 : index
      %36 = alloc(%35) : memref<?xi8>
      %37 = "krnl.getref"(%36, %c0_i64, %29) : (memref<?xi8>, i64, index) -> memref<?x3xf32>
      %38 = dim %arg0, %c1 : memref<?x?x2xf32>
      %39 = muli %38, %c4 : index
      %40 = muli %39, %c3 : index
      %41 = alloc(%40) : memref<?xi8>
      %42 = "krnl.getref"(%41, %c0_i64, %38) : (memref<?xi8>, i64, index) -> memref<?x3xf32>
      %43 = muli %38, %c4 : index
      %44 = muli %43, %c3 : index
      %45 = alloc(%44) : memref<?xi8>
      %46 = "krnl.getref"(%45, %c0_i64, %38) : (memref<?xi8>, i64, index) -> memref<?x3xf32>
      %47:2 = krnl.define_loops 2
      %48 = dim %arg0, %c1 : memref<?x?x2xf32>
      krnl.iterate(%47#0, %47#1) with (%47#0 -> %arg4 = 0 to %48, %47#1 -> %arg5 = 0 to 3) {
        affine.store %cst_1, %15[%arg4, %arg5] : memref<?x3xf32>
        affine.store %cst_1, %19[%arg4, %arg5] : memref<?x3xf32>
        affine.store %cst_1, %24[%arg4, %arg5] : memref<?x3xf32>
        affine.store %cst_1, %28[%arg4, %arg5] : memref<?x3xf32>
        affine.store %cst_1, %33[%arg4, %arg5] : memref<?x3xf32>
        affine.store %cst_1, %37[%arg4, %arg5] : memref<?x3xf32>
        affine.store %cst_1, %42[%arg4, %arg5] : memref<?x3xf32>
        affine.store %cst_1, %46[%arg4, %arg5] : memref<?x3xf32>
        %51 = krnl.define_loops 1
        krnl.iterate(%51) with (%51 -> %arg6 = 0 to 2) {
          %53 = affine.apply #map6(%arg5)[%c0, %c3]
          %54 = affine.apply #map6(%arg5)[%c1, %c3]
          %55 = affine.apply #map6(%arg5)[%c2, %c3]
          %56 = affine.apply #map6(%arg5)[%c3, %c3]
          %57 = affine.load %arg0[%arg3, %arg4, %arg6] : memref<?x?x2xf32>
          %58 = affine.load %arg1[%c0, %53, %arg6] : memref<1x12x2xf32>
          %59 = mulf %57, %58 : f32
          %60 = affine.load %15[%arg4, %arg5] : memref<?x3xf32>
          %61 = addf %60, %59 : f32
          affine.store %61, %15[%arg4, %arg5] : memref<?x3xf32>
          %62 = affine.load %arg1[%c0, %54, %arg6] : memref<1x12x2xf32>
          %63 = mulf %57, %62 : f32
          %64 = affine.load %24[%arg4, %arg5] : memref<?x3xf32>
          %65 = addf %64, %63 : f32
          affine.store %65, %24[%arg4, %arg5] : memref<?x3xf32>
          %66 = affine.load %arg1[%c0, %55, %arg6] : memref<1x12x2xf32>
          %67 = mulf %57, %66 : f32
          %68 = affine.load %33[%arg4, %arg5] : memref<?x3xf32>
          %69 = addf %68, %67 : f32
          affine.store %69, %33[%arg4, %arg5] : memref<?x3xf32>
          %70 = affine.load %arg1[%c0, %56, %arg6] : memref<1x12x2xf32>
          %71 = mulf %57, %70 : f32
          %72 = affine.load %42[%arg4, %arg5] : memref<?x3xf32>
          %73 = addf %72, %71 : f32
          affine.store %73, %42[%arg4, %arg5] : memref<?x3xf32>
        }
        %52 = krnl.define_loops 1
        krnl.iterate(%52) with (%52 -> %arg6 = 0 to 3) {
          %53 = affine.apply #map6(%arg5)[%c0, %c3]
          %54 = affine.apply #map6(%arg5)[%c1, %c3]
          %55 = affine.apply #map6(%arg5)[%c2, %c3]
          %56 = affine.apply #map6(%arg5)[%c3, %c3]
          %57 = affine.load %2[%c0, %arg4, %arg6] : memref<1x?x3xf32>
          %58 = affine.load %arg2[%c0, %53, %arg6] : memref<1x12x3xf32>
          %59 = mulf %57, %58 : f32
          %60 = affine.load %19[%arg4, %arg5] : memref<?x3xf32>
          %61 = addf %60, %59 : f32
          affine.store %61, %19[%arg4, %arg5] : memref<?x3xf32>
          %62 = affine.load %arg2[%c0, %54, %arg6] : memref<1x12x3xf32>
          %63 = mulf %57, %62 : f32
          %64 = affine.load %28[%arg4, %arg5] : memref<?x3xf32>
          %65 = addf %64, %63 : f32
          affine.store %65, %28[%arg4, %arg5] : memref<?x3xf32>
          %66 = affine.load %arg2[%c0, %55, %arg6] : memref<1x12x3xf32>
          %67 = mulf %57, %66 : f32
          %68 = affine.load %37[%arg4, %arg5] : memref<?x3xf32>
          %69 = addf %68, %67 : f32
          affine.store %69, %37[%arg4, %arg5] : memref<?x3xf32>
          %70 = affine.load %arg2[%c0, %56, %arg6] : memref<1x12x3xf32>
          %71 = mulf %57, %70 : f32
          %72 = affine.load %46[%arg4, %arg5] : memref<?x3xf32>
          %73 = addf %72, %71 : f32
          affine.store %73, %46[%arg4, %arg5] : memref<?x3xf32>
        }
      }
      %49:2 = krnl.define_loops 2
      %50 = dim %arg0, %c1 : memref<?x?x2xf32>
      krnl.iterate(%49#0, %49#1) with (%49#0 -> %arg4 = 0 to %50, %49#1 -> %arg5 = 0 to 3) {
        %51 = alloc() : memref<4xi8>
        %52 = "krnl.getref"(%51, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
        %53 = alloc() : memref<4xi8>
        %54 = "krnl.getref"(%53, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
        %55 = alloc() : memref<4xi8>
        %56 = "krnl.getref"(%55, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
        %57 = alloc() : memref<4xi8>
        %58 = "krnl.getref"(%57, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
        %59 = alloc() : memref<4xi8>
        %60 = "krnl.getref"(%59, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
        %61 = affine.load %7[%c0, %arg4, %arg5] : memref<1x?x3xf32>
        %62 = affine.load %15[%arg4, %arg5] : memref<?x3xf32>
        %63 = affine.load %19[%arg4, %arg5] : memref<?x3xf32>
        %64 = addf %62, %63 : f32
        %65 = alloc() : memref<4xi8>
        %66 = "krnl.getref"(%65, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
        affine.store %64, %66[] : memref<f32>
        %67 = affine.load %66[] : memref<f32>
        %68 = subf %cst_1, %67 : f32
        %69 = exp %68 : f32
        %70 = addf %cst, %69 : f32
        %71 = divf %cst, %70 : f32
        affine.store %71, %60[] : memref<f32>
        %72 = affine.load %60[] : memref<f32>
        %73 = affine.load %33[%arg4, %arg5] : memref<?x3xf32>
        %74 = affine.load %37[%arg4, %arg5] : memref<?x3xf32>
        %75 = addf %73, %74 : f32
        %76 = alloc() : memref<4xi8>
        %77 = "krnl.getref"(%76, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
        affine.store %75, %77[] : memref<f32>
        %78 = affine.load %77[] : memref<f32>
        %79 = subf %cst_1, %78 : f32
        %80 = exp %79 : f32
        %81 = addf %cst, %80 : f32
        %82 = divf %cst, %81 : f32
        affine.store %82, %58[] : memref<f32>
        %83 = affine.load %58[] : memref<f32>
        %84 = affine.load %42[%arg4, %arg5] : memref<?x3xf32>
        %85 = affine.load %46[%arg4, %arg5] : memref<?x3xf32>
        %86 = addf %84, %85 : f32
        %87 = alloc() : memref<4xi8>
        %88 = "krnl.getref"(%87, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
        affine.store %86, %88[] : memref<f32>
        %89 = affine.load %88[] : memref<f32>
        %90 = mulf %89, %cst_0 : f32
        %91 = negf %90 : f32
        %92 = exp %91 : f32
        %93 = subf %cst, %92 : f32
        %94 = addf %cst, %92 : f32
        %95 = divf %93, %94 : f32
        %96 = exp %90 : f32
        %97 = subf %96, %cst : f32
        %98 = addf %96, %cst : f32
        %99 = divf %97, %98 : f32
        %100 = cmpf "oge", %89, %cst_1 : f32
        %101 = select %100, %95, %99 : f32
        affine.store %101, %56[] : memref<f32>
        %102 = affine.load %56[] : memref<f32>
        %103 = mulf %83, %61 : f32
        %104 = mulf %72, %102 : f32
        %105 = addf %103, %104 : f32
        affine.store %105, %7[%c0, %arg4, %arg5] : memref<1x?x3xf32>
        %106 = affine.load %24[%arg4, %arg5] : memref<?x3xf32>
        %107 = affine.load %28[%arg4, %arg5] : memref<?x3xf32>
        %108 = addf %106, %107 : f32
        %109 = alloc() : memref<4xi8>
        %110 = "krnl.getref"(%109, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
        affine.store %108, %110[] : memref<f32>
        %111 = affine.load %110[] : memref<f32>
        %112 = subf %cst_1, %111 : f32
        %113 = exp %112 : f32
        %114 = addf %cst, %113 : f32
        %115 = divf %cst, %114 : f32
        affine.store %115, %54[] : memref<f32>
        %116 = affine.load %54[] : memref<f32>
        %117 = alloc() : memref<4xi8>
        %118 = "krnl.getref"(%117, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
        affine.store %105, %118[] : memref<f32>
        %119 = affine.load %118[] : memref<f32>
        %120 = mulf %119, %cst_0 : f32
        %121 = negf %120 : f32
        %122 = exp %121 : f32
        %123 = subf %cst, %122 : f32
        %124 = addf %cst, %122 : f32
        %125 = divf %123, %124 : f32
        %126 = exp %120 : f32
        %127 = subf %126, %cst : f32
        %128 = addf %126, %cst : f32
        %129 = divf %127, %128 : f32
        %130 = cmpf "oge", %119, %cst_1 : f32
        %131 = select %130, %125, %129 : f32
        affine.store %131, %52[] : memref<f32>
        %132 = affine.load %52[] : memref<f32>
        %133 = mulf %116, %132 : f32
        affine.store %133, %2[%c0, %arg4, %arg5] : memref<1x?x3xf32>
        dealloc %117 : memref<4xi8>
        dealloc %109 : memref<4xi8>
        dealloc %87 : memref<4xi8>
        dealloc %76 : memref<4xi8>
        dealloc %65 : memref<4xi8>
        dealloc %59 : memref<4xi8>
        dealloc %57 : memref<4xi8>
        dealloc %55 : memref<4xi8>
        dealloc %53 : memref<4xi8>
        dealloc %51 : memref<4xi8>
      }
      dealloc %45 : memref<?xi8>
      dealloc %41 : memref<?xi8>
      dealloc %36 : memref<?xi8>
      dealloc %32 : memref<?xi8>
      dealloc %27 : memref<?xi8>
      dealloc %23 : memref<?xi8>
      dealloc %18 : memref<?xi8>
      dealloc %14 : memref<?xi8>
    }
    dealloc %6 : memref<?xi8>
    return %2 : memref<1x?x3xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 1 : i32} : () -> ()
}

// CHECK-LABEL: bundle_lstm_dyn_mem_pools

// CHECK-DAG: [[C0:%.+]] = constant 0 : index
// CHECK-DAG: [[C1:%.+]] = constant 1 : index
// CHECK-DAG: [[C4:%.+]] = constant 4 : index
// CHECK-DAG: [[C3:%.+]] = constant 3 : index
// CHECK-DAG: [[C0_I64:%.+]] = constant 0 : i64
// CHECK-DAG: [[C36_I64:%.+]] = constant 36 : i64
// CHECK-DAG: [[C32_I64:%.+]] = constant 32 : i64
// CHECK-DAG: [[C28_I64:%.+]] = constant 28 : i64
// CHECK-DAG: [[C24_I64:%.+]] = constant 24 : i64
// CHECK-DAG: [[C20_I64:%.+]] = constant 20 : i64
// CHECK-DAG: [[C16_I64:%.+]] = constant 16 : i64
// CHECK-DAG: [[C12_I64:%.+]] = constant 12 : i64
// CHECK-DAG: [[C8_I64:%.+]] = constant 8 : i64
// CHECK-DAG: [[C4_I64:%.+]] = constant 4 : i64

// CHECK: [[DIM1:%.+]] = dim %arg0, %c1 : memref<?x?x2xf32>
// CHECK: [[MUL1:%.+]] = muli [[DIM1]], [[C4]] : index
// CHECK: [[MUL2:%.+]] = muli [[MUL1]], [[C3]] : index
// CHECK: [[DYN_POOL_1:%.+]] = alloc([[MUL2]]) : memref<?xi8>
// CHECK: dim
// CHECK: [[RES:%.+]] = alloc
// CHECK: "krnl.getref"([[DYN_POOL_1]], [[C0_I64]], [[DIM1]]) : (memref<?xi8>, i64, index) -> memref<1x?x3xf32>

// CHECK: krnl.define_loops 3
// CHECK: krnl.iterate
// CHECK: krnl.define_loops 1
// CHECK: krnl.iterate

/// Inside loop block memory bundling
// CHECK: [[SIZE1:%.+]] = dim %arg0, [[C1]] : memref<?x?x2xf32>
// CHECK: [[SIZE2:%.+]] = dim %arg0, [[C1]] : memref<?x?x2xf32>
// CHECK: [[SIZE3:%.+]] = dim %arg0, [[C1]] : memref<?x?x2xf32>
// CHECK: [[SIZE4:%.+]] = dim %arg0, [[C1]] : memref<?x?x2xf32>

// CHECK: [[OFF1:%.+]] = index_cast
// CHECK: [[OFF2:%.+]] = index_cast
// CHECK: [[OFF3:%.+]] = index_cast
// CHECK: [[OFF4:%.+]] = index_cast
// CHECK: [[OFF5:%.+]] = index_cast
// CHECK: [[OFF6:%.+]] = index_cast
// CHECK: [[OFF7:%.+]] = index_cast

// CHECK: [[DYN_POOL_2:%.+]] = alloc({{.*}}) : memref<?xi8>
// CHECK: "krnl.getref"([[DYN_POOL_2]], [[OFF7]], [[SIZE1]]) : (memref<?xi8>, i64, index) -> memref<?x3xf32>
// CHECK: "krnl.getref"([[DYN_POOL_2]], [[OFF6]], [[SIZE1]]) : (memref<?xi8>, i64, index) -> memref<?x3xf32>
// CHECK: "krnl.getref"([[DYN_POOL_2]], [[OFF5]], [[SIZE2]]) : (memref<?xi8>, i64, index) -> memref<?x3xf32>
// CHECK: "krnl.getref"([[DYN_POOL_2]], [[OFF4]], [[SIZE2]]) : (memref<?xi8>, i64, index) -> memref<?x3xf32>
// CHECK: "krnl.getref"([[DYN_POOL_2]], [[OFF3]], [[SIZE3]]) : (memref<?xi8>, i64, index) -> memref<?x3xf32>
// CHECK: "krnl.getref"([[DYN_POOL_2]], [[OFF2]], [[SIZE3]]) : (memref<?xi8>, i64, index) -> memref<?x3xf32>
// CHECK: "krnl.getref"([[DYN_POOL_2]], [[OFF1]], [[SIZE4]]) : (memref<?xi8>, i64, index) -> memref<?x3xf32>
// CHECK: "krnl.getref"([[DYN_POOL_2]], [[C0_I64]], [[SIZE4]]) : (memref<?xi8>, i64, index) -> memref<?x3xf32>

// CHECK: krnl.iterate
// CHECK: krnl.iterate
// CHECK: krnl.iterate
// CHECK: krnl.define_loops 2
// CHECK: krnl.iterate

/// Inside loop block memory bundling
// CHECK: [[STATIC_POOL_1:%.+]] = alloc() : memref<40xi8>
// CHECK: "krnl.getref"([[STATIC_POOL_1]], [[C36_I64]]) : (memref<40xi8>, i64) -> memref<f32>
// CHECK: "krnl.getref"([[STATIC_POOL_1]], [[C32_I64]]) : (memref<40xi8>, i64) -> memref<f32>
// CHECK: "krnl.getref"([[STATIC_POOL_1]], [[C28_I64]]) : (memref<40xi8>, i64) -> memref<f32>
// CHECK: "krnl.getref"([[STATIC_POOL_1]], [[C24_I64]]) : (memref<40xi8>, i64) -> memref<f32>
// CHECK: "krnl.getref"([[STATIC_POOL_1]], [[C20_I64]]) : (memref<40xi8>, i64) -> memref<f32>

// CHECK: "krnl.getref"([[STATIC_POOL_1]], [[C16_I64]]) : (memref<40xi8>, i64) -> memref<f32>
// CHECK: "krnl.getref"([[STATIC_POOL_1]], [[C12_I64]]) : (memref<40xi8>, i64) -> memref<f32>
// CHECK: "krnl.getref"([[STATIC_POOL_1]], [[C8_I64]]) : (memref<40xi8>, i64) -> memref<f32>
// CHECK: "krnl.getref"([[STATIC_POOL_1]], [[C4_I64]]) : (memref<40xi8>, i64) -> memref<f32>
// CHECK: "krnl.getref"([[STATIC_POOL_1]], [[C0_I64]]) : (memref<40xi8>, i64) -> memref<f32>

// CHECK: dealloc [[STATIC_POOL_1]] : memref<40xi8>

// CHECK: dealloc [[DYN_POOL_2]] : memref<?xi8>
// CHECK: dealloc [[DYN_POOL_1]] : memref<?xi8>
// CHECK: return [[RES]] : memref<1x?x3xf32>

