#map0 = affine_map<(d0) -> (0, d0 * 2)>
#map1 = affine_map<(d0) -> (4, d0 * 2 + 3)>
#map2 = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0 - ((s2 ceildiv s4) * s4 - s2), -(d0 * s3 - s2) + s0, d0 * s3 + (s1 - 1) * s4 - s2 - ((s2 ceildiv s4) * s4 - s2) + 1, d0 * s3 + (s1 - 1) * s4 - s2 - (d0 * s3 - s2) + 1)>
module  {
  func @main_graph(%arg0: memref<1x1x4x4xf32>) -> memref<1x1x2x2xf32> attributes {input_names = ["dequantize_output"], output_names = ["average_pool_output"]} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.alloc() {alignment = 16 : i64} : memref<1x1x2x2xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 2 {
          affine.for %arg4 = 0 to 2 {
            %1 = memref.alloca() : memref<f32>
            affine.store %cst, %1[] : memref<f32>
            %2 = affine.max #map0(%arg3)
            %3 = affine.min #map1(%arg3)
            %4 = affine.max #map0(%arg4)
            %5 = affine.min #map1(%arg4)
            %6 = arith.subi %3, %2 : index
            %7 = arith.subi %5, %4 : index
            affine.for %arg5 = 0 to min #map2(%arg3)[%c4, %c3, %c0, %c2, %c1] {
              affine.for %arg6 = 0 to min #map2(%arg4)[%c4, %c3, %c0, %c2, %c1] {
                %14 = arith.addi %arg5, %2 : index
                %15 = arith.addi %arg6, %4 : index
                %16 = memref.load %arg0[%arg1, %arg2, %14, %15] : memref<1x1x4x4xf32>
                %17 = affine.load %1[] : memref<f32>
                %18 = arith.addf %17, %16 : f32
                affine.store %18, %1[] : memref<f32>
              }
            }
            %8 = affine.load %1[] : memref<f32>
            affine.store %8, %0[%arg1, %arg2, %arg3, %arg4] : memref<1x1x2x2xf32>
            %9 = affine.load %0[%arg1, %arg2, %arg3, %arg4] : memref<1x1x2x2xf32>
            %10 = arith.muli %6, %7 : index
            %11 = arith.index_cast %10 : index to i64
            %12 = arith.sitofp %11 : i64 to f32
            %13 = arith.divf %9, %12 : f32
            affine.store %13, %0[%arg1, %arg2, %arg3, %arg4] : memref<1x1x2x2xf32>
          }
        }
      }
    }
    return %0 : memref<1x1x2x2xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 4 , 4] , \22name\22 : \22dequantize_output\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 2 , 2] , \22name\22 : \22average_pool_output\22 }\0A\0A]\00"} : () -> ()
}
