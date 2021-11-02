#map0 = affine_map<(d0) -> (0, d0 * 2)>
#map1 = affine_map<(d0) -> (4, d0 * 2 + 3)>
#map2 = affine_map<(d0) -> (4, d0 * -2 + 4, d0 * 2 + 3, 3)>
module attributes {llvm.data_layout = "e"}  {
  llvm.func @omTensorListGetOmtArray(!llvm.ptr<i8>) -> !llvm.ptr<ptr<i8>>
  llvm.func @omTensorSetDataType(!llvm.ptr<i8>, i32)
  llvm.func @omTensorGetDataType(!llvm.ptr<i8>) -> i32
  llvm.func @omTensorGetStrides(!llvm.ptr<i8>) -> !llvm.ptr<i64>
  llvm.func @omTensorGetShape(!llvm.ptr<i8>) -> !llvm.ptr<i64>
  llvm.func @omTensorSetDataPtr(!llvm.ptr<i8>, i32, !llvm.ptr<i8>, !llvm.ptr<i8>)
  llvm.func @omTensorGetDataPtr(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  llvm.func @omTensorCreateEmptyDeprecated(i64) -> !llvm.ptr<i8>
  llvm.func @omTensorListCreateWithOwnership(!llvm.ptr<ptr<i8>>, i32, i32) -> !llvm.ptr<i8>
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @main_graph(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64) -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> attributes {input_names = ["dequantize_output"], output_names = ["average_pool_output"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %5 = llvm.insertvalue %arg7, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %7 = llvm.insertvalue %arg8, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %8 = llvm.insertvalue %arg5, %7[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %9 = llvm.insertvalue %arg9, %8[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %10 = llvm.insertvalue %arg6, %9[3, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %11 = llvm.insertvalue %arg10, %10[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f32
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = llvm.mlir.constant(2 : index) : i64
    %15 = llvm.mlir.constant(2 : index) : i64
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.mlir.constant(4 : index) : i64
    %18 = llvm.mlir.constant(4 : index) : i64
    %19 = llvm.mlir.constant(4 : index) : i64
    %20 = llvm.mlir.null : !llvm.ptr<f32>
    %21 = llvm.getelementptr %20[%19] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %22 = llvm.ptrtoint %21 : !llvm.ptr<f32> to i64
    %23 = llvm.mlir.constant(16 : index) : i64
    %24 = llvm.add %22, %23  : i64
    %25 = llvm.call @malloc(%24) : (i64) -> !llvm.ptr<i8>
    %26 = llvm.bitcast %25 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %27 = llvm.ptrtoint %26 : !llvm.ptr<f32> to i64
    %28 = llvm.mlir.constant(1 : index) : i64
    %29 = llvm.sub %23, %28  : i64
    %30 = llvm.add %27, %29  : i64
    %31 = llvm.urem %30, %23  : i64
    %32 = llvm.sub %30, %31  : i64
    %33 = llvm.inttoptr %32 : i64 to !llvm.ptr<f32>
    %34 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %35 = llvm.insertvalue %26, %34[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %36 = llvm.insertvalue %33, %35[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %37 = llvm.mlir.constant(0 : index) : i64
    %38 = llvm.insertvalue %37, %36[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %39 = llvm.insertvalue %12, %38[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %40 = llvm.insertvalue %13, %39[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %41 = llvm.insertvalue %14, %40[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %42 = llvm.insertvalue %15, %41[3, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %43 = llvm.insertvalue %18, %42[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %44 = llvm.insertvalue %17, %43[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %45 = llvm.insertvalue %15, %44[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %46 = llvm.insertvalue %16, %45[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %47 = builtin.unrealized_conversion_cast %46 : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> to memref<1x1x2x2xf32>
    affine.for %arg11 = 0 to 1 {
      affine.for %arg12 = 0 to 1 {
        affine.for %arg13 = 0 to 2 {
          affine.for %arg14 = 0 to 2 {
            %48 = llvm.mlir.constant(1 : index) : i64
            %49 = llvm.mlir.null : !llvm.ptr<f32>
            %50 = llvm.getelementptr %49[%48] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
            %51 = llvm.ptrtoint %50 : !llvm.ptr<f32> to i64
            %52 = llvm.alloca %51 x f32 : (i64) -> !llvm.ptr<f32>
            %53 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
            %54 = llvm.insertvalue %52, %53[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
            %55 = llvm.insertvalue %52, %54[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
            %56 = llvm.mlir.constant(0 : index) : i64
            %57 = llvm.insertvalue %56, %55[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
            %58 = builtin.unrealized_conversion_cast %57 : !llvm.struct<(ptr<f32>, ptr<f32>, i64)> to memref<f32>
            affine.store %cst, %58[] : memref<f32>
            %59 = affine.max #map0(%arg13)
            %60 = affine.min #map1(%arg13)
            %61 = affine.max #map0(%arg14)
            %62 = affine.min #map1(%arg14)
            %63 = arith.subi %60, %59 : index
            %64 = arith.subi %62, %61 : index
            affine.for %arg15 = 0 to min #map2(%arg13) {
              affine.for %arg16 = 0 to min #map2(%arg14) {
                %71 = arith.addi %arg15, %59 : index
                %72 = arith.addi %arg16, %61 : index
                %73 = builtin.unrealized_conversion_cast %arg11 : index to i64
                %74 = builtin.unrealized_conversion_cast %arg12 : index to i64
                %75 = builtin.unrealized_conversion_cast %71 : index to i64
                %76 = builtin.unrealized_conversion_cast %72 : index to i64
                %77 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
                %78 = llvm.mlir.constant(16 : index) : i64
                %79 = llvm.mul %73, %78  : i64
                %80 = llvm.mlir.constant(16 : index) : i64
                %81 = llvm.mul %74, %80  : i64
                %82 = llvm.add %79, %81  : i64
                %83 = llvm.mlir.constant(4 : index) : i64
                %84 = llvm.mul %75, %83  : i64
                %85 = llvm.add %82, %84  : i64
                %86 = llvm.add %85, %76  : i64
                %87 = llvm.getelementptr %77[%86] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
                %88 = llvm.load %87 : !llvm.ptr<f32>
                %89 = affine.load %58[] : memref<f32>
                %90 = arith.addf %89, %88 : f32
                affine.store %90, %58[] : memref<f32>
              }
            }
            %65 = affine.load %58[] : memref<f32>
            affine.store %65, %47[%arg11, %arg12, %arg13, %arg14] : memref<1x1x2x2xf32>
            %66 = affine.load %47[%arg11, %arg12, %arg13, %arg14] : memref<1x1x2x2xf32>
            %67 = arith.muli %63, %64 : index
            %68 = arith.index_cast %67 : index to i64
            %69 = arith.sitofp %68 : i64 to f32
            %70 = arith.divf %66, %69 : f32
            affine.store %70, %47[%arg11, %arg12, %arg13, %arg14] : memref<1x1x2x2xf32>
          }
        }
      }
    }
    llvm.return %46 : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
  }
  llvm.func @_mlir_ciface_main_graph(%arg0: !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>) attributes {input_names = ["dequantize_output"], output_names = ["average_pool_output"]} {
    %0 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %6 = llvm.extractvalue %0[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %7 = llvm.extractvalue %0[3, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %8 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %9 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %10 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %11 = llvm.extractvalue %0[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %12 = llvm.call @main_graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11) : (!llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    llvm.store %12, %arg0 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>
    llvm.return
  }
  llvm.mlir.global external constant @_in_signature("[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 4 , 4] , \22name\22 : \22dequantize_output\22 }\0A\0A]\00")
  llvm.mlir.global external constant @_out_signature("[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 2 , 2] , \22name\22 : \22average_pool_output\22 }\0A\0A]\00")
  llvm.func @omInputSignature() -> !llvm.ptr<i8> {
    %0 = llvm.mlir.addressof @_in_signature : !llvm.ptr<array<85 x i8>>
    %1 = llvm.bitcast %0 : !llvm.ptr<array<85 x i8>> to !llvm.ptr<i8>
    llvm.return %1 : !llvm.ptr<i8>
  }
  llvm.func @_mlir_ciface_omInputSignature() -> !llvm.ptr<i8> {
    %0 = llvm.call @omInputSignature() : () -> !llvm.ptr<i8>
    llvm.return %0 : !llvm.ptr<i8>
  }
  llvm.func @omOutputSignature() -> !llvm.ptr<i8> {
    %0 = llvm.mlir.addressof @_out_signature : !llvm.ptr<array<86 x i8>>
    %1 = llvm.bitcast %0 : !llvm.ptr<array<86 x i8>> to !llvm.ptr<i8>
    llvm.return %1 : !llvm.ptr<i8>
  }
  llvm.func @_mlir_ciface_omOutputSignature() -> !llvm.ptr<i8> {
    %0 = llvm.call @omOutputSignature() : () -> !llvm.ptr<i8>
    llvm.return %0 : !llvm.ptr<i8>
  }
  llvm.func @run_main_graph(%arg0: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
    %0 = llvm.call @omTensorListGetOmtArray(%arg0) : (!llvm.ptr<i8>) -> !llvm.ptr<ptr<i8>>
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.alloca %1 x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> : (i32) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.getelementptr %0[%3] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    %5 = llvm.load %4 : !llvm.ptr<ptr<i8>>
    %6 = llvm.alloca %1 x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)> : (i32) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>
    %7 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %8 = llvm.call @omTensorGetDataPtr(%5) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %9 = llvm.bitcast %8 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %10 = llvm.insertvalue %9, %7[0 : i32] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %11 = llvm.insertvalue %9, %10[1 : i32] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %12 = llvm.mlir.constant(0 : i64) : i64
    %13 = llvm.insertvalue %12, %11[2 : i32] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %14 = llvm.call @omTensorGetShape(%5) : (!llvm.ptr<i8>) -> !llvm.ptr<i64>
    %15 = llvm.call @omTensorGetStrides(%5) : (!llvm.ptr<i8>) -> !llvm.ptr<i64>
    %16 = llvm.mlir.constant(0 : i64) : i64
    %17 = llvm.getelementptr %14[%16] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %18 = llvm.load %17 : !llvm.ptr<i64>
    %19 = llvm.ptrtoint %18 : !llvm.ptr<i64> to i64
    %20 = llvm.insertvalue %19, %13[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %21 = llvm.getelementptr %15[%16] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %22 = llvm.load %21 : !llvm.ptr<i64>
    %23 = llvm.ptrtoint %22 : !llvm.ptr<i64> to i64
    %24 = llvm.insertvalue %23, %20[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %25 = llvm.mlir.constant(1 : i64) : i64
    %26 = llvm.getelementptr %14[%25] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %27 = llvm.load %26 : !llvm.ptr<i64>
    %28 = llvm.ptrtoint %27 : !llvm.ptr<i64> to i64
    %29 = llvm.insertvalue %28, %24[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %30 = llvm.getelementptr %15[%25] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %31 = llvm.load %30 : !llvm.ptr<i64>
    %32 = llvm.ptrtoint %31 : !llvm.ptr<i64> to i64
    %33 = llvm.insertvalue %32, %29[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %34 = llvm.mlir.constant(2 : i64) : i64
    %35 = llvm.getelementptr %14[%34] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %36 = llvm.load %35 : !llvm.ptr<i64>
    %37 = llvm.ptrtoint %36 : !llvm.ptr<i64> to i64
    %38 = llvm.insertvalue %37, %33[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %39 = llvm.getelementptr %15[%34] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %40 = llvm.load %39 : !llvm.ptr<i64>
    %41 = llvm.ptrtoint %40 : !llvm.ptr<i64> to i64
    %42 = llvm.insertvalue %41, %38[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %43 = llvm.mlir.constant(3 : i64) : i64
    %44 = llvm.getelementptr %14[%43] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %45 = llvm.load %44 : !llvm.ptr<i64>
    %46 = llvm.ptrtoint %45 : !llvm.ptr<i64> to i64
    %47 = llvm.insertvalue %46, %42[3, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %48 = llvm.getelementptr %15[%43] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %49 = llvm.load %48 : !llvm.ptr<i64>
    %50 = llvm.ptrtoint %49 : !llvm.ptr<i64> to i64
    %51 = llvm.insertvalue %50, %47[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    llvm.store %51, %6 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>
    llvm.call @_mlir_ciface_main_graph(%2, %6) : (!llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>, !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>) -> ()
    %52 = llvm.load %2 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>
    %53 = llvm.mlir.constant(1 : i64) : i32
    %54 = llvm.mlir.constant(8 : i64) : i64
    %55 = llvm.call @malloc(%54) : (i64) -> !llvm.ptr<i8>
    %56 = llvm.bitcast %55 : !llvm.ptr<i8> to !llvm.ptr<ptr<i8>>
    %57 = llvm.mlir.constant(4 : i64) : i64
    %58 = llvm.call @omTensorCreateEmptyDeprecated(%57) : (i64) -> !llvm.ptr<i8>
    %59 = llvm.mlir.constant(1 : i32) : i32
    %60 = llvm.extractvalue %52[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %61 = llvm.bitcast %60 : !llvm.ptr<f32> to !llvm.ptr<i8>
    %62 = llvm.extractvalue %52[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %63 = llvm.bitcast %62 : !llvm.ptr<f32> to !llvm.ptr<i8>
    llvm.call @omTensorSetDataPtr(%58, %59, %61, %63) : (!llvm.ptr<i8>, i32, !llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    %64 = llvm.mlir.constant(1 : i32) : i32
    llvm.call @omTensorSetDataType(%58, %64) : (!llvm.ptr<i8>, i32) -> ()
    %65 = llvm.call @omTensorGetShape(%58) : (!llvm.ptr<i8>) -> !llvm.ptr<i64>
    %66 = llvm.call @omTensorGetStrides(%58) : (!llvm.ptr<i8>) -> !llvm.ptr<i64>
    %67 = llvm.mlir.constant(0 : i64) : i64
    %68 = llvm.extractvalue %52[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %69 = llvm.getelementptr %65[%67] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %68, %69 : !llvm.ptr<i64>
    %70 = llvm.extractvalue %52[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %71 = llvm.getelementptr %66[%67] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %70, %71 : !llvm.ptr<i64>
    %72 = llvm.mlir.constant(1 : i64) : i64
    %73 = llvm.extractvalue %52[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %74 = llvm.getelementptr %65[%72] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %73, %74 : !llvm.ptr<i64>
    %75 = llvm.extractvalue %52[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %76 = llvm.getelementptr %66[%72] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %75, %76 : !llvm.ptr<i64>
    %77 = llvm.mlir.constant(2 : i64) : i64
    %78 = llvm.extractvalue %52[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %79 = llvm.getelementptr %65[%77] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %78, %79 : !llvm.ptr<i64>
    %80 = llvm.extractvalue %52[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %81 = llvm.getelementptr %66[%77] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %80, %81 : !llvm.ptr<i64>
    %82 = llvm.mlir.constant(3 : i64) : i64
    %83 = llvm.extractvalue %52[3, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %84 = llvm.getelementptr %65[%82] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %83, %84 : !llvm.ptr<i64>
    %85 = llvm.extractvalue %52[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %86 = llvm.getelementptr %66[%82] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %85, %86 : !llvm.ptr<i64>
    %87 = llvm.mlir.constant(0 : i32) : i32
    %88 = llvm.getelementptr %56[%87] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
    llvm.store %58, %88 : !llvm.ptr<ptr<i8>>
    %89 = llvm.call @omTensorListCreateWithOwnership(%56, %53, %1) : (!llvm.ptr<ptr<i8>>, i32, i32) -> !llvm.ptr<i8>
    llvm.return %89 : !llvm.ptr<i8>
  }
}

