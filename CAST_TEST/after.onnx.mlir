module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.compile_options" = "--EmitLLVMIR -O0 --omit-compile-info transpose_cast_transpose.mlir -o after", "onnx-mlir.compiler_version" = "onnx-mlir version 0.5.1 (b6d6a0e)", "onnx-mlir.op_stats" = "{\0A  \22func.return.4D\22 : 1,\0A  \22onnx.Cast.4D\22 : 1,\0A  \22onnx.Transpose.4D\22 : 2\0A}\0A", "onnx-mlir.symbol-postfix" = "after"} {
  llvm.mlir.global internal constant @om_compilation_info_json_after("{}\00") {addr_space = 0 : i32}
  llvm.func @strncmp(!llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.mlir.global internal constant @"om_Wrong size for the dimension 3 of the input 0: expect 4, but got %lld\0A_after"("Wrong size for the dimension 3 of the input 0: expect 4, but got %lld\0A") {addr_space = 0 : i32, alignment = 16 : i64}
  llvm.mlir.global internal constant @"om_Wrong size for the dimension 2 of the input 0: expect 3, but got %lld\0A_after"("Wrong size for the dimension 2 of the input 0: expect 3, but got %lld\0A") {addr_space = 0 : i32, alignment = 16 : i64}
  llvm.mlir.global internal constant @"om_Wrong size for the dimension 1 of the input 0: expect 2, but got %lld\0A_after"("Wrong size for the dimension 1 of the input 0: expect 2, but got %lld\0A") {addr_space = 0 : i32, alignment = 16 : i64}
  llvm.mlir.global internal constant @"om_Wrong size for the dimension 0 of the input 0: expect 1, but got %lld\0A_after"("Wrong size for the dimension 0 of the input 0: expect 1, but got %lld\0A") {addr_space = 0 : i32, alignment = 16 : i64}
  llvm.mlir.global internal constant @"om_Wrong rank for the input 0: expect 4, but got %lld\0A_after"("Wrong rank for the input 0: expect 4, but got %lld\0A") {addr_space = 0 : i32, alignment = 16 : i64}
  llvm.mlir.global internal constant @"om_Wrong data type for the input 0: expect f32\0A_after"("Wrong data type for the input 0: expect f32\0A") {addr_space = 0 : i32, alignment = 16 : i64}
  llvm.mlir.global internal constant @"om_Wrong number of input tensors: expect 1, but got %lld\0A_after"("Wrong number of input tensors: expect 1, but got %lld\0A") {addr_space = 0 : i32, alignment = 16 : i64}
  llvm.func @printf(!llvm.ptr, ...)
  llvm.func @__errno_location() -> !llvm.ptr
  llvm.mlir.global external constant @_entry_point_1_after("run_main_graph_after\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_1_in_sig_after("[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 2 , 3 , 4] , \22name\22 : \22input_0\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_1_out_sig_after("[   { \22type\22 : \22i64\22 , \22dims\22 : [1 , 4 , 2 , 3] , \22name\22 : \22output_0\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_0_after("run_main_graph\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_0_in_sig_after("[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 2 , 3 , 4] , \22name\22 : \22input_0\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_0_out_sig_after("[   { \22type\22 : \22i64\22 , \22dims\22 : [1 , 4 , 2 , 3] , \22name\22 : \22output_0\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.func @omGetExternalConstantAddr(!llvm.ptr, !llvm.ptr, i64)
  llvm.func @omUnloadConstantData(!llvm.ptr, i64) -> i1
  llvm.func @omLoadConstantData(!llvm.ptr, !llvm.ptr, i64, i64) -> i1
  llvm.func @omTensorListGetSize(!llvm.ptr) -> i64
  llvm.func @omTensorPrint(!llvm.ptr, !llvm.ptr)
  llvm.func @omTensorListGetOmtArray(!llvm.ptr) -> !llvm.ptr
  llvm.func @omTensorSetDataType(!llvm.ptr, i64)
  llvm.func @omTensorGetDataType(!llvm.ptr) -> i64
  llvm.func @omTensorGetStrides(!llvm.ptr) -> !llvm.ptr
  llvm.func @omTensorGetShape(!llvm.ptr) -> !llvm.ptr
  llvm.func @omTensorGetRank(!llvm.ptr) -> i64
  llvm.func @omTensorSetDataPtr(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr)
  llvm.func @omTensorGetDataPtr(!llvm.ptr) -> !llvm.ptr
  llvm.func @omTensorDestroy(!llvm.ptr)
  llvm.func @omTensorCreateUntyped(i64) -> !llvm.ptr
  llvm.func @omTensorListCreate(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main_graph_after(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64) -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(6 : index) : i64
    %1 = llvm.mlir.constant(12 : index) : i64
    %2 = llvm.mlir.constant(16 : index) : i64
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.mlir.constant(24 : index) : i64
    %5 = llvm.mlir.constant(8 : index) : i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(3 : index) : i64
    %9 = llvm.mlir.constant(4 : index) : i64
    %10 = llvm.mlir.constant(2 : index) : i64
    %11 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %12 = llvm.getelementptr %3[24] : (!llvm.ptr) -> !llvm.ptr, f32
    %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %14 = llvm.add %13, %2 : i64
    %15 = llvm.call @malloc(%14) : (i64) -> !llvm.ptr
    %16 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %17 = llvm.sub %2, %7 : i64
    %18 = llvm.add %16, %17 : i64
    %19 = llvm.urem %18, %2 : i64
    %20 = llvm.sub %18, %19 : i64
    %21 = llvm.inttoptr %20 : i64 to !llvm.ptr
    llvm.br ^bb1(%6 : i64)
  ^bb1(%22: i64):  // 2 preds: ^bb0, ^bb5
    %23 = llvm.icmp "slt" %22, %8 : i64
    llvm.cond_br %23, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%6 : i64)
  ^bb3(%24: i64):  // 2 preds: ^bb2, ^bb4
    %25 = llvm.icmp "slt" %24, %9 : i64
    llvm.cond_br %25, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %26 = llvm.mul %6, %4 overflow<nsw, nuw> : i64
    %27 = llvm.mul %6, %1 overflow<nsw, nuw> : i64
    %28 = llvm.add %26, %27 overflow<nsw, nuw> : i64
    %29 = llvm.mul %22, %9 overflow<nsw, nuw> : i64
    %30 = llvm.add %28, %29 overflow<nsw, nuw> : i64
    %31 = llvm.add %30, %24 overflow<nsw, nuw> : i64
    %32 = llvm.getelementptr inbounds|nuw %arg1[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %33 = llvm.load %32 : !llvm.ptr -> f32
    %34 = llvm.mul %6, %4 overflow<nsw, nuw> : i64
    %35 = llvm.mul %22, %5 overflow<nsw, nuw> : i64
    %36 = llvm.add %34, %35 overflow<nsw, nuw> : i64
    %37 = llvm.mul %24, %10 overflow<nsw, nuw> : i64
    %38 = llvm.add %36, %37 overflow<nsw, nuw> : i64
    %39 = llvm.add %38, %6 overflow<nsw, nuw> : i64
    %40 = llvm.getelementptr inbounds|nuw %21[%39] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %33, %40 : f32, !llvm.ptr
    %41 = llvm.mul %6, %4 overflow<nsw, nuw> : i64
    %42 = llvm.mul %7, %1 overflow<nsw, nuw> : i64
    %43 = llvm.add %41, %42 overflow<nsw, nuw> : i64
    %44 = llvm.mul %22, %9 overflow<nsw, nuw> : i64
    %45 = llvm.add %43, %44 overflow<nsw, nuw> : i64
    %46 = llvm.add %45, %24 overflow<nsw, nuw> : i64
    %47 = llvm.getelementptr inbounds|nuw %arg1[%46] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %48 = llvm.load %47 : !llvm.ptr -> f32
    %49 = llvm.mul %6, %4 overflow<nsw, nuw> : i64
    %50 = llvm.mul %22, %5 overflow<nsw, nuw> : i64
    %51 = llvm.add %49, %50 overflow<nsw, nuw> : i64
    %52 = llvm.mul %24, %10 overflow<nsw, nuw> : i64
    %53 = llvm.add %51, %52 overflow<nsw, nuw> : i64
    %54 = llvm.add %53, %7 overflow<nsw, nuw> : i64
    %55 = llvm.getelementptr inbounds|nuw %21[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %48, %55 : f32, !llvm.ptr
    %56 = llvm.add %24, %7 : i64
    llvm.br ^bb3(%56 : i64)
  ^bb5:  // pred: ^bb3
    %57 = llvm.add %22, %7 : i64
    llvm.br ^bb1(%57 : i64)
  ^bb6:  // pred: ^bb1
    %58 = llvm.getelementptr %3[24] : (!llvm.ptr) -> !llvm.ptr, i64
    %59 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %60 = llvm.add %59, %2 : i64
    %61 = llvm.call @malloc(%60) : (i64) -> !llvm.ptr
    %62 = llvm.ptrtoint %61 : !llvm.ptr to i64
    %63 = llvm.sub %2, %7 : i64
    %64 = llvm.add %62, %63 : i64
    %65 = llvm.urem %64, %2 : i64
    %66 = llvm.sub %64, %65 : i64
    %67 = llvm.inttoptr %66 : i64 to !llvm.ptr
    llvm.br ^bb7(%6 : i64)
  ^bb7(%68: i64):  // 2 preds: ^bb6, ^bb14
    %69 = llvm.icmp "slt" %68, %8 : i64
    llvm.cond_br %69, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%6 : i64)
  ^bb9(%70: i64):  // 2 preds: ^bb8, ^bb13
    %71 = llvm.icmp "slt" %70, %9 : i64
    llvm.cond_br %71, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%6 : i64)
  ^bb11(%72: i64):  // 2 preds: ^bb10, ^bb12
    %73 = llvm.icmp "slt" %72, %10 : i64
    llvm.cond_br %73, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %74 = llvm.mul %6, %4 overflow<nsw, nuw> : i64
    %75 = llvm.mul %68, %5 overflow<nsw, nuw> : i64
    %76 = llvm.add %74, %75 overflow<nsw, nuw> : i64
    %77 = llvm.mul %70, %10 overflow<nsw, nuw> : i64
    %78 = llvm.add %76, %77 overflow<nsw, nuw> : i64
    %79 = llvm.add %78, %72 overflow<nsw, nuw> : i64
    %80 = llvm.getelementptr inbounds|nuw %21[%79] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %81 = llvm.load %80 : !llvm.ptr -> f32
    %82 = llvm.fptosi %81 : f32 to i64
    %83 = llvm.mul %6, %4 overflow<nsw, nuw> : i64
    %84 = llvm.mul %68, %5 overflow<nsw, nuw> : i64
    %85 = llvm.add %83, %84 overflow<nsw, nuw> : i64
    %86 = llvm.mul %70, %10 overflow<nsw, nuw> : i64
    %87 = llvm.add %85, %86 overflow<nsw, nuw> : i64
    %88 = llvm.add %87, %72 overflow<nsw, nuw> : i64
    %89 = llvm.getelementptr inbounds|nuw %67[%88] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %82, %89 : i64, !llvm.ptr
    %90 = llvm.add %72, %7 : i64
    llvm.br ^bb11(%90 : i64)
  ^bb13:  // pred: ^bb11
    %91 = llvm.add %70, %7 : i64
    llvm.br ^bb9(%91 : i64)
  ^bb14:  // pred: ^bb9
    %92 = llvm.add %68, %7 : i64
    llvm.br ^bb7(%92 : i64)
  ^bb15:  // pred: ^bb7
    llvm.call @free(%15) : (!llvm.ptr) -> ()
    %93 = llvm.getelementptr %3[24] : (!llvm.ptr) -> !llvm.ptr, i64
    %94 = llvm.ptrtoint %93 : !llvm.ptr to i64
    %95 = llvm.add %94, %2 : i64
    %96 = llvm.call @malloc(%95) : (i64) -> !llvm.ptr
    %97 = llvm.ptrtoint %96 : !llvm.ptr to i64
    %98 = llvm.sub %2, %7 : i64
    %99 = llvm.add %97, %98 : i64
    %100 = llvm.urem %99, %2 : i64
    %101 = llvm.sub %99, %100 : i64
    %102 = llvm.inttoptr %101 : i64 to !llvm.ptr
    %103 = llvm.insertvalue %96, %11[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %104 = llvm.insertvalue %102, %103[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %105 = llvm.insertvalue %6, %104[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %106 = llvm.insertvalue %7, %105[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %107 = llvm.insertvalue %9, %106[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %108 = llvm.insertvalue %10, %107[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %109 = llvm.insertvalue %8, %108[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %110 = llvm.insertvalue %4, %109[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %111 = llvm.insertvalue %0, %110[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %112 = llvm.insertvalue %8, %111[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %113 = llvm.insertvalue %7, %112[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    llvm.br ^bb16(%6 : i64)
  ^bb16(%114: i64):  // 2 preds: ^bb15, ^bb20
    %115 = llvm.icmp "slt" %114, %9 : i64
    llvm.cond_br %115, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    llvm.br ^bb18(%6 : i64)
  ^bb18(%116: i64):  // 2 preds: ^bb17, ^bb19
    %117 = llvm.icmp "slt" %116, %10 : i64
    llvm.cond_br %117, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %118 = llvm.mul %6, %4 overflow<nsw, nuw> : i64
    %119 = llvm.mul %6, %5 overflow<nsw, nuw> : i64
    %120 = llvm.add %118, %119 overflow<nsw, nuw> : i64
    %121 = llvm.mul %114, %10 overflow<nsw, nuw> : i64
    %122 = llvm.add %120, %121 overflow<nsw, nuw> : i64
    %123 = llvm.add %122, %116 overflow<nsw, nuw> : i64
    %124 = llvm.getelementptr inbounds|nuw %67[%123] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %125 = llvm.load %124 : !llvm.ptr -> i64
    %126 = llvm.mul %6, %4 overflow<nsw, nuw> : i64
    %127 = llvm.mul %114, %0 overflow<nsw, nuw> : i64
    %128 = llvm.add %126, %127 overflow<nsw, nuw> : i64
    %129 = llvm.mul %116, %8 overflow<nsw, nuw> : i64
    %130 = llvm.add %128, %129 overflow<nsw, nuw> : i64
    %131 = llvm.add %130, %6 overflow<nsw, nuw> : i64
    %132 = llvm.getelementptr inbounds|nuw %102[%131] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %125, %132 : i64, !llvm.ptr
    %133 = llvm.mul %6, %4 overflow<nsw, nuw> : i64
    %134 = llvm.mul %7, %5 overflow<nsw, nuw> : i64
    %135 = llvm.add %133, %134 overflow<nsw, nuw> : i64
    %136 = llvm.mul %114, %10 overflow<nsw, nuw> : i64
    %137 = llvm.add %135, %136 overflow<nsw, nuw> : i64
    %138 = llvm.add %137, %116 overflow<nsw, nuw> : i64
    %139 = llvm.getelementptr inbounds|nuw %67[%138] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %140 = llvm.load %139 : !llvm.ptr -> i64
    %141 = llvm.mul %6, %4 overflow<nsw, nuw> : i64
    %142 = llvm.mul %114, %0 overflow<nsw, nuw> : i64
    %143 = llvm.add %141, %142 overflow<nsw, nuw> : i64
    %144 = llvm.mul %116, %8 overflow<nsw, nuw> : i64
    %145 = llvm.add %143, %144 overflow<nsw, nuw> : i64
    %146 = llvm.add %145, %7 overflow<nsw, nuw> : i64
    %147 = llvm.getelementptr inbounds|nuw %102[%146] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %140, %147 : i64, !llvm.ptr
    %148 = llvm.mul %6, %4 overflow<nsw, nuw> : i64
    %149 = llvm.mul %10, %5 overflow<nsw, nuw> : i64
    %150 = llvm.add %148, %149 overflow<nsw, nuw> : i64
    %151 = llvm.mul %114, %10 overflow<nsw, nuw> : i64
    %152 = llvm.add %150, %151 overflow<nsw, nuw> : i64
    %153 = llvm.add %152, %116 overflow<nsw, nuw> : i64
    %154 = llvm.getelementptr inbounds|nuw %67[%153] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %155 = llvm.load %154 : !llvm.ptr -> i64
    %156 = llvm.mul %6, %4 overflow<nsw, nuw> : i64
    %157 = llvm.mul %114, %0 overflow<nsw, nuw> : i64
    %158 = llvm.add %156, %157 overflow<nsw, nuw> : i64
    %159 = llvm.mul %116, %8 overflow<nsw, nuw> : i64
    %160 = llvm.add %158, %159 overflow<nsw, nuw> : i64
    %161 = llvm.add %160, %10 overflow<nsw, nuw> : i64
    %162 = llvm.getelementptr inbounds|nuw %102[%161] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %155, %162 : i64, !llvm.ptr
    %163 = llvm.add %116, %7 : i64
    llvm.br ^bb18(%163 : i64)
  ^bb20:  // pred: ^bb18
    %164 = llvm.add %114, %7 : i64
    llvm.br ^bb16(%164 : i64)
  ^bb21:  // pred: ^bb16
    llvm.call @free(%61) : (!llvm.ptr) -> ()
    llvm.return %113 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  }
  llvm.func @_mlir_ciface_main_graph_after(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %6 = llvm.extractvalue %0[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %7 = llvm.extractvalue %0[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %8 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %9 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %10 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %11 = llvm.extractvalue %0[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %12 = llvm.call @main_graph_after(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    llvm.store %12, %arg0 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @run_main_graph_after(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.constant(7 : i64) : i64
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %3 = llvm.mlir.addressof @"om_Wrong size for the dimension 3 of the input 0: expect 4, but got %lld\0A_after" : !llvm.ptr
    %4 = llvm.mlir.addressof @"om_Wrong size for the dimension 2 of the input 0: expect 3, but got %lld\0A_after" : !llvm.ptr
    %5 = llvm.mlir.constant(3 : i64) : i64
    %6 = llvm.mlir.addressof @"om_Wrong size for the dimension 1 of the input 0: expect 2, but got %lld\0A_after" : !llvm.ptr
    %7 = llvm.mlir.constant(2 : i64) : i64
    %8 = llvm.mlir.addressof @"om_Wrong size for the dimension 0 of the input 0: expect 1, but got %lld\0A_after" : !llvm.ptr
    %9 = llvm.mlir.addressof @"om_Wrong rank for the input 0: expect 4, but got %lld\0A_after" : !llvm.ptr
    %10 = llvm.mlir.constant(4 : i64) : i64
    %11 = llvm.mlir.addressof @"om_Wrong data type for the input 0: expect f32\0A_after" : !llvm.ptr
    %12 = llvm.mlir.zero : !llvm.ptr
    %13 = llvm.mlir.constant(22 : i32) : i32
    %14 = llvm.mlir.addressof @"om_Wrong number of input tensors: expect 1, but got %lld\0A_after" : !llvm.ptr
    %15 = llvm.mlir.constant(1 : i64) : i64
    %16 = llvm.call @omTensorListGetSize(%arg0) : (!llvm.ptr) -> i64
    %17 = llvm.icmp "ne" %15, %16 : i64
    llvm.cond_br %17, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.call @printf(%14, %16) vararg(!llvm.func<void (ptr, i64, ...)>) : (!llvm.ptr, i64) -> ()
    %18 = llvm.call @__errno_location() : () -> !llvm.ptr
    llvm.store %13, %18 : i32, !llvm.ptr
    llvm.return %12 : !llvm.ptr
  ^bb2:  // pred: ^bb0
    %19 = llvm.call @omTensorListGetOmtArray(%arg0) : (!llvm.ptr) -> !llvm.ptr
    %20 = llvm.load %19 : !llvm.ptr -> !llvm.ptr
    %21 = llvm.call @omTensorGetDataType(%20) : (!llvm.ptr) -> i64
    %22 = llvm.icmp "ne" %15, %21 : i64
    llvm.cond_br %22, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.call @printf(%11) vararg(!llvm.func<void (ptr, ...)>) : (!llvm.ptr) -> ()
    %23 = llvm.call @__errno_location() : () -> !llvm.ptr
    llvm.store %13, %23 : i32, !llvm.ptr
    llvm.return %12 : !llvm.ptr
  ^bb4:  // pred: ^bb2
    %24 = llvm.call @omTensorGetRank(%20) : (!llvm.ptr) -> i64
    %25 = llvm.icmp "ne" %10, %24 : i64
    llvm.cond_br %25, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @printf(%9, %24) vararg(!llvm.func<void (ptr, i64, ...)>) : (!llvm.ptr, i64) -> ()
    %26 = llvm.call @__errno_location() : () -> !llvm.ptr
    llvm.store %13, %26 : i32, !llvm.ptr
    llvm.return %12 : !llvm.ptr
  ^bb6:  // pred: ^bb4
    %27 = llvm.call @omTensorGetShape(%20) : (!llvm.ptr) -> !llvm.ptr
    %28 = llvm.load %27 : !llvm.ptr -> i64
    %29 = llvm.icmp "ne" %15, %28 : i64
    llvm.cond_br %29, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    llvm.call @printf(%8, %28) vararg(!llvm.func<void (ptr, i64, ...)>) : (!llvm.ptr, i64) -> ()
    %30 = llvm.call @__errno_location() : () -> !llvm.ptr
    llvm.store %13, %30 : i32, !llvm.ptr
    llvm.return %12 : !llvm.ptr
  ^bb8:  // pred: ^bb6
    %31 = llvm.getelementptr %27[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %32 = llvm.load %31 : !llvm.ptr -> i64
    %33 = llvm.icmp "ne" %7, %32 : i64
    llvm.cond_br %33, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    llvm.call @printf(%6, %32) vararg(!llvm.func<void (ptr, i64, ...)>) : (!llvm.ptr, i64) -> ()
    %34 = llvm.call @__errno_location() : () -> !llvm.ptr
    llvm.store %13, %34 : i32, !llvm.ptr
    llvm.return %12 : !llvm.ptr
  ^bb10:  // pred: ^bb8
    %35 = llvm.getelementptr %27[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %36 = llvm.load %35 : !llvm.ptr -> i64
    %37 = llvm.icmp "ne" %5, %36 : i64
    llvm.cond_br %37, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    llvm.call @printf(%4, %36) vararg(!llvm.func<void (ptr, i64, ...)>) : (!llvm.ptr, i64) -> ()
    %38 = llvm.call @__errno_location() : () -> !llvm.ptr
    llvm.store %13, %38 : i32, !llvm.ptr
    llvm.return %12 : !llvm.ptr
  ^bb12:  // pred: ^bb10
    %39 = llvm.getelementptr %27[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %40 = llvm.load %39 : !llvm.ptr -> i64
    %41 = llvm.icmp "ne" %10, %40 : i64
    llvm.cond_br %41, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    llvm.call @printf(%3, %40) vararg(!llvm.func<void (ptr, i64, ...)>) : (!llvm.ptr, i64) -> ()
    %42 = llvm.call @__errno_location() : () -> !llvm.ptr
    llvm.store %13, %42 : i32, !llvm.ptr
    llvm.return %12 : !llvm.ptr
  ^bb14:  // pred: ^bb12
    %43 = llvm.call @omTensorListGetOmtArray(%arg0) : (!llvm.ptr) -> !llvm.ptr
    %44 = llvm.alloca %15 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    %45 = llvm.load %43 : !llvm.ptr -> !llvm.ptr
    %46 = llvm.alloca %15 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    %47 = llvm.call @omTensorGetDataPtr(%45) : (!llvm.ptr) -> !llvm.ptr
    %48 = llvm.insertvalue %47, %2[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %49 = llvm.insertvalue %47, %48[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %50 = llvm.insertvalue %1, %49[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %51 = llvm.call @omTensorGetShape(%45) : (!llvm.ptr) -> !llvm.ptr
    %52 = llvm.call @omTensorGetStrides(%45) : (!llvm.ptr) -> !llvm.ptr
    %53 = llvm.load %51 : !llvm.ptr -> i64
    %54 = llvm.insertvalue %53, %50[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %55 = llvm.load %52 : !llvm.ptr -> i64
    %56 = llvm.insertvalue %55, %54[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %57 = llvm.getelementptr %51[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %58 = llvm.load %57 : !llvm.ptr -> i64
    %59 = llvm.insertvalue %58, %56[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %60 = llvm.getelementptr %52[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %61 = llvm.load %60 : !llvm.ptr -> i64
    %62 = llvm.insertvalue %61, %59[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %63 = llvm.getelementptr %51[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %64 = llvm.load %63 : !llvm.ptr -> i64
    %65 = llvm.insertvalue %64, %62[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %66 = llvm.getelementptr %52[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %67 = llvm.load %66 : !llvm.ptr -> i64
    %68 = llvm.insertvalue %67, %65[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %69 = llvm.getelementptr %51[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %70 = llvm.load %69 : !llvm.ptr -> i64
    %71 = llvm.insertvalue %70, %68[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %72 = llvm.getelementptr %52[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %73 = llvm.load %72 : !llvm.ptr -> i64
    %74 = llvm.insertvalue %73, %71[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    llvm.store %74, %46 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    llvm.call @_mlir_ciface_main_graph_after(%44, %46) : (!llvm.ptr, !llvm.ptr) -> ()
    %75 = llvm.load %44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %76 = llvm.alloca %15 x !llvm.ptr : (i64) -> !llvm.ptr
    %77 = llvm.call @omTensorCreateUntyped(%10) : (i64) -> !llvm.ptr
    %78 = llvm.extractvalue %75[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %79 = llvm.extractvalue %75[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    llvm.call @omTensorSetDataPtr(%77, %15, %78, %79) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @omTensorSetDataType(%77, %0) : (!llvm.ptr, i64) -> ()
    %80 = llvm.call @omTensorGetShape(%77) : (!llvm.ptr) -> !llvm.ptr
    %81 = llvm.call @omTensorGetStrides(%77) : (!llvm.ptr) -> !llvm.ptr
    %82 = llvm.extractvalue %75[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    llvm.store %82, %80 : i64, !llvm.ptr
    %83 = llvm.extractvalue %75[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    llvm.store %83, %81 : i64, !llvm.ptr
    %84 = llvm.extractvalue %75[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %85 = llvm.getelementptr %80[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %84, %85 : i64, !llvm.ptr
    %86 = llvm.extractvalue %75[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %87 = llvm.getelementptr %81[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %86, %87 : i64, !llvm.ptr
    %88 = llvm.extractvalue %75[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %89 = llvm.getelementptr %80[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %88, %89 : i64, !llvm.ptr
    %90 = llvm.extractvalue %75[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %91 = llvm.getelementptr %81[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %90, %91 : i64, !llvm.ptr
    %92 = llvm.extractvalue %75[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %93 = llvm.getelementptr %80[3] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %92, %93 : i64, !llvm.ptr
    %94 = llvm.extractvalue %75[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %95 = llvm.getelementptr %81[3] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %94, %95 : i64, !llvm.ptr
    llvm.store %77, %76 : !llvm.ptr, !llvm.ptr
    %96 = llvm.call @omTensorListCreate(%76, %15) : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.return %96 : !llvm.ptr
  }
  llvm.func @run_main_graph(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @run_main_graph_after(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global internal constant @_entry_point_arrays_after() {addr_space = 0 : i32} : !llvm.array<3 x ptr> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @_entry_point_1_after : !llvm.ptr
    %2 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %3 = llvm.mlir.addressof @_entry_point_0_after : !llvm.ptr
    %4 = llvm.insertvalue %3, %2[0] : !llvm.array<3 x ptr>
    %5 = llvm.insertvalue %1, %4[1] : !llvm.array<3 x ptr>
    %6 = llvm.insertvalue %0, %5[2] : !llvm.array<3 x ptr>
    llvm.return %6 : !llvm.array<3 x ptr>
  }
  llvm.func @omQueryEntryPoints_after(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.addressof @_entry_point_arrays_after : !llvm.ptr
    %1 = llvm.mlir.constant(2 : i64) : i64
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.icmp "ne" %arg0, %2 : !llvm.ptr
    llvm.cond_br %3, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.store %1, %arg0 : i64, !llvm.ptr
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omQueryEntryPoints(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @omQueryEntryPoints_after(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omInputSignature_after(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @_entry_point_1_in_sig_after : !llvm.ptr
    %2 = llvm.mlir.constant(21 : i64) : i64
    %3 = llvm.mlir.addressof @_entry_point_1_after : !llvm.ptr
    %4 = llvm.mlir.addressof @_entry_point_0_in_sig_after : !llvm.ptr
    %5 = llvm.mlir.constant(15 : i64) : i64
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.addressof @_entry_point_0_after : !llvm.ptr
    %8 = llvm.call @strncmp(%arg0, %7, %5) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %9 = llvm.icmp "eq" %8, %6 : i32
    llvm.cond_br %9, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.return %4 : !llvm.ptr
  ^bb2:  // pred: ^bb0
    %10 = llvm.call @strncmp(%arg0, %3, %2) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %11 = llvm.icmp "eq" %10, %6 : i32
    llvm.cond_br %11, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.return %1 : !llvm.ptr
  ^bb4:  // pred: ^bb2
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omInputSignature(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @omInputSignature_after(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omOutputSignature_after(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @_entry_point_1_out_sig_after : !llvm.ptr
    %2 = llvm.mlir.constant(21 : i64) : i64
    %3 = llvm.mlir.addressof @_entry_point_1_after : !llvm.ptr
    %4 = llvm.mlir.addressof @_entry_point_0_out_sig_after : !llvm.ptr
    %5 = llvm.mlir.constant(15 : i64) : i64
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.addressof @_entry_point_0_after : !llvm.ptr
    %8 = llvm.call @strncmp(%arg0, %7, %5) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %9 = llvm.icmp "eq" %8, %6 : i32
    llvm.cond_br %9, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.return %4 : !llvm.ptr
  ^bb2:  // pred: ^bb0
    %10 = llvm.call @strncmp(%arg0, %3, %2) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %11 = llvm.icmp "eq" %10, %6 : i32
    llvm.cond_br %11, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.return %1 : !llvm.ptr
  ^bb4:  // pred: ^bb2
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omOutputSignature(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @omOutputSignature_after(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omCompilationInfo_after() -> !llvm.ptr {
    %0 = llvm.mlir.addressof @om_compilation_info_json_after : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omCompilationInfo() -> !llvm.ptr {
    %0 = llvm.call @omCompilationInfo_after() : () -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
}
