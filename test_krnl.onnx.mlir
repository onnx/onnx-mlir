module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "test_krnl"} {
  llvm.func @strncmp(!llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.mlir.global external constant @_entry_point_1_test_krnl("run_main_graph_test_krnl\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_1_in_sig_test_krnl("[    { \22type\22 : \22f32\22 , \22dims\22 : [2 , 3] , \22name\22 : \22input_0\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [3 , 4] , \22name\22 : \22input_1\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_1_out_sig_test_krnl("[   { \22type\22 : \22f32\22 , \22dims\22 : [2 , 4] , \22name\22 : \22output_0\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_0_test_krnl("run_main_graph\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_0_in_sig_test_krnl("[    { \22type\22 : \22f32\22 , \22dims\22 : [2 , 3] , \22name\22 : \22input_0\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [3 , 4] , \22name\22 : \22input_1\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_0_out_sig_test_krnl("[   { \22type\22 : \22f32\22 , \22dims\22 : [2 , 4] , \22name\22 : \22output_0\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.func @omGetExternalConstantAddr(!llvm.ptr, !llvm.ptr, i64)
  llvm.func @omMMapBinaryFile(!llvm.ptr, !llvm.ptr, i64, i64) -> i1
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
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main_graph_test_krnl(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(16 : index) : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(2 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(4 : index) : i64
    %7 = llvm.mlir.constant(3 : index) : i64
    %8 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.getelementptr %1[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %10 = llvm.ptrtoint %9 : !llvm.ptr to i64
    %11 = llvm.add %10, %0 : i64
    %12 = llvm.call @malloc(%11) : (i64) -> !llvm.ptr
    %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %14 = llvm.sub %0, %5 : i64
    %15 = llvm.add %13, %14 : i64
    %16 = llvm.urem %15, %0 : i64
    %17 = llvm.sub %15, %16 : i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    %19 = llvm.insertvalue %12, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %18, %19[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %3, %20[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %4, %21[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %6, %22[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.insertvalue %6, %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %5, %24[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%3 : i64)
  ^bb1(%26: i64):  // 2 preds: ^bb0, ^bb8
    %27 = llvm.icmp "slt" %26, %4 : i64
    llvm.cond_br %27, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%3 : i64)
  ^bb3(%28: i64):  // 2 preds: ^bb2, ^bb7
    %29 = llvm.icmp "slt" %28, %6 : i64
    llvm.cond_br %29, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%3, %2 : i64, f32)
  ^bb5(%30: i64, %31: f32):  // 2 preds: ^bb4, ^bb6
    %32 = llvm.icmp "slt" %30, %7 : i64
    llvm.cond_br %32, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %33 = llvm.mul %26, %7 overflow<nsw, nuw> : i64
    %34 = llvm.add %33, %30 overflow<nsw, nuw> : i64
    %35 = llvm.getelementptr inbounds|nuw %arg1[%34] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %36 = llvm.load %35 : !llvm.ptr -> f32
    %37 = llvm.mul %30, %6 overflow<nsw, nuw> : i64
    %38 = llvm.add %37, %28 overflow<nsw, nuw> : i64
    %39 = llvm.getelementptr inbounds|nuw %arg8[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %40 = llvm.load %39 : !llvm.ptr -> f32
    %41 = llvm.fmul %36, %40 : f32
    %42 = llvm.fadd %31, %41 : f32
    %43 = llvm.add %30, %5 : i64
    llvm.br ^bb5(%43, %42 : i64, f32)
  ^bb7:  // pred: ^bb5
    %44 = llvm.mul %26, %6 overflow<nsw, nuw> : i64
    %45 = llvm.add %44, %28 overflow<nsw, nuw> : i64
    %46 = llvm.getelementptr inbounds|nuw %18[%45] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %31, %46 : f32, !llvm.ptr
    %47 = llvm.add %28, %5 : i64
    llvm.br ^bb3(%47 : i64)
  ^bb8:  // pred: ^bb3
    %48 = llvm.add %26, %5 : i64
    llvm.br ^bb1(%48 : i64)
  ^bb9:  // pred: ^bb1
    llvm.return %25 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
  llvm.func @_mlir_ciface_main_graph_test_krnl(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.extractvalue %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.extractvalue %8[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.call @main_graph_test_krnl(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.store %16, %arg0 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @run_main_graph_test_krnl(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.constant(2 : i64) : i64
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.call @omTensorListGetOmtArray(%arg0) : (!llvm.ptr) -> !llvm.ptr
    %5 = llvm.alloca %3 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    %6 = llvm.load %4 : !llvm.ptr -> !llvm.ptr
    %7 = llvm.alloca %3 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    %8 = llvm.call @omTensorGetDataPtr(%6) : (!llvm.ptr) -> !llvm.ptr
    %9 = llvm.insertvalue %8, %2[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %8, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %1, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.call @omTensorGetShape(%6) : (!llvm.ptr) -> !llvm.ptr
    %13 = llvm.call @omTensorGetStrides(%6) : (!llvm.ptr) -> !llvm.ptr
    %14 = llvm.load %12 : !llvm.ptr -> i64
    %15 = llvm.insertvalue %14, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.load %13 : !llvm.ptr -> i64
    %17 = llvm.insertvalue %16, %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.getelementptr %12[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %19 = llvm.load %18 : !llvm.ptr -> i64
    %20 = llvm.insertvalue %19, %17[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.getelementptr %13[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %22 = llvm.load %21 : !llvm.ptr -> i64
    %23 = llvm.insertvalue %22, %20[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.store %23, %7 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %24 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    %25 = llvm.load %24 : !llvm.ptr -> !llvm.ptr
    %26 = llvm.alloca %3 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    %27 = llvm.call @omTensorGetDataPtr(%25) : (!llvm.ptr) -> !llvm.ptr
    %28 = llvm.insertvalue %27, %2[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.insertvalue %27, %28[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.insertvalue %1, %29[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.call @omTensorGetShape(%25) : (!llvm.ptr) -> !llvm.ptr
    %32 = llvm.call @omTensorGetStrides(%25) : (!llvm.ptr) -> !llvm.ptr
    %33 = llvm.load %31 : !llvm.ptr -> i64
    %34 = llvm.insertvalue %33, %30[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.load %32 : !llvm.ptr -> i64
    %36 = llvm.insertvalue %35, %34[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.getelementptr %31[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %38 = llvm.load %37 : !llvm.ptr -> i64
    %39 = llvm.insertvalue %38, %36[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.getelementptr %32[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %41 = llvm.load %40 : !llvm.ptr -> i64
    %42 = llvm.insertvalue %41, %39[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.store %42, %26 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.call @_mlir_ciface_main_graph_test_krnl(%5, %7, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %43 = llvm.load %5 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %44 = llvm.alloca %3 x !llvm.ptr : (i64) -> !llvm.ptr
    %45 = llvm.call @omTensorCreateUntyped(%0) : (i64) -> !llvm.ptr
    %46 = llvm.extractvalue %43[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %47 = llvm.extractvalue %43[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @omTensorSetDataPtr(%45, %3, %46, %47) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @omTensorSetDataType(%45, %3) : (!llvm.ptr, i64) -> ()
    %48 = llvm.call @omTensorGetShape(%45) : (!llvm.ptr) -> !llvm.ptr
    %49 = llvm.call @omTensorGetStrides(%45) : (!llvm.ptr) -> !llvm.ptr
    %50 = llvm.extractvalue %43[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.store %50, %48 : i64, !llvm.ptr
    %51 = llvm.extractvalue %43[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.store %51, %49 : i64, !llvm.ptr
    %52 = llvm.extractvalue %43[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.getelementptr %48[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %52, %53 : i64, !llvm.ptr
    %54 = llvm.extractvalue %43[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.getelementptr %49[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %54, %55 : i64, !llvm.ptr
    llvm.store %45, %44 : !llvm.ptr, !llvm.ptr
    %56 = llvm.call @omTensorListCreate(%44, %3) : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.return %56 : !llvm.ptr
  }
  llvm.func @run_main_graph(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @run_main_graph_test_krnl(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global internal constant @_entry_point_arrays_test_krnl() {addr_space = 0 : i32} : !llvm.array<3 x ptr> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @_entry_point_1_test_krnl : !llvm.ptr
    %2 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %3 = llvm.mlir.addressof @_entry_point_0_test_krnl : !llvm.ptr
    %4 = llvm.insertvalue %3, %2[0] : !llvm.array<3 x ptr> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.array<3 x ptr> 
    %6 = llvm.insertvalue %0, %5[2] : !llvm.array<3 x ptr> 
    llvm.return %6 : !llvm.array<3 x ptr>
  }
  llvm.func @omQueryEntryPoints_test_krnl(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.addressof @_entry_point_arrays_test_krnl : !llvm.ptr
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
    %0 = llvm.call @omQueryEntryPoints_test_krnl(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omInputSignature_test_krnl(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @_entry_point_1_in_sig_test_krnl : !llvm.ptr
    %2 = llvm.mlir.constant(25 : i64) : i64
    %3 = llvm.mlir.addressof @_entry_point_1_test_krnl : !llvm.ptr
    %4 = llvm.mlir.addressof @_entry_point_0_in_sig_test_krnl : !llvm.ptr
    %5 = llvm.mlir.constant(15 : i64) : i64
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.addressof @_entry_point_0_test_krnl : !llvm.ptr
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
    %0 = llvm.call @omInputSignature_test_krnl(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omOutputSignature_test_krnl(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @_entry_point_1_out_sig_test_krnl : !llvm.ptr
    %2 = llvm.mlir.constant(25 : i64) : i64
    %3 = llvm.mlir.addressof @_entry_point_1_test_krnl : !llvm.ptr
    %4 = llvm.mlir.addressof @_entry_point_0_out_sig_test_krnl : !llvm.ptr
    %5 = llvm.mlir.constant(15 : i64) : i64
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.addressof @_entry_point_0_test_krnl : !llvm.ptr
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
    %0 = llvm.call @omOutputSignature_test_krnl(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
}
