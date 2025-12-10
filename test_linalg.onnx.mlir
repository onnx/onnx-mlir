module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "test_linalg"} {
  llvm.func @strncmp(!llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.mlir.global external constant @_entry_point_1_test_linalg("run_main_graph_test_linalg\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_1_in_sig_test_linalg("[ ]") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_1_out_sig_test_linalg("[ ]") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_0_test_linalg("run_main_graph\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_0_in_sig_test_linalg("[ ]") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_0_out_sig_test_linalg("[ ]") {addr_space = 0 : i32}
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
  llvm.func @main_graph_test_linalg(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(64 : index) : i64
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
  ^bb1(%26: i64):  // 2 preds: ^bb0, ^bb5
    %27 = llvm.icmp "slt" %26, %4 : i64
    llvm.cond_br %27, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%3 : i64)
  ^bb3(%28: i64):  // 2 preds: ^bb2, ^bb4
    %29 = llvm.icmp "slt" %28, %6 : i64
    llvm.cond_br %29, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %30 = llvm.mul %26, %6 overflow<nsw, nuw> : i64
    %31 = llvm.add %30, %28 overflow<nsw, nuw> : i64
    %32 = llvm.getelementptr inbounds|nuw %18[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %32 : f32, !llvm.ptr
    %33 = llvm.add %28, %5 : i64
    llvm.br ^bb3(%33 : i64)
  ^bb5:  // pred: ^bb3
    %34 = llvm.add %26, %5 : i64
    llvm.br ^bb1(%34 : i64)
  ^bb6:  // pred: ^bb1
    llvm.br ^bb7(%3 : i64)
  ^bb7(%35: i64):  // 2 preds: ^bb6, ^bb14
    %36 = llvm.icmp "slt" %35, %4 : i64
    llvm.cond_br %36, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%3 : i64)
  ^bb9(%37: i64):  // 2 preds: ^bb8, ^bb13
    %38 = llvm.icmp "slt" %37, %6 : i64
    llvm.cond_br %38, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%3 : i64)
  ^bb11(%39: i64):  // 2 preds: ^bb10, ^bb12
    %40 = llvm.icmp "slt" %39, %7 : i64
    llvm.cond_br %40, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %41 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %42 = llvm.mul %35, %arg5 overflow<nsw, nuw> : i64
    %43 = llvm.mul %39, %arg6 overflow<nsw, nuw> : i64
    %44 = llvm.add %42, %43 overflow<nsw, nuw> : i64
    %45 = llvm.getelementptr inbounds|nuw %41[%44] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %46 = llvm.load %45 : !llvm.ptr -> f32
    %47 = llvm.getelementptr %arg8[%arg9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %48 = llvm.mul %39, %arg12 overflow<nsw, nuw> : i64
    %49 = llvm.mul %37, %arg13 overflow<nsw, nuw> : i64
    %50 = llvm.add %48, %49 overflow<nsw, nuw> : i64
    %51 = llvm.getelementptr inbounds|nuw %47[%50] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %52 = llvm.load %51 : !llvm.ptr -> f32
    %53 = llvm.mul %35, %6 overflow<nsw, nuw> : i64
    %54 = llvm.add %53, %37 overflow<nsw, nuw> : i64
    %55 = llvm.getelementptr inbounds|nuw %18[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %56 = llvm.load %55 : !llvm.ptr -> f32
    %57 = llvm.fmul %46, %52 : f32
    %58 = llvm.fadd %56, %57 : f32
    %59 = llvm.mul %35, %6 overflow<nsw, nuw> : i64
    %60 = llvm.add %59, %37 overflow<nsw, nuw> : i64
    %61 = llvm.getelementptr inbounds|nuw %18[%60] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %58, %61 : f32, !llvm.ptr
    %62 = llvm.add %39, %5 : i64
    llvm.br ^bb11(%62 : i64)
  ^bb13:  // pred: ^bb11
    %63 = llvm.add %37, %5 : i64
    llvm.br ^bb9(%63 : i64)
  ^bb14:  // pred: ^bb9
    %64 = llvm.add %35, %5 : i64
    llvm.br ^bb7(%64 : i64)
  ^bb15:  // pred: ^bb7
    llvm.return %25 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
  llvm.func @_mlir_ciface_main_graph_test_linalg(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {llvm.emit_c_interface} {
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
    %16 = llvm.call @main_graph_test_linalg(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.store %16, %arg0 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @run_main_graph_test_linalg(%arg0: !llvm.ptr) -> !llvm.ptr {
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
    llvm.call @_mlir_ciface_main_graph_test_linalg(%5, %7, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
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
    %0 = llvm.call @run_main_graph_test_linalg(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global internal constant @_entry_point_arrays_test_linalg() {addr_space = 0 : i32} : !llvm.array<3 x ptr> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @_entry_point_1_test_linalg : !llvm.ptr
    %2 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %3 = llvm.mlir.addressof @_entry_point_0_test_linalg : !llvm.ptr
    %4 = llvm.insertvalue %3, %2[0] : !llvm.array<3 x ptr> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.array<3 x ptr> 
    %6 = llvm.insertvalue %0, %5[2] : !llvm.array<3 x ptr> 
    llvm.return %6 : !llvm.array<3 x ptr>
  }
  llvm.func @omQueryEntryPoints_test_linalg(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.addressof @_entry_point_arrays_test_linalg : !llvm.ptr
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
    %0 = llvm.call @omQueryEntryPoints_test_linalg(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omInputSignature_test_linalg(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @_entry_point_1_in_sig_test_linalg : !llvm.ptr
    %2 = llvm.mlir.constant(27 : i64) : i64
    %3 = llvm.mlir.addressof @_entry_point_1_test_linalg : !llvm.ptr
    %4 = llvm.mlir.addressof @_entry_point_0_in_sig_test_linalg : !llvm.ptr
    %5 = llvm.mlir.constant(15 : i64) : i64
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.addressof @_entry_point_0_test_linalg : !llvm.ptr
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
    %0 = llvm.call @omInputSignature_test_linalg(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omOutputSignature_test_linalg(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @_entry_point_1_out_sig_test_linalg : !llvm.ptr
    %2 = llvm.mlir.constant(27 : i64) : i64
    %3 = llvm.mlir.addressof @_entry_point_1_test_linalg : !llvm.ptr
    %4 = llvm.mlir.addressof @_entry_point_0_out_sig_test_linalg : !llvm.ptr
    %5 = llvm.mlir.constant(15 : i64) : i64
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.addressof @_entry_point_0_test_linalg : !llvm.ptr
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
    %0 = llvm.call @omOutputSignature_test_linalg(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
}
