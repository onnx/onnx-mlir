// RUN: onnx-mlir-opt --maccel=NNPA --convert-krnl-to-llvm="use-opaque-pointers=true func-call-error-exit=true"  %s -split-input-file | FileCheck %s

// -----

func.func @test_stick() -> () {
  %0 = memref.alloc() : memref<10x10xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  "zlow.stick"(%0, %1) : (memref<10x10xf32>, memref<1x1x32x64xf32>) -> ()
  return

// CHECK-DAG:       [[VAR_69_1_:%.+]] = llvm.call @zdnn_transform_ztensor
// CHECK-DAG:       [[VAR_70_1_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_71_1_:%.+]] = llvm.icmp "ne" [[VAR_70_1_]], [[VAR_69_1_]] : i32
// CHECK:           llvm.cond_br [[VAR_71_1_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_72_1_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_TRANSFORM_ZTENSOR): returned %#x\0A" : !llvm.ptr<array<68 x i8>>
// CHECK:           [[VAR_73_1_:%.+]] = llvm.bitcast [[VAR_72_1_]] : !llvm.ptr<array<68 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_73_1_]], [[VAR_69_1_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_74_1_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_74_1_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return
}

// -----

func.func @test_unstick() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<10x10xf32>
  "zlow.unstick"(%0, %1) : (memref<1x1x32x64xf32>, memref<10x10xf32>) -> ()
  return

// CHECK-DAG:       [[VAR_69_2_:%.+]] = llvm.call @zdnn_transform_origtensor
// CHECK-DAG:       [[VAR_70_2_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_71_2_:%.+]] = llvm.icmp "ne" [[VAR_70_2_]], [[VAR_69_2_]] : i32
// CHECK:           llvm.cond_br [[VAR_71_2_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_72_2_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_TRANSFORM_ORIGTENSOR): returned %#x\0A" : !llvm.ptr<array<71 x i8>>
// CHECK:           [[VAR_73_2_:%.+]] = llvm.bitcast [[VAR_72_2_]] : !llvm.ptr<array<71 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_73_2_]], [[VAR_69_2_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_74_2_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_74_2_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return

}

// -----

// Check whether the lowering of zlow.relu calls the correct zDNN API or not.
func.func @test_call_zdnn_relu() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.relu"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.call @zdnn_relu
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_105_:%.+]] = llvm.icmp "ne" [[VAR_104_]], [[VAR_103_]] : i32
// CHECK:           llvm.cond_br [[VAR_105_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_106_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_RELU): returned %#x\0A" : !llvm.ptr<array<55 x i8>>
// CHECK:           [[VAR_107_:%.+]] = llvm.bitcast [[VAR_106_]] : !llvm.ptr<array<55 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_107_]], [[VAR_103_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_108_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_108_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return
}

// -----

// Check whether the lowering of zlow.tanh calls the correct zDNN API or not.
func.func @test_call_zdnn_tanh() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.tanh"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

// CHECK-DAG:       [[VAR_102_1_:%.+]] = llvm.call @zdnn_tanh
// CHECK-DAG:       [[VAR_103_1_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_104_1_:%.+]] = llvm.icmp "ne" [[VAR_103_1_]], [[VAR_102_1_]] : i32
// CHECK:           llvm.cond_br [[VAR_104_1_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_105_1_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_TANH): returned %#x\0A" : !llvm.ptr<array<55 x i8>>
// CHECK:           [[VAR_106_1_:%.+]] = llvm.bitcast [[VAR_105_1_]] : !llvm.ptr<array<55 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_106_1_]], [[VAR_102_1_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_107_1_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_107_1_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_tanh() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_tanh() : () -> ()
// CHECK:           llvm.return

}

// -----

// Check whether the lowering of zlow.sigmoid calls the correct zDNN API or not.
func.func @test_call_zdnn_sigmoid() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.sigmoid"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

// CHECK-DAG:       [[VAR_102_2_:%.+]] = llvm.call @zdnn_sigmoid
// CHECK-DAG:       [[VAR_103_2_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_104_2_:%.+]] = llvm.icmp "ne" [[VAR_103_2_]], [[VAR_102_2_]] : i32
// CHECK:           llvm.cond_br [[VAR_104_2_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_105_2_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_SIGMOID): returned %#x\0A" : !llvm.ptr<array<58 x i8>>
// CHECK:           [[VAR_106_2_:%.+]] = llvm.bitcast [[VAR_105_2_]] : !llvm.ptr<array<58 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_106_2_]], [[VAR_102_2_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_107_2_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_107_2_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return
}

// -----

// Check whether the lowering of zlow.add calls the correct zDNN API or not.
func.func @test_call_zdnn_add() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %2 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.add"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

// CHECK-DAG:       [[VAR_138_:%.+]] = llvm.call @zdnn_add
// CHECK-DAG:       [[VAR_139_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_140_:%.+]] = llvm.icmp "ne" [[VAR_139_]], [[VAR_138_]] : i32
// CHECK:           llvm.cond_br [[VAR_140_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_141_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_ADD): returned %#x\0A" : !llvm.ptr<array<54 x i8>>
// CHECK:           [[VAR_142_:%.+]] = llvm.bitcast [[VAR_141_]] : !llvm.ptr<array<54 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_142_]], [[VAR_138_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_143_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_143_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return
}

// -----

// Check whether the lowering of zlow.sub calls the correct zDNN API or not.
func.func @test_call_zdnn_sub() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %2 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.sub"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

// CHECK-DAG:       [[VAR_138_1_:%.+]] = llvm.call @zdnn_sub
// CHECK-DAG:       [[VAR_139_1_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_140_1_:%.+]] = llvm.icmp "ne" [[VAR_139_1_]], [[VAR_138_1_]] : i32
// CHECK:           llvm.cond_br [[VAR_140_1_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_141_1_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_SUB): returned %#x\0A" : !llvm.ptr<array<54 x i8>>
// CHECK:           [[VAR_142_1_:%.+]] = llvm.bitcast [[VAR_141_1_]] : !llvm.ptr<array<54 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_142_1_]], [[VAR_138_1_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_143_1_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_143_1_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return
}

// -----

// Check whether the lowering of zlow.softmax calls the correct zDNN API or not.
func.func @test_call_zdnn_softmax() -> () {
  %0 = memref.alloc() : memref<1x1x1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x1x1x32x64xf32>
  %work_area = memref.alloc() {alignment = 4096 : i64} : memref<8192xi8>
  %shape = memref.alloc() : memref<3xi64>
  "zlow.softmax"(%0, %work_area, %shape, %1) {act_func = "ACT_NONE"} : (memref<1x1x1x1x32x64xf32>, memref<8192xi8>, memref<3xi64>, memref<1x1x1x1x32x64xf32>) -> ()
  return

// CHECK-DAG:       [[VAR_145_:%.+]] = llvm.call @zdnn_softmax
// CHECK-DAG:       [[VAR_146_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_147_:%.+]] = llvm.icmp "ne" [[VAR_146_]], [[VAR_145_]] : i32
// CHECK:           llvm.cond_br [[VAR_147_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_148_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_SOFTMAX): returned %#x\0A" : !llvm.ptr<array<58 x i8>>
// CHECK:           [[VAR_149_:%.+]] = llvm.bitcast [[VAR_148_]] : !llvm.ptr<array<58 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_149_]], [[VAR_145_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_150_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_150_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return

}

// -----

// COM: Check whether the lowering of zlow.stickForLSTM calls the correct zDNN API or not.
// COM: We should call zdnn_transform_ztensor with zTensor and four pointers to the buffers fori F, I, C, and O gates. 
func.func @test_stick_for_lstm() -> () {
  %f = memref.alloc() : memref<1x10x10xf32>
  %i = memref.alloc() : memref<1x10x10xf32>
  %c = memref.alloc() : memref<1x10x10xf32>
  %o = memref.alloc() : memref<1x10x10xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  "zlow.stickForLSTM"(%f, %i, %c, %o, %1) : (memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x1x32x64xf32>) -> ()
  return

// CHECK-DAG:       [[VAR_144_1_:%.+]] = llvm.call @zdnn_transform_ztensor
// CHECK-DAG:       [[VAR_145_1_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_146_1_:%.+]] = llvm.icmp "ne" [[VAR_145_1_]], [[VAR_144_1_]] : i32
// CHECK:           llvm.cond_br [[VAR_146_1_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_147_1_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_TRANSFORM_ZTENSOR): returned %#x\0A" : !llvm.ptr<array<68 x i8>>
// CHECK:           [[VAR_148_1_:%.+]] = llvm.bitcast [[VAR_147_1_]] : !llvm.ptr<array<68 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_148_1_]], [[VAR_144_1_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_149_1_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_149_1_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return

}

// -----

// COM: Check whether the lowering of zlow.stickForGRU calls the correct zDNN API or not.
// COM: We should call zdnn_transform_ztensor with zTensor and three pointers to the buffers for Z, R, and H gates. 
func.func @test_stick_for_gru() -> () {
  %g = memref.alloc() : memref<1x10x10xf32>
  %r = memref.alloc() : memref<1x10x10xf32>
  %h = memref.alloc() : memref<1x10x10xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  "zlow.stickForGRU"(%g, %r, %h, %1) : (memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x1x32x64xf32>) -> ()
  return

// CHECK-DAG:       [[VAR_121_6_:%.+]] = llvm.call @zdnn_transform_ztensor
// CHECK-DAG:       [[VAR_122_6_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_123_6_:%.+]] = llvm.icmp "ne" [[VAR_122_6_]], [[VAR_121_6_]] : i32
// CHECK:           llvm.cond_br [[VAR_123_6_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_124_6_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_TRANSFORM_ZTENSOR): returned %#x\0A" : !llvm.ptr<array<68 x i8>>
// CHECK:           [[VAR_125_6_:%.+]] = llvm.bitcast [[VAR_124_6_]] : !llvm.ptr<array<68 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_125_6_]], [[VAR_121_6_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_126_6_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_126_6_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return
}

// -----

// Check whether the lowering of zlow.min calls the correct zDNN API or not.
func.func @test_call_zdnn_min() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %2 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.min"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

// CHECK-DAG:       [[VAR_138_6_:%.+]] = llvm.call @zdnn_min
// CHECK-DAG:       [[VAR_139_6_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_140_6_:%.+]] = llvm.icmp "ne" [[VAR_139_6_]], [[VAR_138_6_]] : i32
// CHECK:           llvm.cond_br [[VAR_140_6_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_141_6_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_MIN): returned %#x\0A" : !llvm.ptr<array<54 x i8>>
// CHECK:           [[VAR_142_6_:%.+]] = llvm.bitcast [[VAR_141_6_]] : !llvm.ptr<array<54 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_142_6_]], [[VAR_138_6_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_143_6_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_143_6_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return
}

// -----

// Check whether the lowering of zlow.max calls the correct zDNN API or not.
func.func @test_call_zdnn_max() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %2 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.max"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

// CHECK-DAG:       [[VAR_138_7_:%.+]] = llvm.call @zdnn_max
// CHECK-DAG:       [[VAR_139_7_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_140_7_:%.+]] = llvm.icmp "ne" [[VAR_139_7_]], [[VAR_138_7_]] : i32
// CHECK:           llvm.cond_br [[VAR_140_7_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_141_7_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_MAX): returned %#x\0A" : !llvm.ptr<array<54 x i8>>
// CHECK:           [[VAR_142_7_:%.+]] = llvm.bitcast [[VAR_141_7_]] : !llvm.ptr<array<54 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_142_7_]], [[VAR_138_7_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_143_7_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_143_7_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return
}

// -----

// Check whether the lowering of zlow.exp calls the correct zDNN API or not.
func.func @test_call_zdnn_exp() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.exp"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

// CHECK-DAG:       [[VAR_102_12_:%.+]] = llvm.call @zdnn_exp
// CHECK-DAG:       [[VAR_103_11_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_104_12_:%.+]] = llvm.icmp "ne" [[VAR_103_11_]], [[VAR_102_12_]] : i32
// CHECK:           llvm.cond_br [[VAR_104_12_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_105_11_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_EXP): returned %#x\0A" : !llvm.ptr<array<54 x i8>>
// CHECK:           [[VAR_106_12_:%.+]] = llvm.bitcast [[VAR_105_11_]] : !llvm.ptr<array<54 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_106_12_]], [[VAR_102_12_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_107_11_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_107_11_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return
}

// -----

// Check whether the lowering of zlow.log calls the correct zDNN API or not.
func.func @test_call_zdnn_log() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.log"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

// CHECK-DAG:       [[VAR_102_13_:%.+]] = llvm.call @zdnn_log
// CHECK-DAG:       [[VAR_103_12_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_104_13_:%.+]] = llvm.icmp "ne" [[VAR_103_12_]], [[VAR_102_13_]] : i32
// CHECK:           llvm.cond_br [[VAR_104_13_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_105_12_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_LOG): returned %#x\0A" : !llvm.ptr<array<54 x i8>>
// CHECK:           [[VAR_106_13_:%.+]] = llvm.bitcast [[VAR_105_12_]] : !llvm.ptr<array<54 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_106_13_]], [[VAR_102_13_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_107_12_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_107_12_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return

}

// -----

// Check whether the lowering of zlow.matmul calls the correct zDNN API or not.
func.func @test_matmul_no_bcast_unstacked(%x: memref<2048xf32>,%y: memref<2048xf32>,%bias: memref<2048xf32>, %shape: memref<3xi64>) -> memref<2048xf32> {
  %res = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  "zlow.matmul"(%x, %y, %bias, %shape, %res) {is_bcast = 0 : si64, is_stacked = 0 : si64} : (memref<2048xf32>, memref<2048xf32>, memref<2048xf32>, memref<3xi64>, memref<2048xf32>) -> ()
  return %res : memref<2048xf32>

// CHECK-DAG:       [[VAR_146_2_:%.+]] = llvm.call @zdnn_matmul_op
// CHECK-DAG:       [[VAR_147_2_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_148_2_:%.+]] = llvm.icmp "ne" [[VAR_147_2_]], [[VAR_146_2_]] : i32
// CHECK:           llvm.cond_br [[VAR_148_2_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_149_2_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_MATMUL_OP): returned %#x\0A" : !llvm.ptr<array<60 x i8>>
// CHECK:           [[VAR_150_1_:%.+]] = llvm.bitcast [[VAR_149_2_]] : !llvm.ptr<array<60 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_150_1_]], [[VAR_146_2_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_151_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_151_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return
}

// -----

// Check whether conv2d calls the correct zDNN API or not.
func.func @test_call_zdnn_cond2d() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %kernel = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %bias = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %shape = memref.alloc() : memref<7xi64>
  "zlow.conv2d"(%input, %kernel, %bias, %shape, %output) {kernel_shape = [5, 5], strides = [2, 2], padding_type = "SAME_PADDING", act_func = "ACT_NONE" } : (memref<2048xf32>, memref<2048xf32>, memref<2048xf32>, memref<7xi64>, memref<2048xf32>)-> ()
  return

// CHECK-DAG:       [[VAR_215_:%.+]] = llvm.call @zdnn_conv2d
// CHECK-DAG:       [[VAR_216_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_217_:%.+]] = llvm.icmp "ne" [[VAR_216_]], [[VAR_215_]] : i32
// CHECK:           llvm.cond_br [[VAR_217_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_218_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_CONV2D): returned %#x\0A" : !llvm.ptr<array<57 x i8>>
// CHECK:           [[VAR_219_:%.+]] = llvm.bitcast [[VAR_218_]] : !llvm.ptr<array<57 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_219_]], [[VAR_215_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_220_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_220_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return
}

// -----

// Check whether avgpool2d calls the correct zDNN API or not.
func.func @test_call_zdnn_avgpool2d() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %shape = memref.alloc() : memref<6xi64>
  "zlow.avgpool2d"(%input, %shape, %output) {kernel_shape = [5, 5], strides = [2, 2], padding_type = "SAME_PADDING" } : (memref<2048xf32>, memref<6xi64>, memref<2048xf32>)-> ()
  return

// CHECK-DAG:       [[VAR_121_16_:%.+]] = llvm.call @zdnn_avgpool2d
// CHECK-DAG:       [[VAR_122_16_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_123_16_:%.+]] = llvm.icmp "ne" [[VAR_122_16_]], [[VAR_121_16_]] : i32
// CHECK:           llvm.cond_br [[VAR_123_16_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_124_16_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_AVGPOOL2D): returned %#x\0A" : !llvm.ptr<array<60 x i8>>
// CHECK:           [[VAR_125_16_:%.+]] = llvm.bitcast [[VAR_124_16_]] : !llvm.ptr<array<60 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_125_16_]], [[VAR_121_16_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_126_16_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_126_16_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return

}

// -----

// Check whether maxpool2d calls the correct zDNN API or not.
func.func @test_call_zdnn_maxpool2d() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %shape = memref.alloc() : memref<6xi64>
  "zlow.maxpool2d"(%input, %shape, %output) {kernel_shape = [5, 5], strides = [2, 2], padding_type = "SAME_PADDING" } : (memref<2048xf32>, memref<6xi64>, memref<2048xf32>)-> ()
  return

// CHECK-DAG:       [[VAR_121_17_:%.+]] = llvm.call @zdnn_maxpool2d
// CHECK-DAG:       [[VAR_122_17_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_123_17_:%.+]] = llvm.icmp "ne" [[VAR_122_17_]], [[VAR_121_17_]] : i32
// CHECK:           llvm.cond_br [[VAR_123_17_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_124_17_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_MAXPOOL2D): returned %#x\0A" : !llvm.ptr<array<60 x i8>>
// CHECK:           [[VAR_125_17_:%.+]] = llvm.bitcast [[VAR_124_17_]] : !llvm.ptr<array<60 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_125_17_]], [[VAR_121_17_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_126_17_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_126_17_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return

}


// -----

// Check whether meanreduce2d calls the correct zDNN API or not.
func.func @test_call_zdnn_meanreduce2d() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32>
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32>
  %shape = memref.alloc() : memref<4xi64>
  "zlow.meanreduce2d"(%input, %shape, %output) : (memref<2048xf32>, memref<4xi64>, memref<2048xf32>)-> ()
  return

// CHECK-DAG:       [[VAR_113_15_:%.+]] = llvm.call @zdnn_meanreduce2d
// CHECK-DAG:       [[VAR_114_18_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_115_15_:%.+]] = llvm.icmp "ne" [[VAR_114_18_]], [[VAR_113_15_]] : i32
// CHECK:           llvm.cond_br [[VAR_115_15_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_116_18_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_MEANREDUCE2D): returned %#x\0A" : !llvm.ptr<array<63 x i8>>
// CHECK:           [[VAR_117_18_:%.+]] = llvm.bitcast [[VAR_116_18_]] : !llvm.ptr<array<63 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_117_18_]], [[VAR_113_15_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_118_18_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_118_18_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return

}

// -----

// Check whether batchnorm calls the correct zDNN API or not.
func.func @test_call_zdnn_batchnorm() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32>
  %a = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32>
  %b = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32>
  %shape = memref.alloc() : memref<4xi64>
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32>
  "zlow.batchnorm"(%input, %a, %b, %shape, %output) : (memref<2048xf32>, memref<2048xf32>, memref<2048xf32>, memref<4xi64>, memref<2048xf32>)-> ()
  return

// CHECK-DAG:       [[VAR_202_3_:%.+]] = llvm.call @zdnn_batchnorm
// CHECK-DAG:       [[VAR_203_3_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           [[VAR_204_3_:%.+]] = llvm.icmp "ne" [[VAR_203_3_]], [[VAR_202_3_]] : i32
// CHECK:           llvm.cond_br [[VAR_204_3_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_205_3_:%.+]] = llvm.mlir.addressof @"om_onnx-mlir: Error in zDNN call(ZDNN_BATCHNORM): returned %#x\0A" : !llvm.ptr<array<60 x i8>>
// CHECK:           [[VAR_206_3_:%.+]] = llvm.bitcast [[VAR_205_3_]] : !llvm.ptr<array<60 x i8>> to !llvm.ptr
// CHECK:           llvm.call @printf([[VAR_206_3_]], [[VAR_202_3_]]) : (!llvm.ptr, i32) -> ()
// CHECK:           [[VAR_207_3_:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           llvm.call @exit([[VAR_207_3_]]) : (i32) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return

}
