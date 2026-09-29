// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// -----

func.func @test_lower_both_zlow_and_krnl() -> () {
  %0 = memref.alloc() : memref<10x10xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %2 = "krnl.global"() {name = "constant_0", shape = [1, 2], value = dense<[[0., 1.0]]> : tensor<1x2xf32>} : () -> memref<1x2xf32>
  "zlow.stick"(%0, %1) {no_saturation = -1 : si64} : (memref<10x10xf32>, memref<1x1x32x64xf16>) -> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnn_transform_ztensor(!llvm.ptr, ...) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.mlir.global internal constant @constant_0(dense<{{.}}[0.000000e+00, 1.000000e+00]{{.}}> : tensor<1x2xf32>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<1 x array<2 x f32>>
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_lower_both_zlow_and_krnl() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = llvm.mlir.constant(100 : index) : i64
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_5_:%.+]] = llvm.getelementptr [[VAR_4_]]{{.}}[[VAR_3_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           [[VAR_6_:%.+]] = llvm.ptrtoint [[VAR_5_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.call @malloc([[VAR_6_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           [[VAR_9_:%.+]] = llvm.insertvalue [[VAR_7_]], [[VAR_8_]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_10_:%.+]] = llvm.insertvalue [[VAR_7_]], [[VAR_9_]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_12_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_10_]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           [[VAR_13_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_12_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           [[VAR_14_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_13_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           [[VAR_15_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_14_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_15_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_17_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_18_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_19_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_20_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_21_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_22_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_23_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_26_:%.+]] = llvm.getelementptr [[VAR_25_]]{{.}}[[VAR_24_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_27_:%.+]] = llvm.ptrtoint [[VAR_26_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.call @malloc([[VAR_27_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_29_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_30_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_29_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_30_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_33_:%.+]] = llvm.insertvalue [[VAR_32_]], [[VAR_31_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_34_:%.+]] = llvm.insertvalue [[VAR_17_]], [[VAR_33_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_35_:%.+]] = llvm.insertvalue [[VAR_18_]], [[VAR_34_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_36_:%.+]] = llvm.insertvalue [[VAR_19_]], [[VAR_35_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_37_:%.+]] = llvm.insertvalue [[VAR_20_]], [[VAR_36_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_23_]], [[VAR_37_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_22_]], [[VAR_38_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_40_:%.+]] = llvm.insertvalue [[VAR_20_]], [[VAR_39_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_21_]], [[VAR_40_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_42_:%.+]] = llvm.mlir.addressof @constant_0 : !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_43_:%.+]] = llvm.bitcast [[VAR_42_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_44_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           [[VAR_45_:%.+]] = llvm.insertvalue [[VAR_43_]], [[VAR_44_]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_46_:%.+]] = llvm.insertvalue [[VAR_43_]], [[VAR_45_]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_47_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_48_:%.+]] = llvm.insertvalue [[VAR_47_]], [[VAR_46_]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_50_:%.+]] = llvm.insertvalue [[VAR_49_]], [[VAR_48_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_50_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_54_:%.+]] = llvm.insertvalue [[VAR_53_]], [[VAR_52_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_54_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_57_:%.+]] = llvm.extractvalue [[VAR_16_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.extractvalue [[VAR_16_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_59_:%.+]] = llvm.extractvalue [[VAR_41_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_60_:%.+]] = llvm.bitcast [[VAR_59_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_61_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_62_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_63_:%.+]] = llvm.alloca [[VAR_62_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_64_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.mlir.constant(255 : i64) : i64
// CHECK:           [[VAR_66_:%.+]] = llvm.bitcast [[VAR_63_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_64_]], [[VAR_65_]], [[VAR_66_]], [[VAR_57_]], [[VAR_58_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_67_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.alloca [[VAR_67_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_69_:%.+]] = llvm.bitcast [[VAR_63_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_70_:%.+]] = llvm.bitcast [[VAR_68_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_69_]], [[VAR_70_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_72_:%.+]] = llvm.alloca [[VAR_61_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_73_:%.+]] = llvm.bitcast [[VAR_68_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_73_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.getelementptr [[VAR_72_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_63_]], [[VAR_75_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_76_:%.+]] = llvm.getelementptr [[VAR_72_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_68_]], [[VAR_76_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_77_:%.+]] = llvm.getelementptr [[VAR_72_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_74_]], [[VAR_77_]] : i64, !llvm.ptr
// CHECK:           [[VAR_78_:%.+]] = llvm.getelementptr [[VAR_72_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_60_]], [[VAR_78_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_79_:%.+]] = llvm.mlir.constant(false) : i1
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.getelementptr [[VAR_72_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_79_]], [[VAR_80_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.getelementptr [[VAR_72_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_82_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_82_]], [[VAR_81_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.getelementptr [[VAR_72_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_84_]], [[VAR_83_]] : f32, !llvm.ptr
// CHECK:           [[VAR_85_:%.+]] = llvm.extractvalue [[VAR_16_]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_86_:%.+]] = llvm.bitcast [[VAR_85_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_87_:%.+]] = llvm.bitcast [[VAR_72_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_88_:%.+]] = llvm.call @zdnn_transform_ztensor([[VAR_87_]], [[VAR_86_]]) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_lower_both_zlow_and_krnl() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_lower_both_zlow_and_krnl() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

func.func @test_stick() -> () {
  %0 = memref.alloc() : memref<10x10xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  "zlow.stick"(%0, %1) {no_saturation = -1 : si64} : (memref<10x10xf32>, memref<1x1x32x64xf16>) -> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnn_transform_ztensor(!llvm.ptr, ...) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_stick() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = llvm.mlir.constant(100 : index) : i64
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_5_:%.+]] = llvm.getelementptr [[VAR_4_]]{{.}}[[VAR_3_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           [[VAR_6_:%.+]] = llvm.ptrtoint [[VAR_5_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.call @malloc([[VAR_6_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           [[VAR_9_:%.+]] = llvm.insertvalue [[VAR_7_]], [[VAR_8_]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_10_:%.+]] = llvm.insertvalue [[VAR_7_]], [[VAR_9_]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_12_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_10_]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           [[VAR_13_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_12_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           [[VAR_14_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_13_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           [[VAR_15_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_14_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_15_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_17_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_18_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_19_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_20_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_21_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_22_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_23_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_26_:%.+]] = llvm.getelementptr [[VAR_25_]]{{.}}[[VAR_24_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_27_:%.+]] = llvm.ptrtoint [[VAR_26_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.call @malloc([[VAR_27_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_29_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_30_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_29_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_30_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_33_:%.+]] = llvm.insertvalue [[VAR_32_]], [[VAR_31_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_34_:%.+]] = llvm.insertvalue [[VAR_17_]], [[VAR_33_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_35_:%.+]] = llvm.insertvalue [[VAR_18_]], [[VAR_34_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_36_:%.+]] = llvm.insertvalue [[VAR_19_]], [[VAR_35_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_37_:%.+]] = llvm.insertvalue [[VAR_20_]], [[VAR_36_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_23_]], [[VAR_37_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_22_]], [[VAR_38_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_40_:%.+]] = llvm.insertvalue [[VAR_20_]], [[VAR_39_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_21_]], [[VAR_40_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_42_:%.+]] = llvm.extractvalue [[VAR_16_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_43_:%.+]] = llvm.extractvalue [[VAR_16_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           [[VAR_44_:%.+]] = llvm.extractvalue [[VAR_41_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_45_:%.+]] = llvm.bitcast [[VAR_44_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_46_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_47_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_48_:%.+]] = llvm.alloca [[VAR_47_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_50_:%.+]] = llvm.mlir.constant(255 : i64) : i64
// CHECK:           [[VAR_51_:%.+]] = llvm.bitcast [[VAR_48_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_49_]], [[VAR_50_]], [[VAR_51_]], [[VAR_42_]], [[VAR_43_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_52_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.alloca [[VAR_52_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_54_:%.+]] = llvm.bitcast [[VAR_48_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_55_:%.+]] = llvm.bitcast [[VAR_53_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_54_]], [[VAR_55_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_57_:%.+]] = llvm.alloca [[VAR_46_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.bitcast [[VAR_53_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_59_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_58_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_60_:%.+]] = llvm.getelementptr [[VAR_57_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_48_]], [[VAR_60_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_61_:%.+]] = llvm.getelementptr [[VAR_57_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_53_]], [[VAR_61_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_62_:%.+]] = llvm.getelementptr [[VAR_57_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_59_]], [[VAR_62_]] : i64, !llvm.ptr
// CHECK:           [[VAR_63_:%.+]] = llvm.getelementptr [[VAR_57_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_45_]], [[VAR_63_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_64_:%.+]] = llvm.mlir.constant(false) : i1
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.getelementptr [[VAR_57_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_64_]], [[VAR_65_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.getelementptr [[VAR_57_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_67_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_67_]], [[VAR_66_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.getelementptr [[VAR_57_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_69_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_69_]], [[VAR_68_]] : f32, !llvm.ptr
// CHECK:           [[VAR_70_:%.+]] = llvm.extractvalue [[VAR_16_]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.bitcast [[VAR_70_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_72_:%.+]] = llvm.bitcast [[VAR_57_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_73_:%.+]] = llvm.call @zdnn_transform_ztensor([[VAR_72_]], [[VAR_71_]]) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_stick() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_stick() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

func.func @test_unstick() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<10x10xf32>
  "zlow.unstick"(%0, %1) : (memref<1x1x32x64xf16>, memref<10x10xf32>) -> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnn_transform_origtensor(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_unstick() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_9_:%.+]] = llvm.getelementptr [[VAR_8_]]{{.}}[[VAR_7_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_10_:%.+]] = llvm.ptrtoint [[VAR_9_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.call @malloc([[VAR_10_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_13_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_12_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_13_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_15_]], [[VAR_14_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_16_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_18_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_17_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_18_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_19_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_6_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_22_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_21_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_23_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_22_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.insertvalue [[VAR_4_]], [[VAR_23_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.mlir.constant(100 : index) : i64
// CHECK-DAG:       [[VAR_29_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_30_:%.+]] = llvm.getelementptr [[VAR_29_]]{{.}}[[VAR_28_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           [[VAR_31_:%.+]] = llvm.ptrtoint [[VAR_30_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.call @malloc([[VAR_31_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           [[VAR_34_:%.+]] = llvm.insertvalue [[VAR_32_]], [[VAR_33_]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_35_:%.+]] = llvm.insertvalue [[VAR_32_]], [[VAR_34_]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_37_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_35_]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_37_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_26_]], [[VAR_38_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           [[VAR_40_:%.+]] = llvm.insertvalue [[VAR_26_]], [[VAR_39_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_27_]], [[VAR_40_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_42_:%.+]] = llvm.extractvalue [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_43_:%.+]] = llvm.extractvalue [[VAR_41_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_44_:%.+]] = llvm.extractvalue [[VAR_24_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_45_:%.+]] = llvm.bitcast [[VAR_44_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_46_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_47_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_48_:%.+]] = llvm.alloca [[VAR_47_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_50_:%.+]] = llvm.mlir.constant(255 : i64) : i64
// CHECK:           [[VAR_51_:%.+]] = llvm.bitcast [[VAR_48_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_49_]], [[VAR_50_]], [[VAR_51_]], [[VAR_42_]], [[VAR_43_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_52_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.alloca [[VAR_52_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_54_:%.+]] = llvm.bitcast [[VAR_48_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_55_:%.+]] = llvm.bitcast [[VAR_53_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_54_]], [[VAR_55_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_57_:%.+]] = llvm.alloca [[VAR_46_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.bitcast [[VAR_53_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_59_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_58_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_60_:%.+]] = llvm.getelementptr [[VAR_57_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_48_]], [[VAR_60_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_61_:%.+]] = llvm.getelementptr [[VAR_57_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_53_]], [[VAR_61_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_62_:%.+]] = llvm.getelementptr [[VAR_57_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_59_]], [[VAR_62_]] : i64, !llvm.ptr
// CHECK:           [[VAR_63_:%.+]] = llvm.getelementptr [[VAR_57_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_45_]], [[VAR_63_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_64_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.getelementptr [[VAR_57_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_64_]], [[VAR_65_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.getelementptr [[VAR_57_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_67_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_67_]], [[VAR_66_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.getelementptr [[VAR_57_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_69_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_69_]], [[VAR_68_]] : f32, !llvm.ptr
// CHECK:           [[VAR_70_:%.+]] = llvm.extractvalue [[VAR_41_]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.bitcast [[VAR_70_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_72_:%.+]] = llvm.bitcast [[VAR_57_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_73_:%.+]] = llvm.call @zdnn_transform_origtensor([[VAR_72_]], [[VAR_71_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_unstick() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_unstick() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether the lowering of zlow.relu calls the correct zDNN API or not.
func.func @test_call_zdnn_relu() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.relu"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnnx_relu(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_relu() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_9_:%.+]] = llvm.getelementptr [[VAR_8_]]{{.}}[[VAR_7_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_10_:%.+]] = llvm.ptrtoint [[VAR_9_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.call @malloc([[VAR_10_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_13_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_12_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_13_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_15_]], [[VAR_14_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_16_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_18_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_17_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_18_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_19_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_6_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_22_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_21_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_23_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_22_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.insertvalue [[VAR_4_]], [[VAR_23_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_29_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_34_:%.+]] = llvm.getelementptr [[VAR_33_]]{{.}}[[VAR_32_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_35_:%.+]] = llvm.ptrtoint [[VAR_34_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.call @malloc([[VAR_35_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_37_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_38_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_39_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_42_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_26_]], [[VAR_42_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_44_:%.+]] = llvm.insertvalue [[VAR_27_]], [[VAR_43_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_45_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_44_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_46_:%.+]] = llvm.insertvalue [[VAR_31_]], [[VAR_45_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_47_:%.+]] = llvm.insertvalue [[VAR_30_]], [[VAR_46_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_48_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_47_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.insertvalue [[VAR_29_]], [[VAR_48_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_50_:%.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_53_:%.+]] = llvm.getelementptr [[VAR_52_]]{{.}}[[VAR_50_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_54_:%.+]] = llvm.ptrtoint [[VAR_53_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.call @malloc([[VAR_54_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_57_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_56_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_57_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_59_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_60_:%.+]] = llvm.insertvalue [[VAR_59_]], [[VAR_58_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_61_:%.+]] = llvm.insertvalue [[VAR_50_]], [[VAR_60_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_62_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_61_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_63_:%.+]] = llvm.extractvalue [[VAR_62_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_64_:%.+]] = llvm.getelementptr [[VAR_63_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_64_MEM_:%.+]] = llvm.load [[VAR_64_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.getelementptr [[VAR_63_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_66_MEM_:%.+]] = llvm.load [[VAR_66_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.extractvalue [[VAR_24_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_69_:%.+]] = llvm.bitcast [[VAR_68_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_70_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_72_:%.+]] = llvm.alloca [[VAR_71_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_73_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_75_:%.+]] = llvm.bitcast [[VAR_72_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_73_]], [[VAR_74_]], [[VAR_75_]], [[LOAD_VAR_64_MEM_]], [[LOAD_VAR_66_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_76_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.alloca [[VAR_76_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_78_:%.+]] = llvm.bitcast [[VAR_72_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_79_:%.+]] = llvm.bitcast [[VAR_77_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_78_]], [[VAR_79_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.alloca [[VAR_70_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_82_:%.+]] = llvm.bitcast [[VAR_77_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_82_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_72_]], [[VAR_84_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_85_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_77_]], [[VAR_85_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_86_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_83_]], [[VAR_86_]] : i64, !llvm.ptr
// CHECK:           [[VAR_87_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_69_]], [[VAR_87_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_88_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_89_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_88_]], [[VAR_89_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_90_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_91_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_91_]], [[VAR_90_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_92_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_93_]], [[VAR_92_]] : f32, !llvm.ptr
// CHECK:           [[VAR_94_:%.+]] = llvm.extractvalue [[VAR_49_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.bitcast [[VAR_94_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_97_:%.+]] = llvm.alloca [[VAR_96_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_98_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_72_]], [[VAR_98_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_99_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_77_]], [[VAR_99_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_100_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_83_]], [[VAR_100_]] : i64, !llvm.ptr
// CHECK:           [[VAR_101_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_95_]], [[VAR_101_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_103_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_105_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_105_]], [[VAR_104_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_107_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_107_]], [[VAR_106_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.bitcast [[VAR_81_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_110_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_111_:%.+]] = llvm.call @zdnnx_relu([[VAR_109_]], [[VAR_108_]], [[VAR_110_]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_relu() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_relu() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether the lowering of zlow.tanh calls the correct zDNN API or not.
func.func @test_call_zdnn_tanh() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.tanh"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnnx_tanh(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_tanh() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_9_:%.+]] = llvm.getelementptr [[VAR_8_]]{{.}}[[VAR_7_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_10_:%.+]] = llvm.ptrtoint [[VAR_9_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.call @malloc([[VAR_10_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_13_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_12_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_13_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_15_]], [[VAR_14_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_16_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_18_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_17_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_18_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_19_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_6_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_22_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_21_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_23_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_22_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.insertvalue [[VAR_4_]], [[VAR_23_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_29_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_34_:%.+]] = llvm.getelementptr [[VAR_33_]]{{.}}[[VAR_32_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_35_:%.+]] = llvm.ptrtoint [[VAR_34_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.call @malloc([[VAR_35_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_37_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_38_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_39_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_42_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_26_]], [[VAR_42_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_44_:%.+]] = llvm.insertvalue [[VAR_27_]], [[VAR_43_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_45_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_44_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_46_:%.+]] = llvm.insertvalue [[VAR_31_]], [[VAR_45_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_47_:%.+]] = llvm.insertvalue [[VAR_30_]], [[VAR_46_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_48_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_47_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.insertvalue [[VAR_29_]], [[VAR_48_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_50_:%.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_53_:%.+]] = llvm.getelementptr [[VAR_52_]]{{.}}[[VAR_50_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_54_:%.+]] = llvm.ptrtoint [[VAR_53_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.call @malloc([[VAR_54_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_57_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_56_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_57_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_59_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_60_:%.+]] = llvm.insertvalue [[VAR_59_]], [[VAR_58_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_61_:%.+]] = llvm.insertvalue [[VAR_50_]], [[VAR_60_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_62_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_61_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_63_:%.+]] = llvm.extractvalue [[VAR_62_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_64_:%.+]] = llvm.getelementptr [[VAR_63_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_64_MEM_:%.+]] = llvm.load [[VAR_64_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.getelementptr [[VAR_63_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_66_MEM_:%.+]] = llvm.load [[VAR_66_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.extractvalue [[VAR_24_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_69_:%.+]] = llvm.bitcast [[VAR_68_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_70_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_72_:%.+]] = llvm.alloca [[VAR_71_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_73_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_75_:%.+]] = llvm.bitcast [[VAR_72_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_73_]], [[VAR_74_]], [[VAR_75_]], [[LOAD_VAR_64_MEM_]], [[LOAD_VAR_66_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_76_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.alloca [[VAR_76_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_78_:%.+]] = llvm.bitcast [[VAR_72_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_79_:%.+]] = llvm.bitcast [[VAR_77_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_78_]], [[VAR_79_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.alloca [[VAR_70_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_82_:%.+]] = llvm.bitcast [[VAR_77_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_82_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_72_]], [[VAR_84_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_85_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_77_]], [[VAR_85_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_86_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_83_]], [[VAR_86_]] : i64, !llvm.ptr
// CHECK:           [[VAR_87_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_69_]], [[VAR_87_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_88_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_89_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_88_]], [[VAR_89_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_90_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_91_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_91_]], [[VAR_90_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_92_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_93_]], [[VAR_92_]] : f32, !llvm.ptr
// CHECK:           [[VAR_94_:%.+]] = llvm.extractvalue [[VAR_49_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.bitcast [[VAR_94_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_97_:%.+]] = llvm.alloca [[VAR_96_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_98_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_72_]], [[VAR_98_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_99_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_77_]], [[VAR_99_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_100_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_83_]], [[VAR_100_]] : i64, !llvm.ptr
// CHECK:           [[VAR_101_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_95_]], [[VAR_101_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_103_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_105_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_105_]], [[VAR_104_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_107_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_107_]], [[VAR_106_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.bitcast [[VAR_81_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_110_:%.+]] = llvm.call @zdnnx_tanh([[VAR_108_]], [[VAR_109_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_tanh() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_tanh() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether the lowering of zlow.sigmoid calls the correct zDNN API or not.
func.func @test_call_zdnn_sigmoid() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.sigmoid"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnnx_sigmoid(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_sigmoid() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_9_:%.+]] = llvm.getelementptr [[VAR_8_]]{{.}}[[VAR_7_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_10_:%.+]] = llvm.ptrtoint [[VAR_9_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.call @malloc([[VAR_10_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_13_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_12_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_13_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_15_]], [[VAR_14_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_16_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_18_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_17_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_18_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_19_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_6_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_22_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_21_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_23_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_22_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.insertvalue [[VAR_4_]], [[VAR_23_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_29_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_34_:%.+]] = llvm.getelementptr [[VAR_33_]]{{.}}[[VAR_32_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_35_:%.+]] = llvm.ptrtoint [[VAR_34_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.call @malloc([[VAR_35_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_37_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_38_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_39_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_42_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_26_]], [[VAR_42_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_44_:%.+]] = llvm.insertvalue [[VAR_27_]], [[VAR_43_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_45_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_44_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_46_:%.+]] = llvm.insertvalue [[VAR_31_]], [[VAR_45_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_47_:%.+]] = llvm.insertvalue [[VAR_30_]], [[VAR_46_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_48_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_47_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.insertvalue [[VAR_29_]], [[VAR_48_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_50_:%.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_53_:%.+]] = llvm.getelementptr [[VAR_52_]]{{.}}[[VAR_50_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_54_:%.+]] = llvm.ptrtoint [[VAR_53_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.call @malloc([[VAR_54_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_57_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_56_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_57_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_59_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_60_:%.+]] = llvm.insertvalue [[VAR_59_]], [[VAR_58_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_61_:%.+]] = llvm.insertvalue [[VAR_50_]], [[VAR_60_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_62_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_61_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_63_:%.+]] = llvm.extractvalue [[VAR_62_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_64_:%.+]] = llvm.getelementptr [[VAR_63_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_64_MEM_:%.+]] = llvm.load [[VAR_64_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.getelementptr [[VAR_63_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_66_MEM_:%.+]] = llvm.load [[VAR_66_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.extractvalue [[VAR_24_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_69_:%.+]] = llvm.bitcast [[VAR_68_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_70_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_72_:%.+]] = llvm.alloca [[VAR_71_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_73_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_75_:%.+]] = llvm.bitcast [[VAR_72_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_73_]], [[VAR_74_]], [[VAR_75_]], [[LOAD_VAR_64_MEM_]], [[LOAD_VAR_66_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_76_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.alloca [[VAR_76_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_78_:%.+]] = llvm.bitcast [[VAR_72_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_79_:%.+]] = llvm.bitcast [[VAR_77_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_78_]], [[VAR_79_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.alloca [[VAR_70_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_82_:%.+]] = llvm.bitcast [[VAR_77_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_82_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_72_]], [[VAR_84_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_85_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_77_]], [[VAR_85_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_86_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_83_]], [[VAR_86_]] : i64, !llvm.ptr
// CHECK:           [[VAR_87_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_69_]], [[VAR_87_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_88_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_89_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_88_]], [[VAR_89_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_90_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_91_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_91_]], [[VAR_90_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_92_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_93_]], [[VAR_92_]] : f32, !llvm.ptr
// CHECK:           [[VAR_94_:%.+]] = llvm.extractvalue [[VAR_49_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.bitcast [[VAR_94_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_97_:%.+]] = llvm.alloca [[VAR_96_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_98_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_72_]], [[VAR_98_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_99_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_77_]], [[VAR_99_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_100_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_83_]], [[VAR_100_]] : i64, !llvm.ptr
// CHECK:           [[VAR_101_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_95_]], [[VAR_101_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_103_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_105_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_105_]], [[VAR_104_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_107_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_107_]], [[VAR_106_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.bitcast [[VAR_81_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_110_:%.+]] = llvm.call @zdnnx_sigmoid([[VAR_108_]], [[VAR_109_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_sigmoid() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_sigmoid() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether the lowering of zlow.add calls the correct zDNN API or not.
func.func @test_call_zdnn_add() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %2 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.add"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnnx_add(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_add() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_9_:%.+]] = llvm.getelementptr [[VAR_8_]]{{.}}[[VAR_7_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_10_:%.+]] = llvm.ptrtoint [[VAR_9_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.call @malloc([[VAR_10_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_13_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_12_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_13_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_15_]], [[VAR_14_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_16_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_18_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_17_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_18_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_19_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_6_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_22_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_21_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_23_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_22_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.insertvalue [[VAR_4_]], [[VAR_23_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_29_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_34_:%.+]] = llvm.getelementptr [[VAR_33_]]{{.}}[[VAR_32_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_35_:%.+]] = llvm.ptrtoint [[VAR_34_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.call @malloc([[VAR_35_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_37_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_38_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_39_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_42_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_26_]], [[VAR_42_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_44_:%.+]] = llvm.insertvalue [[VAR_27_]], [[VAR_43_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_45_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_44_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_46_:%.+]] = llvm.insertvalue [[VAR_31_]], [[VAR_45_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_47_:%.+]] = llvm.insertvalue [[VAR_30_]], [[VAR_46_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_48_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_47_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.insertvalue [[VAR_29_]], [[VAR_48_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_50_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_54_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_57_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_59_:%.+]] = llvm.getelementptr [[VAR_58_]]{{.}}[[VAR_57_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_60_:%.+]] = llvm.ptrtoint [[VAR_59_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_61_:%.+]] = llvm.call @malloc([[VAR_60_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_62_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_63_:%.+]] = llvm.insertvalue [[VAR_61_]], [[VAR_62_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_64_:%.+]] = llvm.insertvalue [[VAR_61_]], [[VAR_63_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_66_:%.+]] = llvm.insertvalue [[VAR_65_]], [[VAR_64_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_67_:%.+]] = llvm.insertvalue [[VAR_50_]], [[VAR_66_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_68_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_67_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_69_:%.+]] = llvm.insertvalue [[VAR_52_]], [[VAR_68_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_70_:%.+]] = llvm.insertvalue [[VAR_53_]], [[VAR_69_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_71_:%.+]] = llvm.insertvalue [[VAR_56_]], [[VAR_70_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_72_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_71_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_73_:%.+]] = llvm.insertvalue [[VAR_53_]], [[VAR_72_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.insertvalue [[VAR_54_]], [[VAR_73_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-DAG:       [[VAR_76_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_78_:%.+]] = llvm.getelementptr [[VAR_77_]]{{.}}[[VAR_75_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_79_:%.+]] = llvm.ptrtoint [[VAR_78_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.call @malloc([[VAR_79_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_82_:%.+]] = llvm.insertvalue [[VAR_80_]], [[VAR_81_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.insertvalue [[VAR_80_]], [[VAR_82_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_85_:%.+]] = llvm.insertvalue [[VAR_84_]], [[VAR_83_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_86_:%.+]] = llvm.insertvalue [[VAR_75_]], [[VAR_85_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_87_:%.+]] = llvm.insertvalue [[VAR_76_]], [[VAR_86_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_88_:%.+]] = llvm.extractvalue [[VAR_87_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_89_:%.+]] = llvm.getelementptr [[VAR_88_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_89_MEM_:%.+]] = llvm.load [[VAR_89_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_91_:%.+]] = llvm.getelementptr [[VAR_88_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_91_MEM_:%.+]] = llvm.load [[VAR_91_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.extractvalue [[VAR_24_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.bitcast [[VAR_93_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.alloca [[VAR_96_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_98_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_99_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_100_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_98_]], [[VAR_99_]], [[VAR_100_]], [[LOAD_VAR_89_MEM_]], [[LOAD_VAR_91_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_101_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.alloca [[VAR_101_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_104_:%.+]] = llvm.bitcast [[VAR_102_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_105_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_103_]], [[VAR_104_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.alloca [[VAR_95_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_107_:%.+]] = llvm.bitcast [[VAR_102_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_107_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_109_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_110_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_110_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_111_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_111_]] : i64, !llvm.ptr
// CHECK:           [[VAR_112_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_94_]], [[VAR_112_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_113_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_114_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_113_]], [[VAR_114_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_115_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_116_]], [[VAR_115_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_118_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_118_]], [[VAR_117_]] : f32, !llvm.ptr
// CHECK:           [[VAR_119_:%.+]] = llvm.extractvalue [[VAR_49_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.bitcast [[VAR_119_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_122_:%.+]] = llvm.alloca [[VAR_121_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_123_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_123_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_124_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_124_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_125_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_125_]] : i64, !llvm.ptr
// CHECK:           [[VAR_126_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_120_]], [[VAR_126_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_127_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_128_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_127_]], [[VAR_128_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_129_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_130_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_130_]], [[VAR_129_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_131_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_132_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_132_]], [[VAR_131_]] : f32, !llvm.ptr
// CHECK:           [[VAR_133_:%.+]] = llvm.extractvalue [[VAR_74_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_134_:%.+]] = llvm.bitcast [[VAR_133_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_135_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_136_:%.+]] = llvm.alloca [[VAR_135_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_137_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_137_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_138_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_138_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_139_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_139_]] : i64, !llvm.ptr
// CHECK:           [[VAR_140_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_134_]], [[VAR_140_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_141_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_142_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_141_]], [[VAR_142_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_143_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_144_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_144_]], [[VAR_143_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_145_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_146_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_146_]], [[VAR_145_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_147_:%.+]] = llvm.bitcast [[VAR_106_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_148_:%.+]] = llvm.bitcast [[VAR_122_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_149_:%.+]] = llvm.bitcast [[VAR_136_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_150_:%.+]] = llvm.call @zdnnx_add([[VAR_147_]], [[VAR_148_]], [[VAR_149_]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_add() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_add() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether the lowering of zlow.sub calls the correct zDNN API or not.
func.func @test_call_zdnn_sub() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %2 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.sub"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnnx_sub(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_sub() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_9_:%.+]] = llvm.getelementptr [[VAR_8_]]{{.}}[[VAR_7_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_10_:%.+]] = llvm.ptrtoint [[VAR_9_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.call @malloc([[VAR_10_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_13_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_12_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_13_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_15_]], [[VAR_14_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_16_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_18_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_17_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_18_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_19_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_6_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_22_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_21_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_23_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_22_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.insertvalue [[VAR_4_]], [[VAR_23_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_29_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_34_:%.+]] = llvm.getelementptr [[VAR_33_]]{{.}}[[VAR_32_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_35_:%.+]] = llvm.ptrtoint [[VAR_34_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.call @malloc([[VAR_35_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_37_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_38_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_39_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_42_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_26_]], [[VAR_42_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_44_:%.+]] = llvm.insertvalue [[VAR_27_]], [[VAR_43_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_45_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_44_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_46_:%.+]] = llvm.insertvalue [[VAR_31_]], [[VAR_45_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_47_:%.+]] = llvm.insertvalue [[VAR_30_]], [[VAR_46_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_48_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_47_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.insertvalue [[VAR_29_]], [[VAR_48_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_50_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_54_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_57_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_59_:%.+]] = llvm.getelementptr [[VAR_58_]]{{.}}[[VAR_57_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_60_:%.+]] = llvm.ptrtoint [[VAR_59_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_61_:%.+]] = llvm.call @malloc([[VAR_60_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_62_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_63_:%.+]] = llvm.insertvalue [[VAR_61_]], [[VAR_62_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_64_:%.+]] = llvm.insertvalue [[VAR_61_]], [[VAR_63_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_66_:%.+]] = llvm.insertvalue [[VAR_65_]], [[VAR_64_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_67_:%.+]] = llvm.insertvalue [[VAR_50_]], [[VAR_66_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_68_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_67_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_69_:%.+]] = llvm.insertvalue [[VAR_52_]], [[VAR_68_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_70_:%.+]] = llvm.insertvalue [[VAR_53_]], [[VAR_69_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_71_:%.+]] = llvm.insertvalue [[VAR_56_]], [[VAR_70_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_72_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_71_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_73_:%.+]] = llvm.insertvalue [[VAR_53_]], [[VAR_72_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.insertvalue [[VAR_54_]], [[VAR_73_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-DAG:       [[VAR_76_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_78_:%.+]] = llvm.getelementptr [[VAR_77_]]{{.}}[[VAR_75_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_79_:%.+]] = llvm.ptrtoint [[VAR_78_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.call @malloc([[VAR_79_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_82_:%.+]] = llvm.insertvalue [[VAR_80_]], [[VAR_81_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.insertvalue [[VAR_80_]], [[VAR_82_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_85_:%.+]] = llvm.insertvalue [[VAR_84_]], [[VAR_83_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_86_:%.+]] = llvm.insertvalue [[VAR_75_]], [[VAR_85_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_87_:%.+]] = llvm.insertvalue [[VAR_76_]], [[VAR_86_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_88_:%.+]] = llvm.extractvalue [[VAR_87_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_89_:%.+]] = llvm.getelementptr [[VAR_88_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_89_MEM_:%.+]] = llvm.load [[VAR_89_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_91_:%.+]] = llvm.getelementptr [[VAR_88_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_91_MEM_:%.+]] = llvm.load [[VAR_91_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.extractvalue [[VAR_24_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.bitcast [[VAR_93_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.alloca [[VAR_96_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_98_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_99_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_100_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_98_]], [[VAR_99_]], [[VAR_100_]], [[LOAD_VAR_89_MEM_]], [[LOAD_VAR_91_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_101_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.alloca [[VAR_101_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_104_:%.+]] = llvm.bitcast [[VAR_102_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_105_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_103_]], [[VAR_104_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.alloca [[VAR_95_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_107_:%.+]] = llvm.bitcast [[VAR_102_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_107_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_109_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_110_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_110_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_111_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_111_]] : i64, !llvm.ptr
// CHECK:           [[VAR_112_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_94_]], [[VAR_112_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_113_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_114_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_113_]], [[VAR_114_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_115_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_116_]], [[VAR_115_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_118_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_118_]], [[VAR_117_]] : f32, !llvm.ptr
// CHECK:           [[VAR_119_:%.+]] = llvm.extractvalue [[VAR_49_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.bitcast [[VAR_119_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_122_:%.+]] = llvm.alloca [[VAR_121_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_123_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_123_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_124_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_124_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_125_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_125_]] : i64, !llvm.ptr
// CHECK:           [[VAR_126_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_120_]], [[VAR_126_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_127_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_128_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_127_]], [[VAR_128_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_129_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_130_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_130_]], [[VAR_129_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_131_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_132_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_132_]], [[VAR_131_]] : f32, !llvm.ptr
// CHECK:           [[VAR_133_:%.+]] = llvm.extractvalue [[VAR_74_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_134_:%.+]] = llvm.bitcast [[VAR_133_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_135_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_136_:%.+]] = llvm.alloca [[VAR_135_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_137_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_137_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_138_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_138_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_139_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_139_]] : i64, !llvm.ptr
// CHECK:           [[VAR_140_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_134_]], [[VAR_140_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_141_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_142_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_141_]], [[VAR_142_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_143_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_144_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_144_]], [[VAR_143_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_145_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_146_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_146_]], [[VAR_145_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_147_:%.+]] = llvm.bitcast [[VAR_106_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_148_:%.+]] = llvm.bitcast [[VAR_122_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_149_:%.+]] = llvm.bitcast [[VAR_136_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_150_:%.+]] = llvm.call @zdnnx_sub([[VAR_147_]], [[VAR_148_]], [[VAR_149_]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_sub() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_sub() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether the lowering of zlow.mul calls the correct zDNN API or not.
func.func @test_call_zdnn_mul() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %2 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.mul"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnnx_mul(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_mul() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_9_:%.+]] = llvm.getelementptr [[VAR_8_]]{{.}}[[VAR_7_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_10_:%.+]] = llvm.ptrtoint [[VAR_9_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.call @malloc([[VAR_10_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_13_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_12_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_13_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_15_]], [[VAR_14_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_16_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_18_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_17_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_18_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_19_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_6_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_22_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_21_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_23_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_22_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.insertvalue [[VAR_4_]], [[VAR_23_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_29_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_34_:%.+]] = llvm.getelementptr [[VAR_33_]]{{.}}[[VAR_32_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_35_:%.+]] = llvm.ptrtoint [[VAR_34_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.call @malloc([[VAR_35_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_37_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_38_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_39_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_42_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_26_]], [[VAR_42_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_44_:%.+]] = llvm.insertvalue [[VAR_27_]], [[VAR_43_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_45_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_44_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_46_:%.+]] = llvm.insertvalue [[VAR_31_]], [[VAR_45_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_47_:%.+]] = llvm.insertvalue [[VAR_30_]], [[VAR_46_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_48_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_47_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.insertvalue [[VAR_29_]], [[VAR_48_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_50_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_54_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_57_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_59_:%.+]] = llvm.getelementptr [[VAR_58_]]{{.}}[[VAR_57_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_60_:%.+]] = llvm.ptrtoint [[VAR_59_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_61_:%.+]] = llvm.call @malloc([[VAR_60_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_62_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_63_:%.+]] = llvm.insertvalue [[VAR_61_]], [[VAR_62_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_64_:%.+]] = llvm.insertvalue [[VAR_61_]], [[VAR_63_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_66_:%.+]] = llvm.insertvalue [[VAR_65_]], [[VAR_64_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_67_:%.+]] = llvm.insertvalue [[VAR_50_]], [[VAR_66_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_68_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_67_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_69_:%.+]] = llvm.insertvalue [[VAR_52_]], [[VAR_68_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_70_:%.+]] = llvm.insertvalue [[VAR_53_]], [[VAR_69_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_71_:%.+]] = llvm.insertvalue [[VAR_56_]], [[VAR_70_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_72_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_71_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_73_:%.+]] = llvm.insertvalue [[VAR_53_]], [[VAR_72_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.insertvalue [[VAR_54_]], [[VAR_73_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-DAG:       [[VAR_76_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_78_:%.+]] = llvm.getelementptr [[VAR_77_]]{{.}}[[VAR_75_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_79_:%.+]] = llvm.ptrtoint [[VAR_78_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.call @malloc([[VAR_79_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_82_:%.+]] = llvm.insertvalue [[VAR_80_]], [[VAR_81_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.insertvalue [[VAR_80_]], [[VAR_82_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_85_:%.+]] = llvm.insertvalue [[VAR_84_]], [[VAR_83_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_86_:%.+]] = llvm.insertvalue [[VAR_75_]], [[VAR_85_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_87_:%.+]] = llvm.insertvalue [[VAR_76_]], [[VAR_86_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_88_:%.+]] = llvm.extractvalue [[VAR_87_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_89_:%.+]] = llvm.getelementptr [[VAR_88_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_89_MEM_:%.+]] = llvm.load [[VAR_89_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_91_:%.+]] = llvm.getelementptr [[VAR_88_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_91_MEM_:%.+]] = llvm.load [[VAR_91_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.extractvalue [[VAR_24_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.bitcast [[VAR_93_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.alloca [[VAR_96_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_98_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_99_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_100_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_98_]], [[VAR_99_]], [[VAR_100_]], [[LOAD_VAR_89_MEM_]], [[LOAD_VAR_91_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_101_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.alloca [[VAR_101_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_104_:%.+]] = llvm.bitcast [[VAR_102_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_105_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_103_]], [[VAR_104_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.alloca [[VAR_95_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_107_:%.+]] = llvm.bitcast [[VAR_102_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_107_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_109_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_110_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_110_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_111_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_111_]] : i64, !llvm.ptr
// CHECK:           [[VAR_112_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_94_]], [[VAR_112_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_113_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_114_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_113_]], [[VAR_114_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_115_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_116_]], [[VAR_115_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_118_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_118_]], [[VAR_117_]] : f32, !llvm.ptr
// CHECK:           [[VAR_119_:%.+]] = llvm.extractvalue [[VAR_49_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.bitcast [[VAR_119_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_122_:%.+]] = llvm.alloca [[VAR_121_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_123_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_123_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_124_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_124_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_125_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_125_]] : i64, !llvm.ptr
// CHECK:           [[VAR_126_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_120_]], [[VAR_126_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_127_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_128_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_127_]], [[VAR_128_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_129_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_130_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_130_]], [[VAR_129_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_131_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_132_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_132_]], [[VAR_131_]] : f32, !llvm.ptr
// CHECK:           [[VAR_133_:%.+]] = llvm.extractvalue [[VAR_74_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_134_:%.+]] = llvm.bitcast [[VAR_133_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_135_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_136_:%.+]] = llvm.alloca [[VAR_135_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_137_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_137_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_138_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_138_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_139_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_139_]] : i64, !llvm.ptr
// CHECK:           [[VAR_140_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_134_]], [[VAR_140_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_141_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_142_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_141_]], [[VAR_142_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_143_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_144_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_144_]], [[VAR_143_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_145_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_146_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_146_]], [[VAR_145_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_147_:%.+]] = llvm.bitcast [[VAR_106_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_148_:%.+]] = llvm.bitcast [[VAR_122_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_149_:%.+]] = llvm.bitcast [[VAR_136_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_150_:%.+]] = llvm.call @zdnnx_mul([[VAR_147_]], [[VAR_148_]], [[VAR_149_]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_mul() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_mul() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether the lowering of zlow.div calls the correct zDNN API or not.
func.func @test_call_zdnn_div() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %2 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.div"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnnx_div(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_div() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_9_:%.+]] = llvm.getelementptr [[VAR_8_]]{{.}}[[VAR_7_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_10_:%.+]] = llvm.ptrtoint [[VAR_9_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.call @malloc([[VAR_10_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_13_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_12_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_13_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_15_]], [[VAR_14_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_16_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_18_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_17_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_18_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_19_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_6_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_22_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_21_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_23_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_22_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.insertvalue [[VAR_4_]], [[VAR_23_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_29_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_34_:%.+]] = llvm.getelementptr [[VAR_33_]]{{.}}[[VAR_32_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_35_:%.+]] = llvm.ptrtoint [[VAR_34_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.call @malloc([[VAR_35_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_37_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_38_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_39_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_42_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_26_]], [[VAR_42_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_44_:%.+]] = llvm.insertvalue [[VAR_27_]], [[VAR_43_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_45_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_44_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_46_:%.+]] = llvm.insertvalue [[VAR_31_]], [[VAR_45_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_47_:%.+]] = llvm.insertvalue [[VAR_30_]], [[VAR_46_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_48_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_47_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.insertvalue [[VAR_29_]], [[VAR_48_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_50_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_54_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_57_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_59_:%.+]] = llvm.getelementptr [[VAR_58_]]{{.}}[[VAR_57_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_60_:%.+]] = llvm.ptrtoint [[VAR_59_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_61_:%.+]] = llvm.call @malloc([[VAR_60_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_62_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_63_:%.+]] = llvm.insertvalue [[VAR_61_]], [[VAR_62_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_64_:%.+]] = llvm.insertvalue [[VAR_61_]], [[VAR_63_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_66_:%.+]] = llvm.insertvalue [[VAR_65_]], [[VAR_64_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_67_:%.+]] = llvm.insertvalue [[VAR_50_]], [[VAR_66_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_68_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_67_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_69_:%.+]] = llvm.insertvalue [[VAR_52_]], [[VAR_68_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_70_:%.+]] = llvm.insertvalue [[VAR_53_]], [[VAR_69_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_71_:%.+]] = llvm.insertvalue [[VAR_56_]], [[VAR_70_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_72_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_71_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_73_:%.+]] = llvm.insertvalue [[VAR_53_]], [[VAR_72_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.insertvalue [[VAR_54_]], [[VAR_73_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-DAG:       [[VAR_76_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_78_:%.+]] = llvm.getelementptr [[VAR_77_]]{{.}}[[VAR_75_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_79_:%.+]] = llvm.ptrtoint [[VAR_78_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.call @malloc([[VAR_79_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_82_:%.+]] = llvm.insertvalue [[VAR_80_]], [[VAR_81_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.insertvalue [[VAR_80_]], [[VAR_82_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_85_:%.+]] = llvm.insertvalue [[VAR_84_]], [[VAR_83_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_86_:%.+]] = llvm.insertvalue [[VAR_75_]], [[VAR_85_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_87_:%.+]] = llvm.insertvalue [[VAR_76_]], [[VAR_86_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_88_:%.+]] = llvm.extractvalue [[VAR_87_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_89_:%.+]] = llvm.getelementptr [[VAR_88_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_89_MEM_:%.+]] = llvm.load [[VAR_89_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_91_:%.+]] = llvm.getelementptr [[VAR_88_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_91_MEM_:%.+]] = llvm.load [[VAR_91_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.extractvalue [[VAR_24_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.bitcast [[VAR_93_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.alloca [[VAR_96_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_98_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_99_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_100_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_98_]], [[VAR_99_]], [[VAR_100_]], [[LOAD_VAR_89_MEM_]], [[LOAD_VAR_91_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_101_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.alloca [[VAR_101_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_104_:%.+]] = llvm.bitcast [[VAR_102_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_105_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_103_]], [[VAR_104_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.alloca [[VAR_95_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_107_:%.+]] = llvm.bitcast [[VAR_102_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_107_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_109_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_110_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_110_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_111_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_111_]] : i64, !llvm.ptr
// CHECK:           [[VAR_112_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_94_]], [[VAR_112_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_113_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_114_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_113_]], [[VAR_114_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_115_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_116_]], [[VAR_115_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_118_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_118_]], [[VAR_117_]] : f32, !llvm.ptr
// CHECK:           [[VAR_119_:%.+]] = llvm.extractvalue [[VAR_49_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.bitcast [[VAR_119_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_122_:%.+]] = llvm.alloca [[VAR_121_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_123_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_123_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_124_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_124_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_125_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_125_]] : i64, !llvm.ptr
// CHECK:           [[VAR_126_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_120_]], [[VAR_126_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_127_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_128_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_127_]], [[VAR_128_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_129_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_130_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_130_]], [[VAR_129_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_131_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_132_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_132_]], [[VAR_131_]] : f32, !llvm.ptr
// CHECK:           [[VAR_133_:%.+]] = llvm.extractvalue [[VAR_74_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_134_:%.+]] = llvm.bitcast [[VAR_133_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_135_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_136_:%.+]] = llvm.alloca [[VAR_135_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_137_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_137_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_138_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_138_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_139_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_139_]] : i64, !llvm.ptr
// CHECK:           [[VAR_140_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_134_]], [[VAR_140_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_141_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_142_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_141_]], [[VAR_142_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_143_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_144_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_144_]], [[VAR_143_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_145_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_146_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_146_]], [[VAR_145_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_147_:%.+]] = llvm.bitcast [[VAR_106_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_148_:%.+]] = llvm.bitcast [[VAR_122_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_149_:%.+]] = llvm.bitcast [[VAR_136_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_150_:%.+]] = llvm.call @zdnnx_div([[VAR_147_]], [[VAR_148_]], [[VAR_149_]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_div() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_div() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether the lowering of zlow.softmax calls the correct zDNN API or not.
func.func @test_call_zdnn_softmax() -> () {
  %0 = memref.alloc() : memref<1x1x1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x1x1x32x64xf16>
  %work_area = memref.alloc() alignment = 4096 : memref<8192xi8>
  %shape = memref.alloc() : memref<3xi64>
  "zlow.softmax"(%0, %work_area, %shape, %1) {act_func = "ACT_NONE"} : (memref<1x1x1x1x32x64xf16>, memref<8192xi8>, memref<3xi64>, memref<1x1x1x1x32x64xf16>) -> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnnx_softmax(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_softmax() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_9_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_10_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_13_:%.+]] = llvm.getelementptr [[VAR_12_]]{{.}}[[VAR_11_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_14_:%.+]] = llvm.ptrtoint [[VAR_13_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.call @malloc([[VAR_14_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_16_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_15_]], [[VAR_16_]][0] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK-DAG:       [[VAR_18_:%.+]] = llvm.insertvalue [[VAR_15_]], [[VAR_17_]][1] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK-DAG:       [[VAR_19_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_19_]], [[VAR_18_]][2] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_20_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_22_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_21_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_23_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_22_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_24_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_23_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_25_:%.+]] = llvm.insertvalue [[VAR_4_]], [[VAR_24_]][3, 4] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_26_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_25_]][3, 5] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_27_:%.+]] = llvm.insertvalue [[VAR_10_]], [[VAR_26_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_28_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_27_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_29_:%.+]] = llvm.insertvalue [[VAR_8_]], [[VAR_28_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_30_:%.+]] = llvm.insertvalue [[VAR_7_]], [[VAR_29_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_31_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_30_]][4, 4] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.insertvalue [[VAR_6_]], [[VAR_31_]][4, 5] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_34_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_35_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_38_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_41_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_42_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_43_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_44_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_45_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_46_:%.+]] = llvm.getelementptr [[VAR_45_]]{{.}}[[VAR_44_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_47_:%.+]] = llvm.ptrtoint [[VAR_46_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_48_:%.+]] = llvm.call @malloc([[VAR_47_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_50_:%.+]] = llvm.insertvalue [[VAR_48_]], [[VAR_49_]][0] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.insertvalue [[VAR_48_]], [[VAR_50_]][1] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_53_:%.+]] = llvm.insertvalue [[VAR_52_]], [[VAR_51_]][2] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_54_:%.+]] = llvm.insertvalue [[VAR_33_]], [[VAR_53_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_55_:%.+]] = llvm.insertvalue [[VAR_34_]], [[VAR_54_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_56_:%.+]] = llvm.insertvalue [[VAR_35_]], [[VAR_55_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_57_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_56_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_58_:%.+]] = llvm.insertvalue [[VAR_37_]], [[VAR_57_]][3, 4] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_59_:%.+]] = llvm.insertvalue [[VAR_38_]], [[VAR_58_]][3, 5] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_60_:%.+]] = llvm.insertvalue [[VAR_43_]], [[VAR_59_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_61_:%.+]] = llvm.insertvalue [[VAR_42_]], [[VAR_60_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_62_:%.+]] = llvm.insertvalue [[VAR_41_]], [[VAR_61_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_63_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_62_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_64_:%.+]] = llvm.insertvalue [[VAR_38_]], [[VAR_63_]][4, 4] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.insertvalue [[VAR_39_]], [[VAR_64_]][4, 5] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.mlir.constant(8192 : index) : i64
// CHECK-DAG:       [[VAR_67_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_69_:%.+]] = llvm.getelementptr [[VAR_68_]]{{.}}[[VAR_66_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-DAG:       [[VAR_70_:%.+]] = llvm.ptrtoint [[VAR_69_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_72_:%.+]] = llvm.add [[VAR_70_]], [[VAR_71_]] : i64
// CHECK:           [[VAR_73_:%.+]] = llvm.call @malloc([[VAR_72_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.ptrtoint [[VAR_73_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_76_:%.+]] = llvm.sub [[VAR_71_]], [[VAR_75_]] : i64
// CHECK:           [[VAR_77_:%.+]] = llvm.add [[VAR_74_]], [[VAR_76_]] : i64
// CHECK:           [[VAR_78_:%.+]] = llvm.urem [[VAR_77_]], [[VAR_71_]] : i64
// CHECK:           [[VAR_79_:%.+]] = llvm.sub [[VAR_77_]], [[VAR_78_]] : i64
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.inttoptr [[VAR_79_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_82_:%.+]] = llvm.insertvalue [[VAR_73_]], [[VAR_81_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.insertvalue [[VAR_80_]], [[VAR_82_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_85_:%.+]] = llvm.insertvalue [[VAR_84_]], [[VAR_83_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_86_:%.+]] = llvm.insertvalue [[VAR_66_]], [[VAR_85_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_87_:%.+]] = llvm.insertvalue [[VAR_67_]], [[VAR_86_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_88_:%.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-DAG:       [[VAR_89_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_90_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_91_:%.+]] = llvm.getelementptr [[VAR_90_]]{{.}}[[VAR_88_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_92_:%.+]] = llvm.ptrtoint [[VAR_91_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.call @malloc([[VAR_92_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_95_:%.+]] = llvm.insertvalue [[VAR_93_]], [[VAR_94_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.insertvalue [[VAR_93_]], [[VAR_95_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_98_:%.+]] = llvm.insertvalue [[VAR_97_]], [[VAR_96_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_99_:%.+]] = llvm.insertvalue [[VAR_88_]], [[VAR_98_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_100_:%.+]] = llvm.insertvalue [[VAR_89_]], [[VAR_99_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_101_:%.+]] = llvm.extractvalue [[VAR_100_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_102_:%.+]] = llvm.getelementptr [[VAR_101_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_102_MEM_:%.+]] = llvm.load [[VAR_102_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.getelementptr [[VAR_101_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_104_MEM_:%.+]] = llvm.load [[VAR_104_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.getelementptr [[VAR_101_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_106_MEM_:%.+]] = llvm.load [[VAR_106_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.extractvalue [[VAR_32_]][1] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.bitcast [[VAR_108_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_110_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_111_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_112_:%.+]] = llvm.alloca [[VAR_111_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_113_:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:       [[VAR_114_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_115_:%.+]] = llvm.bitcast [[VAR_112_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_113_]], [[VAR_114_]], [[VAR_115_]], [[LOAD_VAR_102_MEM_]], [[LOAD_VAR_104_MEM_]], [[LOAD_VAR_106_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64) -> ()
// CHECK:           [[VAR_116_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.alloca [[VAR_116_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_118_:%.+]] = llvm.bitcast [[VAR_112_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_119_:%.+]] = llvm.bitcast [[VAR_117_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_118_]], [[VAR_119_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.alloca [[VAR_110_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_122_:%.+]] = llvm.bitcast [[VAR_117_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_123_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_122_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_124_:%.+]] = llvm.getelementptr [[VAR_121_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_112_]], [[VAR_124_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_125_:%.+]] = llvm.getelementptr [[VAR_121_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_117_]], [[VAR_125_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_126_:%.+]] = llvm.getelementptr [[VAR_121_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_123_]], [[VAR_126_]] : i64, !llvm.ptr
// CHECK:           [[VAR_127_:%.+]] = llvm.getelementptr [[VAR_121_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_109_]], [[VAR_127_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_128_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_129_:%.+]] = llvm.getelementptr [[VAR_121_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_128_]], [[VAR_129_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_130_:%.+]] = llvm.getelementptr [[VAR_121_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_131_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_131_]], [[VAR_130_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_132_:%.+]] = llvm.getelementptr [[VAR_121_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_133_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_133_]], [[VAR_132_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_134_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_135_:%.+]] = llvm.extractvalue [[VAR_65_]][1] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_136_:%.+]] = llvm.bitcast [[VAR_135_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_137_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_138_:%.+]] = llvm.alloca [[VAR_137_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_139_:%.+]] = llvm.getelementptr [[VAR_138_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_112_]], [[VAR_139_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_140_:%.+]] = llvm.getelementptr [[VAR_138_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_117_]], [[VAR_140_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_141_:%.+]] = llvm.getelementptr [[VAR_138_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_123_]], [[VAR_141_]] : i64, !llvm.ptr
// CHECK:           [[VAR_142_:%.+]] = llvm.getelementptr [[VAR_138_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_136_]], [[VAR_142_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_143_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_144_:%.+]] = llvm.getelementptr [[VAR_138_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_143_]], [[VAR_144_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_145_:%.+]] = llvm.getelementptr [[VAR_138_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_146_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_146_]], [[VAR_145_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_147_:%.+]] = llvm.getelementptr [[VAR_138_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_148_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_148_]], [[VAR_147_]] : f32, !llvm.ptr
// CHECK:           [[VAR_149_:%.+]] = llvm.extractvalue [[VAR_87_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_150_:%.+]] = llvm.bitcast [[VAR_149_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_151_:%.+]] = llvm.bitcast [[VAR_121_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_152_:%.+]] = llvm.bitcast [[VAR_138_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_153_:%.+]] = llvm.call @zdnnx_softmax([[VAR_151_]], [[VAR_150_]], [[VAR_134_]], [[VAR_152_]]) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_softmax() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_softmax() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// COM: Check whether the lowering of zlow.stickForLSTM calls the correct zDNN API or not.
// COM: We should call zdnn_transform_ztensor with zTensor and four pointers to the buffers fori F, I, C, and O gates.
func.func @test_stick_for_lstm() -> () {
  %f = memref.alloc() : memref<1x10x10xf32>
  %i = memref.alloc() : memref<1x10x10xf32>
  %c = memref.alloc() : memref<1x10x10xf32>
  %o = memref.alloc() : memref<1x10x10xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  "zlow.stickForLSTM"(%f, %i, %c, %o, %1) : (memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x1x32x64xf16>) -> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnn_transform_ztensor(!llvm.ptr, ...) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc_concatenated(!llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_stick_for_lstm() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.mlir.constant(100 : index) : i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(100 : index) : i64
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_7_:%.+]] = llvm.getelementptr [[VAR_6_]]{{.}}[[VAR_5_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           [[VAR_8_:%.+]] = llvm.ptrtoint [[VAR_7_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_9_:%.+]] = llvm.call @malloc([[VAR_8_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_10_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_11_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_10_]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_11_]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_13_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_14_:%.+]] = llvm.insertvalue [[VAR_13_]], [[VAR_12_]][2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_15_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_14_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_15_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_16_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_18_:%.+]] = llvm.insertvalue [[VAR_4_]], [[VAR_17_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_18_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_19_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_21_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_22_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_23_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.constant(100 : index) : i64
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.constant(100 : index) : i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_28_:%.+]] = llvm.getelementptr [[VAR_27_]]{{.}}[[VAR_26_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           [[VAR_29_:%.+]] = llvm.ptrtoint [[VAR_28_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.call @malloc([[VAR_29_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_32_:%.+]] = llvm.insertvalue [[VAR_30_]], [[VAR_31_]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.insertvalue [[VAR_30_]], [[VAR_32_]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_34_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_35_:%.+]] = llvm.insertvalue [[VAR_34_]], [[VAR_33_]][2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_36_:%.+]] = llvm.insertvalue [[VAR_21_]], [[VAR_35_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_37_:%.+]] = llvm.insertvalue [[VAR_22_]], [[VAR_36_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_23_]], [[VAR_37_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_38_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_40_:%.+]] = llvm.insertvalue [[VAR_23_]], [[VAR_39_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_24_]], [[VAR_40_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_42_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_43_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_44_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_45_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_46_:%.+]] = llvm.mlir.constant(100 : index) : i64
// CHECK-DAG:       [[VAR_47_:%.+]] = llvm.mlir.constant(100 : index) : i64
// CHECK-DAG:       [[VAR_48_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_49_:%.+]] = llvm.getelementptr [[VAR_48_]]{{.}}[[VAR_47_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           [[VAR_50_:%.+]] = llvm.ptrtoint [[VAR_49_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.call @malloc([[VAR_50_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_53_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_52_]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_54_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_53_]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_56_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_54_]][2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_57_:%.+]] = llvm.insertvalue [[VAR_42_]], [[VAR_56_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_58_:%.+]] = llvm.insertvalue [[VAR_43_]], [[VAR_57_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_59_:%.+]] = llvm.insertvalue [[VAR_44_]], [[VAR_58_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_60_:%.+]] = llvm.insertvalue [[VAR_46_]], [[VAR_59_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_61_:%.+]] = llvm.insertvalue [[VAR_44_]], [[VAR_60_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_62_:%.+]] = llvm.insertvalue [[VAR_45_]], [[VAR_61_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_63_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_64_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_67_:%.+]] = llvm.mlir.constant(100 : index) : i64
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.mlir.constant(100 : index) : i64
// CHECK-DAG:       [[VAR_69_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_70_:%.+]] = llvm.getelementptr [[VAR_69_]]{{.}}[[VAR_68_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           [[VAR_71_:%.+]] = llvm.ptrtoint [[VAR_70_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_72_:%.+]] = llvm.call @malloc([[VAR_71_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_73_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_74_:%.+]] = llvm.insertvalue [[VAR_72_]], [[VAR_73_]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.insertvalue [[VAR_72_]], [[VAR_74_]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_76_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_77_:%.+]] = llvm.insertvalue [[VAR_76_]], [[VAR_75_]][2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_78_:%.+]] = llvm.insertvalue [[VAR_63_]], [[VAR_77_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_79_:%.+]] = llvm.insertvalue [[VAR_64_]], [[VAR_78_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_80_:%.+]] = llvm.insertvalue [[VAR_65_]], [[VAR_79_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_81_:%.+]] = llvm.insertvalue [[VAR_67_]], [[VAR_80_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_82_:%.+]] = llvm.insertvalue [[VAR_65_]], [[VAR_81_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.insertvalue [[VAR_66_]], [[VAR_82_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_85_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_86_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_87_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_88_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_89_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_90_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_91_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_92_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_93_:%.+]] = llvm.getelementptr [[VAR_92_]]{{.}}[[VAR_91_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_94_:%.+]] = llvm.ptrtoint [[VAR_93_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.call @malloc([[VAR_94_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_97_:%.+]] = llvm.insertvalue [[VAR_95_]], [[VAR_96_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_98_:%.+]] = llvm.insertvalue [[VAR_95_]], [[VAR_97_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_99_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_100_:%.+]] = llvm.insertvalue [[VAR_99_]], [[VAR_98_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_101_:%.+]] = llvm.insertvalue [[VAR_84_]], [[VAR_100_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_102_:%.+]] = llvm.insertvalue [[VAR_85_]], [[VAR_101_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_103_:%.+]] = llvm.insertvalue [[VAR_86_]], [[VAR_102_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_104_:%.+]] = llvm.insertvalue [[VAR_87_]], [[VAR_103_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_105_:%.+]] = llvm.insertvalue [[VAR_90_]], [[VAR_104_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_106_:%.+]] = llvm.insertvalue [[VAR_89_]], [[VAR_105_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_107_:%.+]] = llvm.insertvalue [[VAR_87_]], [[VAR_106_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.insertvalue [[VAR_88_]], [[VAR_107_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.extractvalue [[VAR_20_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_110_:%.+]] = llvm.extractvalue [[VAR_20_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_111_:%.+]] = llvm.extractvalue [[VAR_20_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_112_:%.+]] = llvm.extractvalue [[VAR_108_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_113_:%.+]] = llvm.bitcast [[VAR_112_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_114_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_115_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.alloca [[VAR_115_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:       [[VAR_118_:%.+]] = llvm.mlir.constant(255 : i64) : i64
// CHECK:           [[VAR_119_:%.+]] = llvm.bitcast [[VAR_116_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_117_]], [[VAR_118_]], [[VAR_119_]], [[VAR_109_]], [[VAR_110_]], [[VAR_111_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64) -> ()
// CHECK:           [[VAR_120_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.alloca [[VAR_120_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_122_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_123_:%.+]] = llvm.bitcast [[VAR_116_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_124_:%.+]] = llvm.bitcast [[VAR_121_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_125_:%.+]] = llvm.call @zdnn_generate_transformed_desc_concatenated([[VAR_123_]], [[VAR_122_]], [[VAR_124_]]) : (!llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_126_:%.+]] = llvm.alloca [[VAR_114_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_127_:%.+]] = llvm.bitcast [[VAR_121_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_128_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_127_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_129_:%.+]] = llvm.getelementptr [[VAR_126_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_116_]], [[VAR_129_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_130_:%.+]] = llvm.getelementptr [[VAR_126_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_121_]], [[VAR_130_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_131_:%.+]] = llvm.getelementptr [[VAR_126_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_128_]], [[VAR_131_]] : i64, !llvm.ptr
// CHECK:           [[VAR_132_:%.+]] = llvm.getelementptr [[VAR_126_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_113_]], [[VAR_132_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_133_:%.+]] = llvm.mlir.constant(false) : i1
// CHECK-DAG:       [[VAR_134_:%.+]] = llvm.getelementptr [[VAR_126_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_133_]], [[VAR_134_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_135_:%.+]] = llvm.getelementptr [[VAR_126_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_136_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_136_]], [[VAR_135_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_137_:%.+]] = llvm.getelementptr [[VAR_126_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_138_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_138_]], [[VAR_137_]] : f32, !llvm.ptr
// CHECK:           [[VAR_139_:%.+]] = llvm.extractvalue [[VAR_20_]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_140_:%.+]] = llvm.bitcast [[VAR_139_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_141_:%.+]] = llvm.extractvalue [[VAR_41_]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_142_:%.+]] = llvm.bitcast [[VAR_141_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_143_:%.+]] = llvm.extractvalue [[VAR_62_]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_144_:%.+]] = llvm.bitcast [[VAR_143_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_145_:%.+]] = llvm.extractvalue [[VAR_83_]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_146_:%.+]] = llvm.bitcast [[VAR_145_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_147_:%.+]] = llvm.bitcast [[VAR_126_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_148_:%.+]] = llvm.call @zdnn_transform_ztensor([[VAR_147_]], [[VAR_140_]], [[VAR_142_]], [[VAR_144_]], [[VAR_146_]]) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_stick_for_lstm() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_stick_for_lstm() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// COM: Check whether the lowering of zlow.stickForGRU calls the correct zDNN API or not.
// COM: We should call zdnn_transform_ztensor with zTensor and three pointers to the buffers for Z, R, and H gates.
func.func @test_stick_for_gru() -> () {
  %g = memref.alloc() : memref<1x10x10xf32>
  %r = memref.alloc() : memref<1x10x10xf32>
  %h = memref.alloc() : memref<1x10x10xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  "zlow.stickForGRU"(%g, %r, %h, %1) : (memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x1x32x64xf16>) -> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnn_transform_ztensor(!llvm.ptr, ...) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc_concatenated(!llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_stick_for_gru() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.mlir.constant(100 : index) : i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(100 : index) : i64
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_7_:%.+]] = llvm.getelementptr [[VAR_6_]]{{.}}[[VAR_5_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           [[VAR_8_:%.+]] = llvm.ptrtoint [[VAR_7_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_9_:%.+]] = llvm.call @malloc([[VAR_8_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_10_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_11_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_10_]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_11_]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_13_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_14_:%.+]] = llvm.insertvalue [[VAR_13_]], [[VAR_12_]][2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_15_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_14_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_15_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_16_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_18_:%.+]] = llvm.insertvalue [[VAR_4_]], [[VAR_17_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_18_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_19_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_21_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_22_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_23_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.constant(100 : index) : i64
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.constant(100 : index) : i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_28_:%.+]] = llvm.getelementptr [[VAR_27_]]{{.}}[[VAR_26_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           [[VAR_29_:%.+]] = llvm.ptrtoint [[VAR_28_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.call @malloc([[VAR_29_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_32_:%.+]] = llvm.insertvalue [[VAR_30_]], [[VAR_31_]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.insertvalue [[VAR_30_]], [[VAR_32_]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_34_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_35_:%.+]] = llvm.insertvalue [[VAR_34_]], [[VAR_33_]][2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_36_:%.+]] = llvm.insertvalue [[VAR_21_]], [[VAR_35_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_37_:%.+]] = llvm.insertvalue [[VAR_22_]], [[VAR_36_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_23_]], [[VAR_37_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_38_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_40_:%.+]] = llvm.insertvalue [[VAR_23_]], [[VAR_39_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_24_]], [[VAR_40_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_42_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_43_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_44_:%.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-DAG:       [[VAR_45_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_46_:%.+]] = llvm.mlir.constant(100 : index) : i64
// CHECK-DAG:       [[VAR_47_:%.+]] = llvm.mlir.constant(100 : index) : i64
// CHECK-DAG:       [[VAR_48_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_49_:%.+]] = llvm.getelementptr [[VAR_48_]]{{.}}[[VAR_47_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           [[VAR_50_:%.+]] = llvm.ptrtoint [[VAR_49_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.call @malloc([[VAR_50_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_53_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_52_]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_54_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_53_]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_56_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_54_]][2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_57_:%.+]] = llvm.insertvalue [[VAR_42_]], [[VAR_56_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_58_:%.+]] = llvm.insertvalue [[VAR_43_]], [[VAR_57_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_59_:%.+]] = llvm.insertvalue [[VAR_44_]], [[VAR_58_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_60_:%.+]] = llvm.insertvalue [[VAR_46_]], [[VAR_59_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_61_:%.+]] = llvm.insertvalue [[VAR_44_]], [[VAR_60_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_62_:%.+]] = llvm.insertvalue [[VAR_45_]], [[VAR_61_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_63_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_64_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_67_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_69_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_70_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_72_:%.+]] = llvm.getelementptr [[VAR_71_]]{{.}}[[VAR_70_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_73_:%.+]] = llvm.ptrtoint [[VAR_72_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.call @malloc([[VAR_73_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_76_:%.+]] = llvm.insertvalue [[VAR_74_]], [[VAR_75_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.insertvalue [[VAR_74_]], [[VAR_76_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_78_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_79_:%.+]] = llvm.insertvalue [[VAR_78_]], [[VAR_77_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_80_:%.+]] = llvm.insertvalue [[VAR_63_]], [[VAR_79_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_81_:%.+]] = llvm.insertvalue [[VAR_64_]], [[VAR_80_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_82_:%.+]] = llvm.insertvalue [[VAR_65_]], [[VAR_81_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_83_:%.+]] = llvm.insertvalue [[VAR_66_]], [[VAR_82_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_84_:%.+]] = llvm.insertvalue [[VAR_69_]], [[VAR_83_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_85_:%.+]] = llvm.insertvalue [[VAR_68_]], [[VAR_84_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_86_:%.+]] = llvm.insertvalue [[VAR_66_]], [[VAR_85_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_87_:%.+]] = llvm.insertvalue [[VAR_67_]], [[VAR_86_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_88_:%.+]] = llvm.extractvalue [[VAR_20_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_89_:%.+]] = llvm.extractvalue [[VAR_20_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_90_:%.+]] = llvm.extractvalue [[VAR_20_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_91_:%.+]] = llvm.extractvalue [[VAR_87_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_92_:%.+]] = llvm.bitcast [[VAR_91_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.alloca [[VAR_94_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.mlir.constant(255 : i64) : i64
// CHECK:           [[VAR_98_:%.+]] = llvm.bitcast [[VAR_95_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_96_]], [[VAR_97_]], [[VAR_98_]], [[VAR_88_]], [[VAR_89_]], [[VAR_90_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64) -> ()
// CHECK:           [[VAR_99_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_100_:%.+]] = llvm.alloca [[VAR_99_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_101_:%.+]] = llvm.mlir.constant(16777216 : i64) : i64
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.bitcast [[VAR_95_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_103_:%.+]] = llvm.bitcast [[VAR_100_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.call @zdnn_generate_transformed_desc_concatenated([[VAR_102_]], [[VAR_101_]], [[VAR_103_]]) : (!llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_105_:%.+]] = llvm.alloca [[VAR_93_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.bitcast [[VAR_100_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_107_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_106_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.getelementptr [[VAR_105_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_95_]], [[VAR_108_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_109_:%.+]] = llvm.getelementptr [[VAR_105_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_100_]], [[VAR_109_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_110_:%.+]] = llvm.getelementptr [[VAR_105_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_107_]], [[VAR_110_]] : i64, !llvm.ptr
// CHECK:           [[VAR_111_:%.+]] = llvm.getelementptr [[VAR_105_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_92_]], [[VAR_111_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_112_:%.+]] = llvm.mlir.constant(false) : i1
// CHECK-DAG:       [[VAR_113_:%.+]] = llvm.getelementptr [[VAR_105_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_112_]], [[VAR_113_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_114_:%.+]] = llvm.getelementptr [[VAR_105_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_115_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_115_]], [[VAR_114_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.getelementptr [[VAR_105_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_117_]], [[VAR_116_]] : f32, !llvm.ptr
// CHECK:           [[VAR_118_:%.+]] = llvm.extractvalue [[VAR_20_]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_119_:%.+]] = llvm.bitcast [[VAR_118_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.extractvalue [[VAR_41_]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.bitcast [[VAR_120_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_122_:%.+]] = llvm.extractvalue [[VAR_62_]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_123_:%.+]] = llvm.bitcast [[VAR_122_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_124_:%.+]] = llvm.bitcast [[VAR_105_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_125_:%.+]] = llvm.call @zdnn_transform_ztensor([[VAR_124_]], [[VAR_119_]], [[VAR_121_]], [[VAR_123_]]) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_stick_for_gru() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_stick_for_gru() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether the lowering of zlow.min calls the correct zDNN API or not.
func.func @test_call_zdnn_min() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %2 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.min"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnnx_min(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_min() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_9_:%.+]] = llvm.getelementptr [[VAR_8_]]{{.}}[[VAR_7_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_10_:%.+]] = llvm.ptrtoint [[VAR_9_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.call @malloc([[VAR_10_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_13_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_12_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_13_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_15_]], [[VAR_14_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_16_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_18_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_17_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_18_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_19_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_6_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_22_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_21_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_23_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_22_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.insertvalue [[VAR_4_]], [[VAR_23_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_29_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_34_:%.+]] = llvm.getelementptr [[VAR_33_]]{{.}}[[VAR_32_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_35_:%.+]] = llvm.ptrtoint [[VAR_34_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.call @malloc([[VAR_35_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_37_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_38_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_39_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_42_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_26_]], [[VAR_42_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_44_:%.+]] = llvm.insertvalue [[VAR_27_]], [[VAR_43_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_45_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_44_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_46_:%.+]] = llvm.insertvalue [[VAR_31_]], [[VAR_45_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_47_:%.+]] = llvm.insertvalue [[VAR_30_]], [[VAR_46_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_48_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_47_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.insertvalue [[VAR_29_]], [[VAR_48_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_50_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_54_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_57_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_59_:%.+]] = llvm.getelementptr [[VAR_58_]]{{.}}[[VAR_57_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_60_:%.+]] = llvm.ptrtoint [[VAR_59_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_61_:%.+]] = llvm.call @malloc([[VAR_60_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_62_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_63_:%.+]] = llvm.insertvalue [[VAR_61_]], [[VAR_62_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_64_:%.+]] = llvm.insertvalue [[VAR_61_]], [[VAR_63_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_66_:%.+]] = llvm.insertvalue [[VAR_65_]], [[VAR_64_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_67_:%.+]] = llvm.insertvalue [[VAR_50_]], [[VAR_66_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_68_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_67_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_69_:%.+]] = llvm.insertvalue [[VAR_52_]], [[VAR_68_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_70_:%.+]] = llvm.insertvalue [[VAR_53_]], [[VAR_69_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_71_:%.+]] = llvm.insertvalue [[VAR_56_]], [[VAR_70_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_72_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_71_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_73_:%.+]] = llvm.insertvalue [[VAR_53_]], [[VAR_72_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.insertvalue [[VAR_54_]], [[VAR_73_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-DAG:       [[VAR_76_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_78_:%.+]] = llvm.getelementptr [[VAR_77_]]{{.}}[[VAR_75_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_79_:%.+]] = llvm.ptrtoint [[VAR_78_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.call @malloc([[VAR_79_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_82_:%.+]] = llvm.insertvalue [[VAR_80_]], [[VAR_81_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.insertvalue [[VAR_80_]], [[VAR_82_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_85_:%.+]] = llvm.insertvalue [[VAR_84_]], [[VAR_83_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_86_:%.+]] = llvm.insertvalue [[VAR_75_]], [[VAR_85_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_87_:%.+]] = llvm.insertvalue [[VAR_76_]], [[VAR_86_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_88_:%.+]] = llvm.extractvalue [[VAR_87_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_89_:%.+]] = llvm.getelementptr [[VAR_88_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_89_MEM_:%.+]] = llvm.load [[VAR_89_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_91_:%.+]] = llvm.getelementptr [[VAR_88_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_91_MEM_:%.+]] = llvm.load [[VAR_91_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.extractvalue [[VAR_24_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.bitcast [[VAR_93_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.alloca [[VAR_96_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_98_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_99_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_100_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_98_]], [[VAR_99_]], [[VAR_100_]], [[LOAD_VAR_89_MEM_]], [[LOAD_VAR_91_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_101_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.alloca [[VAR_101_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_104_:%.+]] = llvm.bitcast [[VAR_102_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_105_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_103_]], [[VAR_104_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.alloca [[VAR_95_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_107_:%.+]] = llvm.bitcast [[VAR_102_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_107_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_109_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_110_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_110_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_111_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_111_]] : i64, !llvm.ptr
// CHECK:           [[VAR_112_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_94_]], [[VAR_112_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_113_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_114_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_113_]], [[VAR_114_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_115_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_116_]], [[VAR_115_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_118_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_118_]], [[VAR_117_]] : f32, !llvm.ptr
// CHECK:           [[VAR_119_:%.+]] = llvm.extractvalue [[VAR_49_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.bitcast [[VAR_119_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_122_:%.+]] = llvm.alloca [[VAR_121_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_123_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_123_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_124_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_124_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_125_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_125_]] : i64, !llvm.ptr
// CHECK:           [[VAR_126_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_120_]], [[VAR_126_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_127_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_128_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_127_]], [[VAR_128_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_129_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_130_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_130_]], [[VAR_129_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_131_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_132_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_132_]], [[VAR_131_]] : f32, !llvm.ptr
// CHECK:           [[VAR_133_:%.+]] = llvm.extractvalue [[VAR_74_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_134_:%.+]] = llvm.bitcast [[VAR_133_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_135_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_136_:%.+]] = llvm.alloca [[VAR_135_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_137_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_137_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_138_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_138_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_139_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_139_]] : i64, !llvm.ptr
// CHECK:           [[VAR_140_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_134_]], [[VAR_140_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_141_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_142_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_141_]], [[VAR_142_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_143_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_144_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_144_]], [[VAR_143_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_145_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_146_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_146_]], [[VAR_145_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_147_:%.+]] = llvm.bitcast [[VAR_106_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_148_:%.+]] = llvm.bitcast [[VAR_122_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_149_:%.+]] = llvm.bitcast [[VAR_136_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_150_:%.+]] = llvm.call @zdnnx_min([[VAR_147_]], [[VAR_148_]], [[VAR_149_]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_min() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_min() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether the lowering of zlow.max calls the correct zDNN API or not.
func.func @test_call_zdnn_max() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %2 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.max"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnnx_max(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_max() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_9_:%.+]] = llvm.getelementptr [[VAR_8_]]{{.}}[[VAR_7_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_10_:%.+]] = llvm.ptrtoint [[VAR_9_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.call @malloc([[VAR_10_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_13_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_12_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_13_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_15_]], [[VAR_14_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_16_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_18_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_17_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_18_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_19_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_6_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_22_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_21_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_23_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_22_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.insertvalue [[VAR_4_]], [[VAR_23_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_29_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_34_:%.+]] = llvm.getelementptr [[VAR_33_]]{{.}}[[VAR_32_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_35_:%.+]] = llvm.ptrtoint [[VAR_34_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.call @malloc([[VAR_35_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_37_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_38_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_39_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_42_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_26_]], [[VAR_42_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_44_:%.+]] = llvm.insertvalue [[VAR_27_]], [[VAR_43_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_45_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_44_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_46_:%.+]] = llvm.insertvalue [[VAR_31_]], [[VAR_45_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_47_:%.+]] = llvm.insertvalue [[VAR_30_]], [[VAR_46_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_48_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_47_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.insertvalue [[VAR_29_]], [[VAR_48_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_50_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_54_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_57_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_59_:%.+]] = llvm.getelementptr [[VAR_58_]]{{.}}[[VAR_57_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_60_:%.+]] = llvm.ptrtoint [[VAR_59_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_61_:%.+]] = llvm.call @malloc([[VAR_60_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_62_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_63_:%.+]] = llvm.insertvalue [[VAR_61_]], [[VAR_62_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_64_:%.+]] = llvm.insertvalue [[VAR_61_]], [[VAR_63_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_66_:%.+]] = llvm.insertvalue [[VAR_65_]], [[VAR_64_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_67_:%.+]] = llvm.insertvalue [[VAR_50_]], [[VAR_66_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_68_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_67_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_69_:%.+]] = llvm.insertvalue [[VAR_52_]], [[VAR_68_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_70_:%.+]] = llvm.insertvalue [[VAR_53_]], [[VAR_69_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_71_:%.+]] = llvm.insertvalue [[VAR_56_]], [[VAR_70_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_72_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_71_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_73_:%.+]] = llvm.insertvalue [[VAR_53_]], [[VAR_72_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.insertvalue [[VAR_54_]], [[VAR_73_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-DAG:       [[VAR_76_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_78_:%.+]] = llvm.getelementptr [[VAR_77_]]{{.}}[[VAR_75_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_79_:%.+]] = llvm.ptrtoint [[VAR_78_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.call @malloc([[VAR_79_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_82_:%.+]] = llvm.insertvalue [[VAR_80_]], [[VAR_81_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.insertvalue [[VAR_80_]], [[VAR_82_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_85_:%.+]] = llvm.insertvalue [[VAR_84_]], [[VAR_83_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_86_:%.+]] = llvm.insertvalue [[VAR_75_]], [[VAR_85_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_87_:%.+]] = llvm.insertvalue [[VAR_76_]], [[VAR_86_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_88_:%.+]] = llvm.extractvalue [[VAR_87_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_89_:%.+]] = llvm.getelementptr [[VAR_88_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_89_MEM_:%.+]] = llvm.load [[VAR_89_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_91_:%.+]] = llvm.getelementptr [[VAR_88_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_91_MEM_:%.+]] = llvm.load [[VAR_91_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.extractvalue [[VAR_24_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.bitcast [[VAR_93_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.alloca [[VAR_96_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_98_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_99_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_100_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_98_]], [[VAR_99_]], [[VAR_100_]], [[LOAD_VAR_89_MEM_]], [[LOAD_VAR_91_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_101_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.alloca [[VAR_101_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_104_:%.+]] = llvm.bitcast [[VAR_102_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_105_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_103_]], [[VAR_104_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.alloca [[VAR_95_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_107_:%.+]] = llvm.bitcast [[VAR_102_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_107_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_109_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_110_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_110_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_111_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_111_]] : i64, !llvm.ptr
// CHECK:           [[VAR_112_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_94_]], [[VAR_112_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_113_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_114_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_113_]], [[VAR_114_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_115_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_116_]], [[VAR_115_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_118_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_118_]], [[VAR_117_]] : f32, !llvm.ptr
// CHECK:           [[VAR_119_:%.+]] = llvm.extractvalue [[VAR_49_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.bitcast [[VAR_119_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_122_:%.+]] = llvm.alloca [[VAR_121_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_123_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_123_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_124_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_124_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_125_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_125_]] : i64, !llvm.ptr
// CHECK:           [[VAR_126_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_120_]], [[VAR_126_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_127_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_128_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_127_]], [[VAR_128_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_129_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_130_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_130_]], [[VAR_129_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_131_:%.+]] = llvm.getelementptr [[VAR_122_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_132_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_132_]], [[VAR_131_]] : f32, !llvm.ptr
// CHECK:           [[VAR_133_:%.+]] = llvm.extractvalue [[VAR_74_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_134_:%.+]] = llvm.bitcast [[VAR_133_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_135_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_136_:%.+]] = llvm.alloca [[VAR_135_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_137_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_137_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_138_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_138_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_139_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_139_]] : i64, !llvm.ptr
// CHECK:           [[VAR_140_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_134_]], [[VAR_140_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_141_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_142_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_141_]], [[VAR_142_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_143_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_144_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_144_]], [[VAR_143_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_145_:%.+]] = llvm.getelementptr [[VAR_136_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_146_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_146_]], [[VAR_145_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_147_:%.+]] = llvm.bitcast [[VAR_106_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_148_:%.+]] = llvm.bitcast [[VAR_122_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_149_:%.+]] = llvm.bitcast [[VAR_136_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_150_:%.+]] = llvm.call @zdnnx_max([[VAR_147_]], [[VAR_148_]], [[VAR_149_]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_max() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_max() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether the lowering of zlow.exp calls the correct zDNN API or not.
func.func @test_call_zdnn_exp() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.exp"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnnx_exp(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_exp() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_9_:%.+]] = llvm.getelementptr [[VAR_8_]]{{.}}[[VAR_7_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_10_:%.+]] = llvm.ptrtoint [[VAR_9_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.call @malloc([[VAR_10_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_13_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_12_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_13_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_15_]], [[VAR_14_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_16_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_18_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_17_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_18_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_19_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_6_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_22_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_21_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_23_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_22_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.insertvalue [[VAR_4_]], [[VAR_23_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_29_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_34_:%.+]] = llvm.getelementptr [[VAR_33_]]{{.}}[[VAR_32_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_35_:%.+]] = llvm.ptrtoint [[VAR_34_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.call @malloc([[VAR_35_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_37_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_38_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_39_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_42_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_26_]], [[VAR_42_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_44_:%.+]] = llvm.insertvalue [[VAR_27_]], [[VAR_43_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_45_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_44_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_46_:%.+]] = llvm.insertvalue [[VAR_31_]], [[VAR_45_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_47_:%.+]] = llvm.insertvalue [[VAR_30_]], [[VAR_46_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_48_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_47_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.insertvalue [[VAR_29_]], [[VAR_48_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_50_:%.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_53_:%.+]] = llvm.getelementptr [[VAR_52_]]{{.}}[[VAR_50_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_54_:%.+]] = llvm.ptrtoint [[VAR_53_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.call @malloc([[VAR_54_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_57_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_56_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_57_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_59_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_60_:%.+]] = llvm.insertvalue [[VAR_59_]], [[VAR_58_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_61_:%.+]] = llvm.insertvalue [[VAR_50_]], [[VAR_60_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_62_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_61_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_63_:%.+]] = llvm.extractvalue [[VAR_62_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_64_:%.+]] = llvm.getelementptr [[VAR_63_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_64_MEM_:%.+]] = llvm.load [[VAR_64_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.getelementptr [[VAR_63_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_66_MEM_:%.+]] = llvm.load [[VAR_66_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.extractvalue [[VAR_24_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_69_:%.+]] = llvm.bitcast [[VAR_68_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_70_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_72_:%.+]] = llvm.alloca [[VAR_71_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_73_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_75_:%.+]] = llvm.bitcast [[VAR_72_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_73_]], [[VAR_74_]], [[VAR_75_]], [[LOAD_VAR_64_MEM_]], [[LOAD_VAR_66_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_76_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.alloca [[VAR_76_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_78_:%.+]] = llvm.bitcast [[VAR_72_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_79_:%.+]] = llvm.bitcast [[VAR_77_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_78_]], [[VAR_79_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.alloca [[VAR_70_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_82_:%.+]] = llvm.bitcast [[VAR_77_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_82_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_72_]], [[VAR_84_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_85_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_77_]], [[VAR_85_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_86_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_83_]], [[VAR_86_]] : i64, !llvm.ptr
// CHECK:           [[VAR_87_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_69_]], [[VAR_87_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_88_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_89_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_88_]], [[VAR_89_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_90_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_91_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_91_]], [[VAR_90_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_92_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_93_]], [[VAR_92_]] : f32, !llvm.ptr
// CHECK:           [[VAR_94_:%.+]] = llvm.extractvalue [[VAR_49_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.bitcast [[VAR_94_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_97_:%.+]] = llvm.alloca [[VAR_96_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_98_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_72_]], [[VAR_98_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_99_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_77_]], [[VAR_99_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_100_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_83_]], [[VAR_100_]] : i64, !llvm.ptr
// CHECK:           [[VAR_101_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_95_]], [[VAR_101_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_103_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_105_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_105_]], [[VAR_104_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_107_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_107_]], [[VAR_106_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.bitcast [[VAR_81_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_110_:%.+]] = llvm.call @zdnnx_exp([[VAR_108_]], [[VAR_109_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_exp() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_exp() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether the lowering of zlow.log calls the correct zDNN API or not.
func.func @test_call_zdnn_log() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.log"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnnx_log(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_log() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_9_:%.+]] = llvm.getelementptr [[VAR_8_]]{{.}}[[VAR_7_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_10_:%.+]] = llvm.ptrtoint [[VAR_9_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.call @malloc([[VAR_10_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_13_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_12_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_13_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_15_]], [[VAR_14_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_16_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_18_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_17_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_18_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_19_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_6_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_22_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_21_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_23_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_22_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.insertvalue [[VAR_4_]], [[VAR_23_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_29_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_34_:%.+]] = llvm.getelementptr [[VAR_33_]]{{.}}[[VAR_32_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:           [[VAR_35_:%.+]] = llvm.ptrtoint [[VAR_34_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.call @malloc([[VAR_35_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_37_]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_38_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_39_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_42_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_26_]], [[VAR_42_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_44_:%.+]] = llvm.insertvalue [[VAR_27_]], [[VAR_43_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_45_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_44_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_46_:%.+]] = llvm.insertvalue [[VAR_31_]], [[VAR_45_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_47_:%.+]] = llvm.insertvalue [[VAR_30_]], [[VAR_46_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:           [[VAR_48_:%.+]] = llvm.insertvalue [[VAR_28_]], [[VAR_47_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.insertvalue [[VAR_29_]], [[VAR_48_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_50_:%.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_53_:%.+]] = llvm.getelementptr [[VAR_52_]]{{.}}[[VAR_50_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_54_:%.+]] = llvm.ptrtoint [[VAR_53_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.call @malloc([[VAR_54_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_57_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_56_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.insertvalue [[VAR_55_]], [[VAR_57_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_59_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_60_:%.+]] = llvm.insertvalue [[VAR_59_]], [[VAR_58_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_61_:%.+]] = llvm.insertvalue [[VAR_50_]], [[VAR_60_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_62_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_61_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_63_:%.+]] = llvm.extractvalue [[VAR_62_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_64_:%.+]] = llvm.getelementptr [[VAR_63_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_64_MEM_:%.+]] = llvm.load [[VAR_64_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.getelementptr [[VAR_63_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_66_MEM_:%.+]] = llvm.load [[VAR_66_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.extractvalue [[VAR_24_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_69_:%.+]] = llvm.bitcast [[VAR_68_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_70_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_72_:%.+]] = llvm.alloca [[VAR_71_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_73_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_75_:%.+]] = llvm.bitcast [[VAR_72_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_73_]], [[VAR_74_]], [[VAR_75_]], [[LOAD_VAR_64_MEM_]], [[LOAD_VAR_66_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_76_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.alloca [[VAR_76_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_78_:%.+]] = llvm.bitcast [[VAR_72_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_79_:%.+]] = llvm.bitcast [[VAR_77_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_78_]], [[VAR_79_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.alloca [[VAR_70_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_82_:%.+]] = llvm.bitcast [[VAR_77_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_82_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_72_]], [[VAR_84_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_85_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_77_]], [[VAR_85_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_86_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_83_]], [[VAR_86_]] : i64, !llvm.ptr
// CHECK:           [[VAR_87_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_69_]], [[VAR_87_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_88_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_89_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_88_]], [[VAR_89_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_90_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_91_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_91_]], [[VAR_90_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_92_:%.+]] = llvm.getelementptr [[VAR_81_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_93_]], [[VAR_92_]] : f32, !llvm.ptr
// CHECK:           [[VAR_94_:%.+]] = llvm.extractvalue [[VAR_49_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.bitcast [[VAR_94_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_97_:%.+]] = llvm.alloca [[VAR_96_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_98_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_72_]], [[VAR_98_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_99_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_77_]], [[VAR_99_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_100_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_83_]], [[VAR_100_]] : i64, !llvm.ptr
// CHECK:           [[VAR_101_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_95_]], [[VAR_101_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_103_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_105_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_105_]], [[VAR_104_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.getelementptr [[VAR_97_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_107_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_107_]], [[VAR_106_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.bitcast [[VAR_81_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_110_:%.+]] = llvm.call @zdnnx_log([[VAR_108_]], [[VAR_109_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_log() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_log() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether the lowering of zlow.matmul calls the correct zDNN API or not.
func.func @test_matmul_no_bcast_unstacked(%x: memref<2048xf16>,%y: memref<2048xf16>,%bias: memref<2048xf16>, %shape: memref<3xi64>) -> memref<2048xf16> {
  %res = memref.alloc() alignment = 4096 : memref<2048xf16>
  "zlow.matmul"(%x, %y, %bias, %shape, %res) {is_bcast1 = 0 : si64, is_bcast23 = 0 : si64, is_stacked = 0 : si64} : (memref<2048xf16>, memref<2048xf16>, memref<2048xf16>, memref<3xi64>, memref<2048xf16>) -> ()
  return %res : memref<2048xf16>

// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnnx_matmul_op(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_matmul_no_bcast_unstacked([[arg0_:%.+]]: !llvm.ptr, [[arg1_:%.+]]: !llvm.ptr, [[arg2_:%.+]]: i64, [[arg3_:%.+]]: i64, [[arg4_:%.+]]: i64, [[arg5_:%.+]]: !llvm.ptr, [[arg6_:%.+]]: !llvm.ptr, [[arg7_:%.+]]: i64, [[arg8_:%.+]]: i64, [[arg9_:%.+]]: i64, [[arg10_:%.+]]: !llvm.ptr, [[arg11_:%.+]]: !llvm.ptr, [[arg12_:%.+]]: i64, [[arg13_:%.+]]: i64, [[arg14_:%.+]]: i64, [[arg15_:%.+]]: !llvm.ptr, [[arg16_:%.+]]: !llvm.ptr, [[arg17_:%.+]]: i64, [[arg18_:%.+]]: i64, [[arg19_:%.+]]: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface} {
// CHECK:           [[VAR_0_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_1_:%.+]] = llvm.insertvalue [[arg15_]], [[VAR_0_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_2_:%.+]] = llvm.insertvalue [[arg16_]], [[VAR_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_3_:%.+]] = llvm.insertvalue [[arg17_]], [[VAR_2_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_4_:%.+]] = llvm.insertvalue [[arg18_]], [[VAR_3_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.insertvalue [[arg19_]], [[VAR_4_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_7_:%.+]] = llvm.insertvalue [[arg10_]], [[VAR_6_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_8_:%.+]] = llvm.insertvalue [[arg11_]], [[VAR_7_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_9_:%.+]] = llvm.insertvalue [[arg12_]], [[VAR_8_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_10_:%.+]] = llvm.insertvalue [[arg13_]], [[VAR_9_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.insertvalue [[arg14_]], [[VAR_10_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_13_:%.+]] = llvm.insertvalue [[arg5_]], [[VAR_12_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_14_:%.+]] = llvm.insertvalue [[arg6_]], [[VAR_13_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_15_:%.+]] = llvm.insertvalue [[arg7_]], [[VAR_14_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[arg8_]], [[VAR_15_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_17_:%.+]] = llvm.insertvalue [[arg9_]], [[VAR_16_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_18_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[arg0_]], [[VAR_18_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[arg1_]], [[VAR_19_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_21_:%.+]] = llvm.insertvalue [[arg2_]], [[VAR_20_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_22_:%.+]] = llvm.insertvalue [[arg3_]], [[VAR_21_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_23_:%.+]] = llvm.insertvalue [[arg4_]], [[VAR_22_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_27_:%.+]] = llvm.getelementptr [[VAR_26_]]{{.}}[[VAR_24_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.ptrtoint [[VAR_27_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_29_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_30_:%.+]] = llvm.add [[VAR_28_]], [[VAR_29_]] : i64
// CHECK:           [[VAR_31_:%.+]] = llvm.call @malloc([[VAR_30_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.ptrtoint [[VAR_31_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_34_:%.+]] = llvm.sub [[VAR_29_]], [[VAR_33_]] : i64
// CHECK:           [[VAR_35_:%.+]] = llvm.add [[VAR_32_]], [[VAR_34_]] : i64
// CHECK:           [[VAR_36_:%.+]] = llvm.urem [[VAR_35_]], [[VAR_29_]] : i64
// CHECK:           [[VAR_37_:%.+]] = llvm.sub [[VAR_35_]], [[VAR_36_]] : i64
// CHECK-DAG:       [[VAR_38_:%.+]] = llvm.inttoptr [[VAR_37_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_40_:%.+]] = llvm.insertvalue [[VAR_31_]], [[VAR_39_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_38_]], [[VAR_40_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_42_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_42_]], [[VAR_41_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_44_:%.+]] = llvm.insertvalue [[VAR_24_]], [[VAR_43_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_45_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_44_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_46_:%.+]] = llvm.extractvalue [[VAR_5_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_47_:%.+]] = llvm.getelementptr [[VAR_46_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_47_MEM_:%.+]] = llvm.load [[VAR_47_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.getelementptr [[VAR_46_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_49_MEM_:%.+]] = llvm.load [[VAR_49_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.getelementptr [[VAR_46_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_51_MEM_:%.+]] = llvm.load [[VAR_51_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.extractvalue [[VAR_23_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_54_:%.+]] = llvm.bitcast [[VAR_53_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_57_:%.+]] = llvm.alloca [[VAR_56_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_59_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_60_:%.+]] = llvm.bitcast [[VAR_57_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_58_]], [[VAR_59_]], [[VAR_60_]], [[LOAD_VAR_47_MEM_]], [[LOAD_VAR_49_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_61_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_62_:%.+]] = llvm.alloca [[VAR_61_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_63_:%.+]] = llvm.bitcast [[VAR_57_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_64_:%.+]] = llvm.bitcast [[VAR_62_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_63_]], [[VAR_64_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.alloca [[VAR_55_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_67_:%.+]] = llvm.bitcast [[VAR_62_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_67_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_69_:%.+]] = llvm.getelementptr [[VAR_66_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_57_]], [[VAR_69_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_70_:%.+]] = llvm.getelementptr [[VAR_66_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_62_]], [[VAR_70_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_71_:%.+]] = llvm.getelementptr [[VAR_66_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_68_]], [[VAR_71_]] : i64, !llvm.ptr
// CHECK:           [[VAR_72_:%.+]] = llvm.getelementptr [[VAR_66_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_54_]], [[VAR_72_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_73_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.getelementptr [[VAR_66_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_73_]], [[VAR_74_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.getelementptr [[VAR_66_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_76_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_76_]], [[VAR_75_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.getelementptr [[VAR_66_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_78_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_78_]], [[VAR_77_]] : f32, !llvm.ptr
// CHECK:           [[VAR_79_:%.+]] = llvm.extractvalue [[VAR_17_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.bitcast [[VAR_79_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_82_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.alloca [[VAR_82_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_85_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_86_:%.+]] = llvm.bitcast [[VAR_83_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_84_]], [[VAR_85_]], [[VAR_86_]], [[LOAD_VAR_49_MEM_]], [[LOAD_VAR_51_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_87_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_88_:%.+]] = llvm.alloca [[VAR_87_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_89_:%.+]] = llvm.bitcast [[VAR_83_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_90_:%.+]] = llvm.bitcast [[VAR_88_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_91_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_89_]], [[VAR_90_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_92_:%.+]] = llvm.alloca [[VAR_81_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.bitcast [[VAR_88_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_93_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.getelementptr [[VAR_92_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_83_]], [[VAR_95_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_96_:%.+]] = llvm.getelementptr [[VAR_92_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_88_]], [[VAR_96_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_97_:%.+]] = llvm.getelementptr [[VAR_92_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_94_]], [[VAR_97_]] : i64, !llvm.ptr
// CHECK:           [[VAR_98_:%.+]] = llvm.getelementptr [[VAR_92_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_80_]], [[VAR_98_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_99_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_100_:%.+]] = llvm.getelementptr [[VAR_92_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_99_]], [[VAR_100_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_101_:%.+]] = llvm.getelementptr [[VAR_92_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_102_]], [[VAR_101_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.getelementptr [[VAR_92_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_104_]], [[VAR_103_]] : f32, !llvm.ptr
// CHECK:           [[VAR_105_:%.+]] = llvm.extractvalue [[VAR_11_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.bitcast [[VAR_105_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_107_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.alloca [[VAR_108_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_110_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_111_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_112_:%.+]] = llvm.bitcast [[VAR_109_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_110_]], [[VAR_111_]], [[VAR_112_]], [[LOAD_VAR_51_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64) -> ()
// CHECK:           [[VAR_113_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_114_:%.+]] = llvm.alloca [[VAR_113_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_115_:%.+]] = llvm.bitcast [[VAR_109_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_116_:%.+]] = llvm.bitcast [[VAR_114_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_115_]], [[VAR_116_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_118_:%.+]] = llvm.alloca [[VAR_107_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_119_:%.+]] = llvm.bitcast [[VAR_114_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_119_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.getelementptr [[VAR_118_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_109_]], [[VAR_121_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_122_:%.+]] = llvm.getelementptr [[VAR_118_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_114_]], [[VAR_122_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_123_:%.+]] = llvm.getelementptr [[VAR_118_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_120_]], [[VAR_123_]] : i64, !llvm.ptr
// CHECK:           [[VAR_124_:%.+]] = llvm.getelementptr [[VAR_118_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_106_]], [[VAR_124_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_125_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_126_:%.+]] = llvm.getelementptr [[VAR_118_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_125_]], [[VAR_126_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_127_:%.+]] = llvm.getelementptr [[VAR_118_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_128_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_128_]], [[VAR_127_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_129_:%.+]] = llvm.getelementptr [[VAR_118_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_130_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_130_]], [[VAR_129_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_131_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_132_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_133_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_134_:%.+]] = llvm.extractvalue [[VAR_45_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_135_:%.+]] = llvm.bitcast [[VAR_134_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_136_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_137_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_138_:%.+]] = llvm.alloca [[VAR_137_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_139_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_140_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_141_:%.+]] = llvm.bitcast [[VAR_138_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_139_]], [[VAR_140_]], [[VAR_141_]], [[LOAD_VAR_47_MEM_]], [[LOAD_VAR_51_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_142_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_143_:%.+]] = llvm.alloca [[VAR_142_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_144_:%.+]] = llvm.bitcast [[VAR_138_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_145_:%.+]] = llvm.bitcast [[VAR_143_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_146_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_144_]], [[VAR_145_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_147_:%.+]] = llvm.alloca [[VAR_136_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_148_:%.+]] = llvm.bitcast [[VAR_143_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_149_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_148_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_150_:%.+]] = llvm.getelementptr [[VAR_147_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_138_]], [[VAR_150_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_151_:%.+]] = llvm.getelementptr [[VAR_147_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_143_]], [[VAR_151_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_152_:%.+]] = llvm.getelementptr [[VAR_147_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_149_]], [[VAR_152_]] : i64, !llvm.ptr
// CHECK:           [[VAR_153_:%.+]] = llvm.getelementptr [[VAR_147_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_135_]], [[VAR_153_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_154_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_155_:%.+]] = llvm.getelementptr [[VAR_147_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_154_]], [[VAR_155_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_156_:%.+]] = llvm.getelementptr [[VAR_147_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_157_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_157_]], [[VAR_156_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_158_:%.+]] = llvm.getelementptr [[VAR_147_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_159_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_159_]], [[VAR_158_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_160_:%.+]] = llvm.bitcast [[VAR_66_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_161_:%.+]] = llvm.bitcast [[VAR_92_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_162_:%.+]] = llvm.bitcast [[VAR_118_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_163_:%.+]] = llvm.bitcast [[VAR_147_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_164_:%.+]] = llvm.call @zdnnx_matmul_op([[VAR_160_]], [[VAR_161_]], [[VAR_162_]], [[VAR_131_]], [[VAR_163_]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK:           llvm.return [[VAR_45_]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_matmul_no_bcast_unstacked([[arg0_]]: !llvm.ptr, [[arg1_]]: !llvm.ptr, [[arg2_]]: !llvm.ptr, [[arg3_]]: !llvm.ptr, [[arg4_]]: !llvm.ptr) attributes {llvm.emit_c_interface} {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.load [[arg1_]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_1_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_2_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_3_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_4_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_5_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_6_1_:%.+]] = llvm.load [[arg2_]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_8_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_9_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_10_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_11_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_12_1_:%.+]] = llvm.load [[arg3_]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_14_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_15_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_16_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_17_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_18_1_:%.+]] = llvm.load [[arg4_]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_19_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_20_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_21_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_22_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_23_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_24_1_:%.+]] = llvm.call @test_matmul_no_bcast_unstacked([[VAR_1_1_]], [[VAR_2_1_]], [[VAR_3_1_]], [[VAR_4_1_]], [[VAR_5_1_]], [[VAR_7_1_]], [[VAR_8_1_]], [[VAR_9_1_]], [[VAR_10_1_]], [[VAR_11_1_]], [[VAR_13_1_]], [[VAR_14_1_]], [[VAR_15_1_]], [[VAR_16_1_]], [[VAR_17_1_]], [[VAR_19_1_]], [[VAR_20_1_]], [[VAR_21_1_]], [[VAR_22_1_]], [[VAR_23_1_]]) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           llvm.store [[VAR_24_1_]], [[arg0_]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_2_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_2_:%.+]] = llvm.bitcast [[VAR_0_2_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_2_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether the lowering of zlow.matmul calls the correct zDNN API or not.
func.func @test_matmul_no_bcast_stacked(%x: memref<2048xf16>,%y: memref<2048xf16>,%bias: memref<2048xf16>, %shape: memref<3xi64>) -> memref<2048xf16> {
  %res = memref.alloc() alignment = 4096 : memref<2048xf16>
  "zlow.matmul"(%x, %y, %bias, %shape, %res) {is_bcast1 = 0 : si64, is_bcast23 = 0 : si64, is_stacked = -1 : si64} : (memref<2048xf16>, memref<2048xf16>, memref<2048xf16>, memref<3xi64>, memref<2048xf16>) -> ()
  return %res : memref<2048xf16>

// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnnx_matmul_op(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_matmul_no_bcast_stacked([[arg0_:%.+]]: !llvm.ptr, [[arg1_:%.+]]: !llvm.ptr, [[arg2_:%.+]]: i64, [[arg3_:%.+]]: i64, [[arg4_:%.+]]: i64, [[arg5_:%.+]]: !llvm.ptr, [[arg6_:%.+]]: !llvm.ptr, [[arg7_:%.+]]: i64, [[arg8_:%.+]]: i64, [[arg9_:%.+]]: i64, [[arg10_:%.+]]: !llvm.ptr, [[arg11_:%.+]]: !llvm.ptr, [[arg12_:%.+]]: i64, [[arg13_:%.+]]: i64, [[arg14_:%.+]]: i64, [[arg15_:%.+]]: !llvm.ptr, [[arg16_:%.+]]: !llvm.ptr, [[arg17_:%.+]]: i64, [[arg18_:%.+]]: i64, [[arg19_:%.+]]: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface} {
// CHECK:           [[VAR_0_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_1_:%.+]] = llvm.insertvalue [[arg15_]], [[VAR_0_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_2_:%.+]] = llvm.insertvalue [[arg16_]], [[VAR_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_3_:%.+]] = llvm.insertvalue [[arg17_]], [[VAR_2_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_4_:%.+]] = llvm.insertvalue [[arg18_]], [[VAR_3_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.insertvalue [[arg19_]], [[VAR_4_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_7_:%.+]] = llvm.insertvalue [[arg10_]], [[VAR_6_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_8_:%.+]] = llvm.insertvalue [[arg11_]], [[VAR_7_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_9_:%.+]] = llvm.insertvalue [[arg12_]], [[VAR_8_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_10_:%.+]] = llvm.insertvalue [[arg13_]], [[VAR_9_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.insertvalue [[arg14_]], [[VAR_10_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_13_:%.+]] = llvm.insertvalue [[arg5_]], [[VAR_12_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_14_:%.+]] = llvm.insertvalue [[arg6_]], [[VAR_13_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_15_:%.+]] = llvm.insertvalue [[arg7_]], [[VAR_14_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[arg8_]], [[VAR_15_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_17_:%.+]] = llvm.insertvalue [[arg9_]], [[VAR_16_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_18_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[arg0_]], [[VAR_18_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[arg1_]], [[VAR_19_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_21_:%.+]] = llvm.insertvalue [[arg2_]], [[VAR_20_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_22_:%.+]] = llvm.insertvalue [[arg3_]], [[VAR_21_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_23_:%.+]] = llvm.insertvalue [[arg4_]], [[VAR_22_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_27_:%.+]] = llvm.getelementptr [[VAR_26_]]{{.}}[[VAR_24_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.ptrtoint [[VAR_27_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_29_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_30_:%.+]] = llvm.add [[VAR_28_]], [[VAR_29_]] : i64
// CHECK:           [[VAR_31_:%.+]] = llvm.call @malloc([[VAR_30_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.ptrtoint [[VAR_31_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_34_:%.+]] = llvm.sub [[VAR_29_]], [[VAR_33_]] : i64
// CHECK:           [[VAR_35_:%.+]] = llvm.add [[VAR_32_]], [[VAR_34_]] : i64
// CHECK:           [[VAR_36_:%.+]] = llvm.urem [[VAR_35_]], [[VAR_29_]] : i64
// CHECK:           [[VAR_37_:%.+]] = llvm.sub [[VAR_35_]], [[VAR_36_]] : i64
// CHECK-DAG:       [[VAR_38_:%.+]] = llvm.inttoptr [[VAR_37_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_40_:%.+]] = llvm.insertvalue [[VAR_31_]], [[VAR_39_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_38_]], [[VAR_40_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_42_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_42_]], [[VAR_41_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_44_:%.+]] = llvm.insertvalue [[VAR_24_]], [[VAR_43_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_45_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_44_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_46_:%.+]] = llvm.extractvalue [[VAR_5_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_47_:%.+]] = llvm.getelementptr [[VAR_46_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_47_MEM_:%.+]] = llvm.load [[VAR_47_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.getelementptr [[VAR_46_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_49_MEM_:%.+]] = llvm.load [[VAR_49_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.getelementptr [[VAR_46_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_51_MEM_:%.+]] = llvm.load [[VAR_51_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.getelementptr [[VAR_46_]][3] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_53_MEM_:%.+]] = llvm.load [[VAR_53_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.extractvalue [[VAR_23_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.bitcast [[VAR_55_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_57_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_59_:%.+]] = llvm.alloca [[VAR_58_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_60_:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:       [[VAR_61_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_62_:%.+]] = llvm.bitcast [[VAR_59_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_60_]], [[VAR_61_]], [[VAR_62_]], [[LOAD_VAR_47_MEM_]], [[LOAD_VAR_49_MEM_]], [[LOAD_VAR_51_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64) -> ()
// CHECK:           [[VAR_63_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_64_:%.+]] = llvm.alloca [[VAR_63_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.bitcast [[VAR_59_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_66_:%.+]] = llvm.bitcast [[VAR_64_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_67_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_65_]], [[VAR_66_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.alloca [[VAR_57_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_69_:%.+]] = llvm.bitcast [[VAR_64_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_70_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_69_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_59_]], [[VAR_71_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_72_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_64_]], [[VAR_72_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_73_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_70_]], [[VAR_73_]] : i64, !llvm.ptr
// CHECK:           [[VAR_74_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_56_]], [[VAR_74_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_76_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_75_]], [[VAR_76_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_78_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_78_]], [[VAR_77_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_79_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_80_]], [[VAR_79_]] : f32, !llvm.ptr
// CHECK:           [[VAR_81_:%.+]] = llvm.extractvalue [[VAR_17_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_82_:%.+]] = llvm.bitcast [[VAR_81_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_85_:%.+]] = llvm.alloca [[VAR_84_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_86_:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:       [[VAR_87_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_88_:%.+]] = llvm.bitcast [[VAR_85_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_86_]], [[VAR_87_]], [[VAR_88_]], [[LOAD_VAR_47_MEM_]], [[LOAD_VAR_51_MEM_]], [[LOAD_VAR_53_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64) -> ()
// CHECK:           [[VAR_89_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_90_:%.+]] = llvm.alloca [[VAR_89_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_91_:%.+]] = llvm.bitcast [[VAR_85_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_92_:%.+]] = llvm.bitcast [[VAR_90_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_91_]], [[VAR_92_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.alloca [[VAR_83_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.bitcast [[VAR_90_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_95_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_85_]], [[VAR_97_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_98_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_90_]], [[VAR_98_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_99_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_96_]], [[VAR_99_]] : i64, !llvm.ptr
// CHECK:           [[VAR_100_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_82_]], [[VAR_100_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_101_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_101_]], [[VAR_102_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_104_]], [[VAR_103_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_105_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_106_]], [[VAR_105_]] : f32, !llvm.ptr
// CHECK:           [[VAR_107_:%.+]] = llvm.extractvalue [[VAR_11_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.bitcast [[VAR_107_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_110_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_111_:%.+]] = llvm.alloca [[VAR_110_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_112_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       [[VAR_113_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_114_:%.+]] = llvm.bitcast [[VAR_111_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_112_]], [[VAR_113_]], [[VAR_114_]], [[LOAD_VAR_47_MEM_]], [[LOAD_VAR_53_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_115_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.alloca [[VAR_115_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.bitcast [[VAR_111_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_118_:%.+]] = llvm.bitcast [[VAR_116_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_119_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_117_]], [[VAR_118_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.alloca [[VAR_109_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.bitcast [[VAR_116_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_122_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_121_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_123_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_111_]], [[VAR_123_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_124_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_116_]], [[VAR_124_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_125_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_122_]], [[VAR_125_]] : i64, !llvm.ptr
// CHECK:           [[VAR_126_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_126_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_127_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_128_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_127_]], [[VAR_128_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_129_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_130_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_130_]], [[VAR_129_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_131_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_132_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_132_]], [[VAR_131_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_133_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_134_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_135_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_136_:%.+]] = llvm.extractvalue [[VAR_45_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_137_:%.+]] = llvm.bitcast [[VAR_136_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_138_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_139_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_140_:%.+]] = llvm.alloca [[VAR_139_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_141_:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:       [[VAR_142_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_143_:%.+]] = llvm.bitcast [[VAR_140_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_141_]], [[VAR_142_]], [[VAR_143_]], [[LOAD_VAR_47_MEM_]], [[LOAD_VAR_49_MEM_]], [[LOAD_VAR_53_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64) -> ()
// CHECK:           [[VAR_144_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_145_:%.+]] = llvm.alloca [[VAR_144_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_146_:%.+]] = llvm.bitcast [[VAR_140_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_147_:%.+]] = llvm.bitcast [[VAR_145_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_148_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_146_]], [[VAR_147_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_149_:%.+]] = llvm.alloca [[VAR_138_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_150_:%.+]] = llvm.bitcast [[VAR_145_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_151_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_150_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_152_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_140_]], [[VAR_152_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_153_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_145_]], [[VAR_153_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_154_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_151_]], [[VAR_154_]] : i64, !llvm.ptr
// CHECK:           [[VAR_155_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_137_]], [[VAR_155_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_156_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_157_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_156_]], [[VAR_157_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_158_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_159_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_159_]], [[VAR_158_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_160_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_161_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_161_]], [[VAR_160_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_162_:%.+]] = llvm.bitcast [[VAR_68_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_163_:%.+]] = llvm.bitcast [[VAR_94_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_164_:%.+]] = llvm.bitcast [[VAR_120_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_165_:%.+]] = llvm.bitcast [[VAR_149_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_166_:%.+]] = llvm.call @zdnnx_matmul_op([[VAR_162_]], [[VAR_163_]], [[VAR_164_]], [[VAR_133_]], [[VAR_165_]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK:           llvm.return [[VAR_45_]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_matmul_no_bcast_stacked([[arg0_]]: !llvm.ptr, [[arg1_]]: !llvm.ptr, [[arg2_]]: !llvm.ptr, [[arg3_]]: !llvm.ptr, [[arg4_]]: !llvm.ptr) attributes {llvm.emit_c_interface} {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.load [[arg1_]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_1_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_2_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_3_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_4_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_5_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_6_1_:%.+]] = llvm.load [[arg2_]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_8_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_9_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_10_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_11_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_12_1_:%.+]] = llvm.load [[arg3_]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_14_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_15_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_16_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_17_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_18_1_:%.+]] = llvm.load [[arg4_]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_19_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_20_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_21_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_22_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_23_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_24_1_:%.+]] = llvm.call @test_matmul_no_bcast_stacked([[VAR_1_1_]], [[VAR_2_1_]], [[VAR_3_1_]], [[VAR_4_1_]], [[VAR_5_1_]], [[VAR_7_1_]], [[VAR_8_1_]], [[VAR_9_1_]], [[VAR_10_1_]], [[VAR_11_1_]], [[VAR_13_1_]], [[VAR_14_1_]], [[VAR_15_1_]], [[VAR_16_1_]], [[VAR_17_1_]], [[VAR_19_1_]], [[VAR_20_1_]], [[VAR_21_1_]], [[VAR_22_1_]], [[VAR_23_1_]]) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           llvm.store [[VAR_24_1_]], [[arg0_]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_2_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_2_:%.+]] = llvm.bitcast [[VAR_0_2_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_2_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether the lowering of zlow.matmul calls the correct zDNN API or not.
func.func @test_matmul_bcast_stacked(%x: memref<2048xf16>,%y: memref<2048xf16>,%bias: memref<2048xf16>, %shape: memref<3xi64>) -> memref<2048xf16> {
  %res = memref.alloc() alignment = 4096 : memref<2048xf16>
  "zlow.matmul"(%x, %y, %bias, %shape, %res) {is_bcast1 = 0 : si64, is_bcast23 = -1 : si64, is_stacked = -1 : si64} : (memref<2048xf16>, memref<2048xf16>, memref<2048xf16>, memref<3xi64>, memref<2048xf16>) -> ()
  return %res : memref<2048xf16>

// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnnx_matmul_bcast_op(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_matmul_bcast_stacked([[arg0_:%.+]]: !llvm.ptr, [[arg1_:%.+]]: !llvm.ptr, [[arg2_:%.+]]: i64, [[arg3_:%.+]]: i64, [[arg4_:%.+]]: i64, [[arg5_:%.+]]: !llvm.ptr, [[arg6_:%.+]]: !llvm.ptr, [[arg7_:%.+]]: i64, [[arg8_:%.+]]: i64, [[arg9_:%.+]]: i64, [[arg10_:%.+]]: !llvm.ptr, [[arg11_:%.+]]: !llvm.ptr, [[arg12_:%.+]]: i64, [[arg13_:%.+]]: i64, [[arg14_:%.+]]: i64, [[arg15_:%.+]]: !llvm.ptr, [[arg16_:%.+]]: !llvm.ptr, [[arg17_:%.+]]: i64, [[arg18_:%.+]]: i64, [[arg19_:%.+]]: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface} {
// CHECK:           [[VAR_0_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_1_:%.+]] = llvm.insertvalue [[arg15_]], [[VAR_0_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_2_:%.+]] = llvm.insertvalue [[arg16_]], [[VAR_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_3_:%.+]] = llvm.insertvalue [[arg17_]], [[VAR_2_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_4_:%.+]] = llvm.insertvalue [[arg18_]], [[VAR_3_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.insertvalue [[arg19_]], [[VAR_4_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_7_:%.+]] = llvm.insertvalue [[arg10_]], [[VAR_6_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_8_:%.+]] = llvm.insertvalue [[arg11_]], [[VAR_7_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_9_:%.+]] = llvm.insertvalue [[arg12_]], [[VAR_8_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_10_:%.+]] = llvm.insertvalue [[arg13_]], [[VAR_9_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.insertvalue [[arg14_]], [[VAR_10_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_13_:%.+]] = llvm.insertvalue [[arg5_]], [[VAR_12_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_14_:%.+]] = llvm.insertvalue [[arg6_]], [[VAR_13_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_15_:%.+]] = llvm.insertvalue [[arg7_]], [[VAR_14_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[arg8_]], [[VAR_15_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_17_:%.+]] = llvm.insertvalue [[arg9_]], [[VAR_16_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_18_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[arg0_]], [[VAR_18_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[arg1_]], [[VAR_19_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_21_:%.+]] = llvm.insertvalue [[arg2_]], [[VAR_20_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_22_:%.+]] = llvm.insertvalue [[arg3_]], [[VAR_21_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_23_:%.+]] = llvm.insertvalue [[arg4_]], [[VAR_22_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_27_:%.+]] = llvm.getelementptr [[VAR_26_]]{{.}}[[VAR_24_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.ptrtoint [[VAR_27_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_29_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_30_:%.+]] = llvm.add [[VAR_28_]], [[VAR_29_]] : i64
// CHECK:           [[VAR_31_:%.+]] = llvm.call @malloc([[VAR_30_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.ptrtoint [[VAR_31_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_34_:%.+]] = llvm.sub [[VAR_29_]], [[VAR_33_]] : i64
// CHECK:           [[VAR_35_:%.+]] = llvm.add [[VAR_32_]], [[VAR_34_]] : i64
// CHECK:           [[VAR_36_:%.+]] = llvm.urem [[VAR_35_]], [[VAR_29_]] : i64
// CHECK:           [[VAR_37_:%.+]] = llvm.sub [[VAR_35_]], [[VAR_36_]] : i64
// CHECK-DAG:       [[VAR_38_:%.+]] = llvm.inttoptr [[VAR_37_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_40_:%.+]] = llvm.insertvalue [[VAR_31_]], [[VAR_39_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_38_]], [[VAR_40_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_42_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_42_]], [[VAR_41_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_44_:%.+]] = llvm.insertvalue [[VAR_24_]], [[VAR_43_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_45_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_44_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_46_:%.+]] = llvm.extractvalue [[VAR_5_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_47_:%.+]] = llvm.getelementptr [[VAR_46_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_47_MEM_:%.+]] = llvm.load [[VAR_47_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.getelementptr [[VAR_46_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_49_MEM_:%.+]] = llvm.load [[VAR_49_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.getelementptr [[VAR_46_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_51_MEM_:%.+]] = llvm.load [[VAR_51_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.getelementptr [[VAR_46_]][3] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_53_MEM_:%.+]] = llvm.load [[VAR_53_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.extractvalue [[VAR_23_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.bitcast [[VAR_55_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_57_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_59_:%.+]] = llvm.alloca [[VAR_58_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_60_:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:       [[VAR_61_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_62_:%.+]] = llvm.bitcast [[VAR_59_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_60_]], [[VAR_61_]], [[VAR_62_]], [[LOAD_VAR_47_MEM_]], [[LOAD_VAR_49_MEM_]], [[LOAD_VAR_51_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64) -> ()
// CHECK:           [[VAR_63_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_64_:%.+]] = llvm.alloca [[VAR_63_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.bitcast [[VAR_59_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_66_:%.+]] = llvm.bitcast [[VAR_64_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_67_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_65_]], [[VAR_66_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.alloca [[VAR_57_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_69_:%.+]] = llvm.bitcast [[VAR_64_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_70_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_69_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_59_]], [[VAR_71_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_72_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_64_]], [[VAR_72_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_73_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_70_]], [[VAR_73_]] : i64, !llvm.ptr
// CHECK:           [[VAR_74_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_56_]], [[VAR_74_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_76_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_75_]], [[VAR_76_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_78_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_78_]], [[VAR_77_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_79_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_80_]], [[VAR_79_]] : f32, !llvm.ptr
// CHECK:           [[VAR_81_:%.+]] = llvm.extractvalue [[VAR_17_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_82_:%.+]] = llvm.bitcast [[VAR_81_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_85_:%.+]] = llvm.alloca [[VAR_84_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_86_:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:       [[VAR_87_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_88_:%.+]] = llvm.bitcast [[VAR_85_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_86_]], [[VAR_87_]], [[VAR_88_]], [[LOAD_VAR_47_MEM_]], [[LOAD_VAR_51_MEM_]], [[LOAD_VAR_53_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64) -> ()
// CHECK:           [[VAR_89_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_90_:%.+]] = llvm.alloca [[VAR_89_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_91_:%.+]] = llvm.bitcast [[VAR_85_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_92_:%.+]] = llvm.bitcast [[VAR_90_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_91_]], [[VAR_92_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.alloca [[VAR_83_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.bitcast [[VAR_90_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_95_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_85_]], [[VAR_97_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_98_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_90_]], [[VAR_98_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_99_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_96_]], [[VAR_99_]] : i64, !llvm.ptr
// CHECK:           [[VAR_100_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_82_]], [[VAR_100_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_101_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_101_]], [[VAR_102_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_104_]], [[VAR_103_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_105_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_106_]], [[VAR_105_]] : f32, !llvm.ptr
// CHECK:           [[VAR_107_:%.+]] = llvm.extractvalue [[VAR_11_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.bitcast [[VAR_107_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_110_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_111_:%.+]] = llvm.alloca [[VAR_110_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_112_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       [[VAR_113_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_114_:%.+]] = llvm.bitcast [[VAR_111_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_112_]], [[VAR_113_]], [[VAR_114_]], [[LOAD_VAR_47_MEM_]], [[LOAD_VAR_53_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_115_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.alloca [[VAR_115_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.bitcast [[VAR_111_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_118_:%.+]] = llvm.bitcast [[VAR_116_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_119_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_117_]], [[VAR_118_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.alloca [[VAR_109_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.bitcast [[VAR_116_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_122_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_121_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_123_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_111_]], [[VAR_123_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_124_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_116_]], [[VAR_124_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_125_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_122_]], [[VAR_125_]] : i64, !llvm.ptr
// CHECK:           [[VAR_126_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_126_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_127_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_128_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_127_]], [[VAR_128_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_129_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_130_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_130_]], [[VAR_129_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_131_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_132_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_132_]], [[VAR_131_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_133_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_134_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_135_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_136_:%.+]] = llvm.extractvalue [[VAR_45_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_137_:%.+]] = llvm.bitcast [[VAR_136_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_138_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_139_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_140_:%.+]] = llvm.alloca [[VAR_139_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_141_:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:       [[VAR_142_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_143_:%.+]] = llvm.bitcast [[VAR_140_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_141_]], [[VAR_142_]], [[VAR_143_]], [[LOAD_VAR_47_MEM_]], [[LOAD_VAR_49_MEM_]], [[LOAD_VAR_53_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64) -> ()
// CHECK:           [[VAR_144_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_145_:%.+]] = llvm.alloca [[VAR_144_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_146_:%.+]] = llvm.bitcast [[VAR_140_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_147_:%.+]] = llvm.bitcast [[VAR_145_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_148_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_146_]], [[VAR_147_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_149_:%.+]] = llvm.alloca [[VAR_138_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_150_:%.+]] = llvm.bitcast [[VAR_145_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_151_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_150_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_152_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_140_]], [[VAR_152_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_153_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_145_]], [[VAR_153_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_154_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_151_]], [[VAR_154_]] : i64, !llvm.ptr
// CHECK:           [[VAR_155_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_137_]], [[VAR_155_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_156_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_157_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_156_]], [[VAR_157_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_158_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_159_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_159_]], [[VAR_158_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_160_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_161_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_161_]], [[VAR_160_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_162_:%.+]] = llvm.bitcast [[VAR_68_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_163_:%.+]] = llvm.bitcast [[VAR_94_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_164_:%.+]] = llvm.bitcast [[VAR_120_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_165_:%.+]] = llvm.bitcast [[VAR_149_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_166_:%.+]] = llvm.call @zdnnx_matmul_bcast_op([[VAR_162_]], [[VAR_163_]], [[VAR_164_]], [[VAR_133_]], [[VAR_165_]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK:           llvm.return [[VAR_45_]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_matmul_bcast_stacked([[arg0_]]: !llvm.ptr, [[arg1_]]: !llvm.ptr, [[arg2_]]: !llvm.ptr, [[arg3_]]: !llvm.ptr, [[arg4_]]: !llvm.ptr) attributes {llvm.emit_c_interface} {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.load [[arg1_]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_1_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_2_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_3_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_4_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_5_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_6_1_:%.+]] = llvm.load [[arg2_]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_8_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_9_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_10_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_11_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_12_1_:%.+]] = llvm.load [[arg3_]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_14_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_15_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_16_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_17_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_18_1_:%.+]] = llvm.load [[arg4_]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_19_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_20_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_21_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_22_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_23_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_24_1_:%.+]] = llvm.call @test_matmul_bcast_stacked([[VAR_1_1_]], [[VAR_2_1_]], [[VAR_3_1_]], [[VAR_4_1_]], [[VAR_5_1_]], [[VAR_7_1_]], [[VAR_8_1_]], [[VAR_9_1_]], [[VAR_10_1_]], [[VAR_11_1_]], [[VAR_13_1_]], [[VAR_14_1_]], [[VAR_15_1_]], [[VAR_16_1_]], [[VAR_17_1_]], [[VAR_19_1_]], [[VAR_20_1_]], [[VAR_21_1_]], [[VAR_22_1_]], [[VAR_23_1_]]) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           llvm.store [[VAR_24_1_]], [[arg0_]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_2_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_2_:%.+]] = llvm.bitcast [[VAR_0_2_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_2_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether the lowering of zlow.matmul calls the correct zDNN API or not.
func.func @test_matmul_bcast_unstacked(%x: memref<2048xf16>,%y: memref<2048xf16>,%bias: memref<2048xf16>, %shape: memref<3xi64>) -> memref<2048xf16> {
  %res = memref.alloc() alignment = 4096 : memref<2048xf16>
  "zlow.matmul"(%x, %y, %bias, %shape, %res) {is_bcast1 = 0 : si64, is_bcast23 = -1 : si64, is_stacked = 0 : si64} : (memref<2048xf16>, memref<2048xf16>, memref<2048xf16>, memref<3xi64>, memref<2048xf16>) -> ()
  return %res : memref<2048xf16>

// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnnx_matmul_bcast_op(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_matmul_bcast_unstacked([[arg0_:%.+]]: !llvm.ptr, [[arg1_:%.+]]: !llvm.ptr, [[arg2_:%.+]]: i64, [[arg3_:%.+]]: i64, [[arg4_:%.+]]: i64, [[arg5_:%.+]]: !llvm.ptr, [[arg6_:%.+]]: !llvm.ptr, [[arg7_:%.+]]: i64, [[arg8_:%.+]]: i64, [[arg9_:%.+]]: i64, [[arg10_:%.+]]: !llvm.ptr, [[arg11_:%.+]]: !llvm.ptr, [[arg12_:%.+]]: i64, [[arg13_:%.+]]: i64, [[arg14_:%.+]]: i64, [[arg15_:%.+]]: !llvm.ptr, [[arg16_:%.+]]: !llvm.ptr, [[arg17_:%.+]]: i64, [[arg18_:%.+]]: i64, [[arg19_:%.+]]: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface} {
// CHECK:           [[VAR_0_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_1_:%.+]] = llvm.insertvalue [[arg15_]], [[VAR_0_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_2_:%.+]] = llvm.insertvalue [[arg16_]], [[VAR_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_3_:%.+]] = llvm.insertvalue [[arg17_]], [[VAR_2_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_4_:%.+]] = llvm.insertvalue [[arg18_]], [[VAR_3_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.insertvalue [[arg19_]], [[VAR_4_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_7_:%.+]] = llvm.insertvalue [[arg10_]], [[VAR_6_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_8_:%.+]] = llvm.insertvalue [[arg11_]], [[VAR_7_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_9_:%.+]] = llvm.insertvalue [[arg12_]], [[VAR_8_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_10_:%.+]] = llvm.insertvalue [[arg13_]], [[VAR_9_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.insertvalue [[arg14_]], [[VAR_10_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_13_:%.+]] = llvm.insertvalue [[arg5_]], [[VAR_12_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_14_:%.+]] = llvm.insertvalue [[arg6_]], [[VAR_13_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_15_:%.+]] = llvm.insertvalue [[arg7_]], [[VAR_14_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[arg8_]], [[VAR_15_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_17_:%.+]] = llvm.insertvalue [[arg9_]], [[VAR_16_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_18_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[arg0_]], [[VAR_18_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[arg1_]], [[VAR_19_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_21_:%.+]] = llvm.insertvalue [[arg2_]], [[VAR_20_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_22_:%.+]] = llvm.insertvalue [[arg3_]], [[VAR_21_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_23_:%.+]] = llvm.insertvalue [[arg4_]], [[VAR_22_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_27_:%.+]] = llvm.getelementptr [[VAR_26_]]{{.}}[[VAR_24_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.ptrtoint [[VAR_27_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_29_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_30_:%.+]] = llvm.add [[VAR_28_]], [[VAR_29_]] : i64
// CHECK:           [[VAR_31_:%.+]] = llvm.call @malloc([[VAR_30_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.ptrtoint [[VAR_31_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_34_:%.+]] = llvm.sub [[VAR_29_]], [[VAR_33_]] : i64
// CHECK:           [[VAR_35_:%.+]] = llvm.add [[VAR_32_]], [[VAR_34_]] : i64
// CHECK:           [[VAR_36_:%.+]] = llvm.urem [[VAR_35_]], [[VAR_29_]] : i64
// CHECK:           [[VAR_37_:%.+]] = llvm.sub [[VAR_35_]], [[VAR_36_]] : i64
// CHECK-DAG:       [[VAR_38_:%.+]] = llvm.inttoptr [[VAR_37_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_40_:%.+]] = llvm.insertvalue [[VAR_31_]], [[VAR_39_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_38_]], [[VAR_40_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_42_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_42_]], [[VAR_41_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_44_:%.+]] = llvm.insertvalue [[VAR_24_]], [[VAR_43_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_45_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_44_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_46_:%.+]] = llvm.extractvalue [[VAR_5_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_47_:%.+]] = llvm.getelementptr [[VAR_46_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_47_MEM_:%.+]] = llvm.load [[VAR_47_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.getelementptr [[VAR_46_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_49_MEM_:%.+]] = llvm.load [[VAR_49_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_51_:%.+]] = llvm.getelementptr [[VAR_46_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_51_MEM_:%.+]] = llvm.load [[VAR_51_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.getelementptr [[VAR_46_]][3] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_53_MEM_:%.+]] = llvm.load [[VAR_53_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.extractvalue [[VAR_23_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.bitcast [[VAR_55_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_57_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_59_:%.+]] = llvm.alloca [[VAR_58_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_60_:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:       [[VAR_61_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_62_:%.+]] = llvm.bitcast [[VAR_59_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_60_]], [[VAR_61_]], [[VAR_62_]], [[LOAD_VAR_47_MEM_]], [[LOAD_VAR_49_MEM_]], [[LOAD_VAR_51_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64) -> ()
// CHECK:           [[VAR_63_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_64_:%.+]] = llvm.alloca [[VAR_63_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.bitcast [[VAR_59_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_66_:%.+]] = llvm.bitcast [[VAR_64_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_67_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_65_]], [[VAR_66_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.alloca [[VAR_57_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_69_:%.+]] = llvm.bitcast [[VAR_64_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_70_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_69_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_59_]], [[VAR_71_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_72_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_64_]], [[VAR_72_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_73_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_70_]], [[VAR_73_]] : i64, !llvm.ptr
// CHECK:           [[VAR_74_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_56_]], [[VAR_74_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_76_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_75_]], [[VAR_76_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_78_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_78_]], [[VAR_77_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_79_:%.+]] = llvm.getelementptr [[VAR_68_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_80_]], [[VAR_79_]] : f32, !llvm.ptr
// CHECK:           [[VAR_81_:%.+]] = llvm.extractvalue [[VAR_17_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_82_:%.+]] = llvm.bitcast [[VAR_81_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_85_:%.+]] = llvm.alloca [[VAR_84_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_86_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_87_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_88_:%.+]] = llvm.bitcast [[VAR_85_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_86_]], [[VAR_87_]], [[VAR_88_]], [[LOAD_VAR_51_MEM_]], [[LOAD_VAR_53_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64) -> ()
// CHECK:           [[VAR_89_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_90_:%.+]] = llvm.alloca [[VAR_89_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_91_:%.+]] = llvm.bitcast [[VAR_85_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_92_:%.+]] = llvm.bitcast [[VAR_90_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_91_]], [[VAR_92_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.alloca [[VAR_83_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.bitcast [[VAR_90_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_95_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_85_]], [[VAR_97_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_98_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_90_]], [[VAR_98_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_99_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_96_]], [[VAR_99_]] : i64, !llvm.ptr
// CHECK:           [[VAR_100_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_82_]], [[VAR_100_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_101_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_101_]], [[VAR_102_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_104_]], [[VAR_103_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_105_:%.+]] = llvm.getelementptr [[VAR_94_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_106_]], [[VAR_105_]] : f32, !llvm.ptr
// CHECK:           [[VAR_107_:%.+]] = llvm.extractvalue [[VAR_11_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.bitcast [[VAR_107_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_110_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_111_:%.+]] = llvm.alloca [[VAR_110_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_112_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_113_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_114_:%.+]] = llvm.bitcast [[VAR_111_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_112_]], [[VAR_113_]], [[VAR_114_]], [[LOAD_VAR_53_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64) -> ()
// CHECK:           [[VAR_115_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.alloca [[VAR_115_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.bitcast [[VAR_111_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_118_:%.+]] = llvm.bitcast [[VAR_116_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_119_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_117_]], [[VAR_118_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.alloca [[VAR_109_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.bitcast [[VAR_116_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_122_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_121_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_123_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_111_]], [[VAR_123_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_124_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_116_]], [[VAR_124_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_125_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_122_]], [[VAR_125_]] : i64, !llvm.ptr
// CHECK:           [[VAR_126_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_126_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_127_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_128_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_127_]], [[VAR_128_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_129_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_130_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_130_]], [[VAR_129_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_131_:%.+]] = llvm.getelementptr [[VAR_120_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_132_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_132_]], [[VAR_131_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_133_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_134_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_135_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_136_:%.+]] = llvm.extractvalue [[VAR_45_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_137_:%.+]] = llvm.bitcast [[VAR_136_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_138_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_139_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_140_:%.+]] = llvm.alloca [[VAR_139_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_141_:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:       [[VAR_142_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_143_:%.+]] = llvm.bitcast [[VAR_140_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_141_]], [[VAR_142_]], [[VAR_143_]], [[LOAD_VAR_47_MEM_]], [[LOAD_VAR_49_MEM_]], [[LOAD_VAR_53_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64) -> ()
// CHECK:           [[VAR_144_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_145_:%.+]] = llvm.alloca [[VAR_144_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_146_:%.+]] = llvm.bitcast [[VAR_140_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_147_:%.+]] = llvm.bitcast [[VAR_145_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_148_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_146_]], [[VAR_147_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_149_:%.+]] = llvm.alloca [[VAR_138_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_150_:%.+]] = llvm.bitcast [[VAR_145_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_151_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_150_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_152_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_140_]], [[VAR_152_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_153_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_145_]], [[VAR_153_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_154_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_151_]], [[VAR_154_]] : i64, !llvm.ptr
// CHECK:           [[VAR_155_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_137_]], [[VAR_155_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_156_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_157_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_156_]], [[VAR_157_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_158_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_159_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_159_]], [[VAR_158_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_160_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_161_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_161_]], [[VAR_160_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_162_:%.+]] = llvm.bitcast [[VAR_68_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_163_:%.+]] = llvm.bitcast [[VAR_94_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_164_:%.+]] = llvm.bitcast [[VAR_120_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_165_:%.+]] = llvm.bitcast [[VAR_149_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_166_:%.+]] = llvm.call @zdnnx_matmul_bcast_op([[VAR_162_]], [[VAR_163_]], [[VAR_164_]], [[VAR_133_]], [[VAR_165_]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK:           llvm.return [[VAR_45_]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_matmul_bcast_unstacked([[arg0_]]: !llvm.ptr, [[arg1_]]: !llvm.ptr, [[arg2_]]: !llvm.ptr, [[arg3_]]: !llvm.ptr, [[arg4_]]: !llvm.ptr) attributes {llvm.emit_c_interface} {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.load [[arg1_]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_1_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_2_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_3_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_4_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_5_1_:%.+]] = llvm.extractvalue [[VAR_0_1_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_6_1_:%.+]] = llvm.load [[arg2_]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_8_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_9_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_10_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_11_1_:%.+]] = llvm.extractvalue [[VAR_6_1_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_12_1_:%.+]] = llvm.load [[arg3_]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_14_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_15_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_16_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_17_1_:%.+]] = llvm.extractvalue [[VAR_12_1_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_18_1_:%.+]] = llvm.load [[arg4_]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_19_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_20_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_21_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_22_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_23_1_:%.+]] = llvm.extractvalue [[VAR_18_1_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_24_1_:%.+]] = llvm.call @test_matmul_bcast_unstacked([[VAR_1_1_]], [[VAR_2_1_]], [[VAR_3_1_]], [[VAR_4_1_]], [[VAR_5_1_]], [[VAR_7_1_]], [[VAR_8_1_]], [[VAR_9_1_]], [[VAR_10_1_]], [[VAR_11_1_]], [[VAR_13_1_]], [[VAR_14_1_]], [[VAR_15_1_]], [[VAR_16_1_]], [[VAR_17_1_]], [[VAR_19_1_]], [[VAR_20_1_]], [[VAR_21_1_]], [[VAR_22_1_]], [[VAR_23_1_]]) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           llvm.store [[VAR_24_1_]], [[arg0_]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_2_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_2_:%.+]] = llvm.bitcast [[VAR_0_2_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_2_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether conv2d calls the correct zDNN API or not.
func.func @test_call_zdnn_cond2d() -> () {
  %input = memref.alloc() alignment = 4096 : memref<2048xf16>
  %kernel = memref.alloc() alignment = 4096 : memref<2048xf16>
  %bias = memref.alloc() alignment = 4096 : memref<2048xf16>
  %output = memref.alloc() alignment = 4096 : memref<2048xf16>
  %shape = memref.alloc() : memref<7xi64>
  "zlow.conv2d"(%input, %kernel, %bias, %shape, %output) {kernel_shape = [5, 5], strides = [2, 2], padding_type = "SAME_PADDING", act_func = "ACT_NONE" } : (memref<2048xf16>, memref<2048xf16>, memref<2048xf16>, memref<7xi64>, memref<2048xf16>)-> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnn_conv2d(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_cond2d() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_3_:%.+]] = llvm.getelementptr [[VAR_2_]]{{.}}[[VAR_0_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.ptrtoint [[VAR_3_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_6_:%.+]] = llvm.add [[VAR_4_]], [[VAR_5_]] : i64
// CHECK:           [[VAR_7_:%.+]] = llvm.call @malloc([[VAR_6_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.ptrtoint [[VAR_7_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_9_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_10_:%.+]] = llvm.sub [[VAR_5_]], [[VAR_9_]] : i64
// CHECK:           [[VAR_11_:%.+]] = llvm.add [[VAR_8_]], [[VAR_10_]] : i64
// CHECK:           [[VAR_12_:%.+]] = llvm.urem [[VAR_11_]], [[VAR_5_]] : i64
// CHECK:           [[VAR_13_:%.+]] = llvm.sub [[VAR_11_]], [[VAR_12_]] : i64
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.inttoptr [[VAR_13_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_7_]], [[VAR_15_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_14_]], [[VAR_16_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_18_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_18_]], [[VAR_17_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_19_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_22_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_23_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_25_:%.+]] = llvm.getelementptr [[VAR_24_]]{{.}}[[VAR_22_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.ptrtoint [[VAR_25_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_28_:%.+]] = llvm.add [[VAR_26_]], [[VAR_27_]] : i64
// CHECK:           [[VAR_29_:%.+]] = llvm.call @malloc([[VAR_28_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.ptrtoint [[VAR_29_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_32_:%.+]] = llvm.sub [[VAR_27_]], [[VAR_31_]] : i64
// CHECK:           [[VAR_33_:%.+]] = llvm.add [[VAR_30_]], [[VAR_32_]] : i64
// CHECK:           [[VAR_34_:%.+]] = llvm.urem [[VAR_33_]], [[VAR_27_]] : i64
// CHECK:           [[VAR_35_:%.+]] = llvm.sub [[VAR_33_]], [[VAR_34_]] : i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.inttoptr [[VAR_35_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_29_]], [[VAR_37_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_38_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_39_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_42_:%.+]] = llvm.insertvalue [[VAR_22_]], [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_23_]], [[VAR_42_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_44_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_45_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_46_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_47_:%.+]] = llvm.getelementptr [[VAR_46_]]{{.}}[[VAR_44_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_48_:%.+]] = llvm.ptrtoint [[VAR_47_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_50_:%.+]] = llvm.add [[VAR_48_]], [[VAR_49_]] : i64
// CHECK:           [[VAR_51_:%.+]] = llvm.call @malloc([[VAR_50_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.ptrtoint [[VAR_51_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_54_:%.+]] = llvm.sub [[VAR_49_]], [[VAR_53_]] : i64
// CHECK:           [[VAR_55_:%.+]] = llvm.add [[VAR_52_]], [[VAR_54_]] : i64
// CHECK:           [[VAR_56_:%.+]] = llvm.urem [[VAR_55_]], [[VAR_49_]] : i64
// CHECK:           [[VAR_57_:%.+]] = llvm.sub [[VAR_55_]], [[VAR_56_]] : i64
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.inttoptr [[VAR_57_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_59_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_60_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_59_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_61_:%.+]] = llvm.insertvalue [[VAR_58_]], [[VAR_60_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_62_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_63_:%.+]] = llvm.insertvalue [[VAR_62_]], [[VAR_61_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_64_:%.+]] = llvm.insertvalue [[VAR_44_]], [[VAR_63_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.insertvalue [[VAR_45_]], [[VAR_64_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_67_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_69_:%.+]] = llvm.getelementptr [[VAR_68_]]{{.}}[[VAR_66_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_70_:%.+]] = llvm.ptrtoint [[VAR_69_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_72_:%.+]] = llvm.add [[VAR_70_]], [[VAR_71_]] : i64
// CHECK:           [[VAR_73_:%.+]] = llvm.call @malloc([[VAR_72_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.ptrtoint [[VAR_73_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_76_:%.+]] = llvm.sub [[VAR_71_]], [[VAR_75_]] : i64
// CHECK:           [[VAR_77_:%.+]] = llvm.add [[VAR_74_]], [[VAR_76_]] : i64
// CHECK:           [[VAR_78_:%.+]] = llvm.urem [[VAR_77_]], [[VAR_71_]] : i64
// CHECK:           [[VAR_79_:%.+]] = llvm.sub [[VAR_77_]], [[VAR_78_]] : i64
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.inttoptr [[VAR_79_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_82_:%.+]] = llvm.insertvalue [[VAR_73_]], [[VAR_81_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.insertvalue [[VAR_80_]], [[VAR_82_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_85_:%.+]] = llvm.insertvalue [[VAR_84_]], [[VAR_83_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_86_:%.+]] = llvm.insertvalue [[VAR_66_]], [[VAR_85_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_87_:%.+]] = llvm.insertvalue [[VAR_67_]], [[VAR_86_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_88_:%.+]] = llvm.mlir.constant(7 : index) : i64
// CHECK-DAG:       [[VAR_89_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_90_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_91_:%.+]] = llvm.getelementptr [[VAR_90_]]{{.}}[[VAR_88_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_92_:%.+]] = llvm.ptrtoint [[VAR_91_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.call @malloc([[VAR_92_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_95_:%.+]] = llvm.insertvalue [[VAR_93_]], [[VAR_94_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.insertvalue [[VAR_93_]], [[VAR_95_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_98_:%.+]] = llvm.insertvalue [[VAR_97_]], [[VAR_96_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_99_:%.+]] = llvm.insertvalue [[VAR_88_]], [[VAR_98_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_100_:%.+]] = llvm.insertvalue [[VAR_89_]], [[VAR_99_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_101_:%.+]] = llvm.extractvalue [[VAR_100_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_102_:%.+]] = llvm.getelementptr [[VAR_101_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_102_MEM_:%.+]] = llvm.load [[VAR_102_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.getelementptr [[VAR_101_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_104_MEM_:%.+]] = llvm.load [[VAR_104_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.getelementptr [[VAR_101_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_106_MEM_:%.+]] = llvm.load [[VAR_106_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.getelementptr [[VAR_101_]][3] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_108_MEM_:%.+]] = llvm.load [[VAR_108_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_110_:%.+]] = llvm.getelementptr [[VAR_101_]][4] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_110_MEM_:%.+]] = llvm.load [[VAR_110_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_112_:%.+]] = llvm.getelementptr [[VAR_101_]][5] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_112_MEM_:%.+]] = llvm.load [[VAR_112_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_114_:%.+]] = llvm.getelementptr [[VAR_101_]][6] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_114_MEM_:%.+]] = llvm.load [[VAR_114_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.mlir.constant(5 : i64) : i64
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.mlir.constant(5 : i64) : i64
// CHECK-DAG:       [[VAR_118_:%.+]] = llvm.extractvalue [[VAR_21_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_119_:%.+]] = llvm.bitcast [[VAR_118_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_122_:%.+]] = llvm.alloca [[VAR_121_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_123_:%.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-DAG:       [[VAR_124_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_125_:%.+]] = llvm.bitcast [[VAR_122_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_123_]], [[VAR_124_]], [[VAR_125_]], [[LOAD_VAR_102_MEM_]], [[LOAD_VAR_106_MEM_]], [[LOAD_VAR_108_MEM_]], [[LOAD_VAR_104_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64, i64) -> ()
// CHECK:           [[VAR_126_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_127_:%.+]] = llvm.alloca [[VAR_126_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_128_:%.+]] = llvm.bitcast [[VAR_122_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_129_:%.+]] = llvm.bitcast [[VAR_127_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_130_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_128_]], [[VAR_129_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_131_:%.+]] = llvm.alloca [[VAR_120_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_132_:%.+]] = llvm.bitcast [[VAR_127_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_133_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_132_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_134_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_122_]], [[VAR_134_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_135_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_127_]], [[VAR_135_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_136_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_133_]], [[VAR_136_]] : i64, !llvm.ptr
// CHECK:           [[VAR_137_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_119_]], [[VAR_137_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_138_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_139_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_138_]], [[VAR_139_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_140_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_141_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_141_]], [[VAR_140_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_142_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_143_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_143_]], [[VAR_142_]] : f32, !llvm.ptr
// CHECK:           [[VAR_144_:%.+]] = llvm.extractvalue [[VAR_43_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_145_:%.+]] = llvm.bitcast [[VAR_144_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_146_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_147_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_148_:%.+]] = llvm.alloca [[VAR_147_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_149_:%.+]] = llvm.mlir.constant(11 : i64) : i64
// CHECK-DAG:       [[VAR_150_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_151_:%.+]] = llvm.bitcast [[VAR_148_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_149_]], [[VAR_150_]], [[VAR_151_]], [[VAR_116_]], [[VAR_117_]], [[LOAD_VAR_104_MEM_]], [[LOAD_VAR_110_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64, i64) -> ()
// CHECK:           [[VAR_152_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_153_:%.+]] = llvm.alloca [[VAR_152_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_154_:%.+]] = llvm.bitcast [[VAR_148_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_155_:%.+]] = llvm.bitcast [[VAR_153_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_156_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_154_]], [[VAR_155_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_157_:%.+]] = llvm.alloca [[VAR_146_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_158_:%.+]] = llvm.bitcast [[VAR_153_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_159_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_158_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_160_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_148_]], [[VAR_160_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_161_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_153_]], [[VAR_161_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_162_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_159_]], [[VAR_162_]] : i64, !llvm.ptr
// CHECK:           [[VAR_163_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_145_]], [[VAR_163_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_164_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_165_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_164_]], [[VAR_165_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_166_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_167_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_167_]], [[VAR_166_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_168_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_169_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_169_]], [[VAR_168_]] : f32, !llvm.ptr
// CHECK:           [[VAR_170_:%.+]] = llvm.extractvalue [[VAR_65_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_171_:%.+]] = llvm.bitcast [[VAR_170_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_172_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_173_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_174_:%.+]] = llvm.alloca [[VAR_173_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_175_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_176_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_177_:%.+]] = llvm.bitcast [[VAR_174_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_175_]], [[VAR_176_]], [[VAR_177_]], [[LOAD_VAR_110_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64) -> ()
// CHECK:           [[VAR_178_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_179_:%.+]] = llvm.alloca [[VAR_178_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_180_:%.+]] = llvm.bitcast [[VAR_174_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_181_:%.+]] = llvm.bitcast [[VAR_179_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_182_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_180_]], [[VAR_181_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_183_:%.+]] = llvm.alloca [[VAR_172_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_184_:%.+]] = llvm.bitcast [[VAR_179_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_185_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_184_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_186_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_174_]], [[VAR_186_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_187_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_179_]], [[VAR_187_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_188_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_185_]], [[VAR_188_]] : i64, !llvm.ptr
// CHECK:           [[VAR_189_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_171_]], [[VAR_189_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_190_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_191_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_190_]], [[VAR_191_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_192_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_193_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_193_]], [[VAR_192_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_194_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_195_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_195_]], [[VAR_194_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_196_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_197_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       [[VAR_198_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       [[VAR_199_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_200_:%.+]] = llvm.extractvalue [[VAR_87_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_201_:%.+]] = llvm.bitcast [[VAR_200_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_202_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_203_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_204_:%.+]] = llvm.alloca [[VAR_203_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_205_:%.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-DAG:       [[VAR_206_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_207_:%.+]] = llvm.bitcast [[VAR_204_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_205_]], [[VAR_206_]], [[VAR_207_]], [[LOAD_VAR_102_MEM_]], [[LOAD_VAR_112_MEM_]], [[LOAD_VAR_114_MEM_]], [[LOAD_VAR_110_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64, i64) -> ()
// CHECK:           [[VAR_208_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_209_:%.+]] = llvm.alloca [[VAR_208_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_210_:%.+]] = llvm.bitcast [[VAR_204_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_211_:%.+]] = llvm.bitcast [[VAR_209_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_212_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_210_]], [[VAR_211_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_213_:%.+]] = llvm.alloca [[VAR_202_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_214_:%.+]] = llvm.bitcast [[VAR_209_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_215_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_214_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_216_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_204_]], [[VAR_216_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_217_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_209_]], [[VAR_217_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_218_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_215_]], [[VAR_218_]] : i64, !llvm.ptr
// CHECK:           [[VAR_219_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_201_]], [[VAR_219_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_220_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_221_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_220_]], [[VAR_221_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_222_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_223_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_223_]], [[VAR_222_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_224_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_225_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_225_]], [[VAR_224_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_226_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:       [[VAR_227_:%.+]] = llvm.bitcast [[VAR_131_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_228_:%.+]] = llvm.bitcast [[VAR_157_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_229_:%.+]] = llvm.bitcast [[VAR_183_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_230_:%.+]] = llvm.bitcast [[VAR_213_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_231_:%.+]] = llvm.call @zdnn_conv2d([[VAR_227_]], [[VAR_228_]], [[VAR_229_]], [[VAR_196_]], [[VAR_197_]], [[VAR_198_]], [[VAR_199_]], [[VAR_226_]], [[VAR_230_]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_cond2d() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_cond2d() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether conv2d calls the correct zDNN API or not.
func.func @test_call_zdnn_cond2d_valid_padding() -> () {
  %input = memref.alloc() alignment = 4096 : memref<2048xf16>
  %kernel = memref.alloc() alignment = 4096 : memref<2048xf16>
  %bias = memref.alloc() alignment = 4096 : memref<2048xf16>
  %output = memref.alloc() alignment = 4096 : memref<2048xf16>
  %shape = memref.alloc() : memref<7xi64>
  "zlow.conv2d"(%input, %kernel, %bias, %shape, %output) {kernel_shape = [5, 5], strides = [2, 2], padding_type = "VALID_PADDING", act_func = "ACT_NONE" } : (memref<2048xf16>, memref<2048xf16>, memref<2048xf16>, memref<7xi64>, memref<2048xf16>)-> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnn_conv2d(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_cond2d_valid_padding() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_3_:%.+]] = llvm.getelementptr [[VAR_2_]]{{.}}[[VAR_0_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.ptrtoint [[VAR_3_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_6_:%.+]] = llvm.add [[VAR_4_]], [[VAR_5_]] : i64
// CHECK:           [[VAR_7_:%.+]] = llvm.call @malloc([[VAR_6_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.ptrtoint [[VAR_7_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_9_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_10_:%.+]] = llvm.sub [[VAR_5_]], [[VAR_9_]] : i64
// CHECK:           [[VAR_11_:%.+]] = llvm.add [[VAR_8_]], [[VAR_10_]] : i64
// CHECK:           [[VAR_12_:%.+]] = llvm.urem [[VAR_11_]], [[VAR_5_]] : i64
// CHECK:           [[VAR_13_:%.+]] = llvm.sub [[VAR_11_]], [[VAR_12_]] : i64
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.inttoptr [[VAR_13_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_7_]], [[VAR_15_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_14_]], [[VAR_16_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_18_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_18_]], [[VAR_17_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_19_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_22_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_23_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_25_:%.+]] = llvm.getelementptr [[VAR_24_]]{{.}}[[VAR_22_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.ptrtoint [[VAR_25_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_28_:%.+]] = llvm.add [[VAR_26_]], [[VAR_27_]] : i64
// CHECK:           [[VAR_29_:%.+]] = llvm.call @malloc([[VAR_28_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.ptrtoint [[VAR_29_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_32_:%.+]] = llvm.sub [[VAR_27_]], [[VAR_31_]] : i64
// CHECK:           [[VAR_33_:%.+]] = llvm.add [[VAR_30_]], [[VAR_32_]] : i64
// CHECK:           [[VAR_34_:%.+]] = llvm.urem [[VAR_33_]], [[VAR_27_]] : i64
// CHECK:           [[VAR_35_:%.+]] = llvm.sub [[VAR_33_]], [[VAR_34_]] : i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.inttoptr [[VAR_35_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_29_]], [[VAR_37_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_38_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_39_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_42_:%.+]] = llvm.insertvalue [[VAR_22_]], [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_23_]], [[VAR_42_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_44_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_45_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_46_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_47_:%.+]] = llvm.getelementptr [[VAR_46_]]{{.}}[[VAR_44_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_48_:%.+]] = llvm.ptrtoint [[VAR_47_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_50_:%.+]] = llvm.add [[VAR_48_]], [[VAR_49_]] : i64
// CHECK:           [[VAR_51_:%.+]] = llvm.call @malloc([[VAR_50_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.ptrtoint [[VAR_51_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_54_:%.+]] = llvm.sub [[VAR_49_]], [[VAR_53_]] : i64
// CHECK:           [[VAR_55_:%.+]] = llvm.add [[VAR_52_]], [[VAR_54_]] : i64
// CHECK:           [[VAR_56_:%.+]] = llvm.urem [[VAR_55_]], [[VAR_49_]] : i64
// CHECK:           [[VAR_57_:%.+]] = llvm.sub [[VAR_55_]], [[VAR_56_]] : i64
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.inttoptr [[VAR_57_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_59_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_60_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_59_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_61_:%.+]] = llvm.insertvalue [[VAR_58_]], [[VAR_60_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_62_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_63_:%.+]] = llvm.insertvalue [[VAR_62_]], [[VAR_61_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_64_:%.+]] = llvm.insertvalue [[VAR_44_]], [[VAR_63_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.insertvalue [[VAR_45_]], [[VAR_64_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_67_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_69_:%.+]] = llvm.getelementptr [[VAR_68_]]{{.}}[[VAR_66_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_70_:%.+]] = llvm.ptrtoint [[VAR_69_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_72_:%.+]] = llvm.add [[VAR_70_]], [[VAR_71_]] : i64
// CHECK:           [[VAR_73_:%.+]] = llvm.call @malloc([[VAR_72_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.ptrtoint [[VAR_73_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_76_:%.+]] = llvm.sub [[VAR_71_]], [[VAR_75_]] : i64
// CHECK:           [[VAR_77_:%.+]] = llvm.add [[VAR_74_]], [[VAR_76_]] : i64
// CHECK:           [[VAR_78_:%.+]] = llvm.urem [[VAR_77_]], [[VAR_71_]] : i64
// CHECK:           [[VAR_79_:%.+]] = llvm.sub [[VAR_77_]], [[VAR_78_]] : i64
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.inttoptr [[VAR_79_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_82_:%.+]] = llvm.insertvalue [[VAR_73_]], [[VAR_81_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.insertvalue [[VAR_80_]], [[VAR_82_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_85_:%.+]] = llvm.insertvalue [[VAR_84_]], [[VAR_83_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_86_:%.+]] = llvm.insertvalue [[VAR_66_]], [[VAR_85_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_87_:%.+]] = llvm.insertvalue [[VAR_67_]], [[VAR_86_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_88_:%.+]] = llvm.mlir.constant(7 : index) : i64
// CHECK-DAG:       [[VAR_89_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_90_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_91_:%.+]] = llvm.getelementptr [[VAR_90_]]{{.}}[[VAR_88_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_92_:%.+]] = llvm.ptrtoint [[VAR_91_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.call @malloc([[VAR_92_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_95_:%.+]] = llvm.insertvalue [[VAR_93_]], [[VAR_94_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.insertvalue [[VAR_93_]], [[VAR_95_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_98_:%.+]] = llvm.insertvalue [[VAR_97_]], [[VAR_96_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_99_:%.+]] = llvm.insertvalue [[VAR_88_]], [[VAR_98_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_100_:%.+]] = llvm.insertvalue [[VAR_89_]], [[VAR_99_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_101_:%.+]] = llvm.extractvalue [[VAR_100_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_102_:%.+]] = llvm.getelementptr [[VAR_101_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_102_MEM_:%.+]] = llvm.load [[VAR_102_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.getelementptr [[VAR_101_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_104_MEM_:%.+]] = llvm.load [[VAR_104_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.getelementptr [[VAR_101_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_106_MEM_:%.+]] = llvm.load [[VAR_106_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.getelementptr [[VAR_101_]][3] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_108_MEM_:%.+]] = llvm.load [[VAR_108_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_110_:%.+]] = llvm.getelementptr [[VAR_101_]][4] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_110_MEM_:%.+]] = llvm.load [[VAR_110_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_112_:%.+]] = llvm.getelementptr [[VAR_101_]][5] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_112_MEM_:%.+]] = llvm.load [[VAR_112_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_114_:%.+]] = llvm.getelementptr [[VAR_101_]][6] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_114_MEM_:%.+]] = llvm.load [[VAR_114_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.mlir.constant(5 : i64) : i64
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.mlir.constant(5 : i64) : i64
// CHECK-DAG:       [[VAR_118_:%.+]] = llvm.extractvalue [[VAR_21_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_119_:%.+]] = llvm.bitcast [[VAR_118_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_122_:%.+]] = llvm.alloca [[VAR_121_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_123_:%.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-DAG:       [[VAR_124_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_125_:%.+]] = llvm.bitcast [[VAR_122_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_123_]], [[VAR_124_]], [[VAR_125_]], [[LOAD_VAR_102_MEM_]], [[LOAD_VAR_106_MEM_]], [[LOAD_VAR_108_MEM_]], [[LOAD_VAR_104_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64, i64) -> ()
// CHECK:           [[VAR_126_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_127_:%.+]] = llvm.alloca [[VAR_126_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_128_:%.+]] = llvm.bitcast [[VAR_122_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_129_:%.+]] = llvm.bitcast [[VAR_127_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_130_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_128_]], [[VAR_129_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_131_:%.+]] = llvm.alloca [[VAR_120_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_132_:%.+]] = llvm.bitcast [[VAR_127_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_133_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_132_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_134_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_122_]], [[VAR_134_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_135_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_127_]], [[VAR_135_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_136_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_133_]], [[VAR_136_]] : i64, !llvm.ptr
// CHECK:           [[VAR_137_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_119_]], [[VAR_137_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_138_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_139_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_138_]], [[VAR_139_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_140_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_141_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_141_]], [[VAR_140_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_142_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_143_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_143_]], [[VAR_142_]] : f32, !llvm.ptr
// CHECK:           [[VAR_144_:%.+]] = llvm.extractvalue [[VAR_43_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_145_:%.+]] = llvm.bitcast [[VAR_144_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_146_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_147_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_148_:%.+]] = llvm.alloca [[VAR_147_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_149_:%.+]] = llvm.mlir.constant(11 : i64) : i64
// CHECK-DAG:       [[VAR_150_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_151_:%.+]] = llvm.bitcast [[VAR_148_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_149_]], [[VAR_150_]], [[VAR_151_]], [[VAR_116_]], [[VAR_117_]], [[LOAD_VAR_104_MEM_]], [[LOAD_VAR_110_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64, i64) -> ()
// CHECK:           [[VAR_152_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_153_:%.+]] = llvm.alloca [[VAR_152_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_154_:%.+]] = llvm.bitcast [[VAR_148_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_155_:%.+]] = llvm.bitcast [[VAR_153_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_156_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_154_]], [[VAR_155_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_157_:%.+]] = llvm.alloca [[VAR_146_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_158_:%.+]] = llvm.bitcast [[VAR_153_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_159_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_158_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_160_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_148_]], [[VAR_160_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_161_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_153_]], [[VAR_161_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_162_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_159_]], [[VAR_162_]] : i64, !llvm.ptr
// CHECK:           [[VAR_163_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_145_]], [[VAR_163_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_164_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_165_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_164_]], [[VAR_165_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_166_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_167_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_167_]], [[VAR_166_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_168_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_169_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_169_]], [[VAR_168_]] : f32, !llvm.ptr
// CHECK:           [[VAR_170_:%.+]] = llvm.extractvalue [[VAR_65_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_171_:%.+]] = llvm.bitcast [[VAR_170_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_172_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_173_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_174_:%.+]] = llvm.alloca [[VAR_173_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_175_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_176_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_177_:%.+]] = llvm.bitcast [[VAR_174_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_175_]], [[VAR_176_]], [[VAR_177_]], [[LOAD_VAR_110_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64) -> ()
// CHECK:           [[VAR_178_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_179_:%.+]] = llvm.alloca [[VAR_178_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_180_:%.+]] = llvm.bitcast [[VAR_174_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_181_:%.+]] = llvm.bitcast [[VAR_179_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_182_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_180_]], [[VAR_181_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_183_:%.+]] = llvm.alloca [[VAR_172_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_184_:%.+]] = llvm.bitcast [[VAR_179_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_185_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_184_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_186_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_174_]], [[VAR_186_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_187_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_179_]], [[VAR_187_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_188_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_185_]], [[VAR_188_]] : i64, !llvm.ptr
// CHECK:           [[VAR_189_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_171_]], [[VAR_189_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_190_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_191_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_190_]], [[VAR_191_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_192_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_193_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_193_]], [[VAR_192_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_194_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_195_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_195_]], [[VAR_194_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_196_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_197_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       [[VAR_198_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       [[VAR_199_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_200_:%.+]] = llvm.extractvalue [[VAR_87_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_201_:%.+]] = llvm.bitcast [[VAR_200_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_202_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_203_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_204_:%.+]] = llvm.alloca [[VAR_203_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_205_:%.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-DAG:       [[VAR_206_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_207_:%.+]] = llvm.bitcast [[VAR_204_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_205_]], [[VAR_206_]], [[VAR_207_]], [[LOAD_VAR_102_MEM_]], [[LOAD_VAR_112_MEM_]], [[LOAD_VAR_114_MEM_]], [[LOAD_VAR_110_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64, i64) -> ()
// CHECK:           [[VAR_208_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_209_:%.+]] = llvm.alloca [[VAR_208_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_210_:%.+]] = llvm.bitcast [[VAR_204_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_211_:%.+]] = llvm.bitcast [[VAR_209_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_212_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_210_]], [[VAR_211_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_213_:%.+]] = llvm.alloca [[VAR_202_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_214_:%.+]] = llvm.bitcast [[VAR_209_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_215_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_214_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_216_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_204_]], [[VAR_216_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_217_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_209_]], [[VAR_217_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_218_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_215_]], [[VAR_218_]] : i64, !llvm.ptr
// CHECK:           [[VAR_219_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_201_]], [[VAR_219_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_220_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_221_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_220_]], [[VAR_221_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_222_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_223_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_223_]], [[VAR_222_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_224_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_225_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_225_]], [[VAR_224_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_226_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:       [[VAR_227_:%.+]] = llvm.bitcast [[VAR_131_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_228_:%.+]] = llvm.bitcast [[VAR_157_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_229_:%.+]] = llvm.bitcast [[VAR_183_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_230_:%.+]] = llvm.bitcast [[VAR_213_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_231_:%.+]] = llvm.call @zdnn_conv2d([[VAR_227_]], [[VAR_228_]], [[VAR_229_]], [[VAR_196_]], [[VAR_197_]], [[VAR_198_]], [[VAR_199_]], [[VAR_226_]], [[VAR_230_]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_cond2d_valid_padding() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_cond2d_valid_padding() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether conv2d calls the correct zDNN API or not.
func.func @test_call_zdnn_cond2d_relu_act() -> () {
  %input = memref.alloc() alignment = 4096 : memref<2048xf16>
  %kernel = memref.alloc() alignment = 4096 : memref<2048xf16>
  %bias = memref.alloc() alignment = 4096 : memref<2048xf16>
  %output = memref.alloc() alignment = 4096 : memref<2048xf16>
  %shape = memref.alloc() : memref<7xi64>
  "zlow.conv2d"(%input, %kernel, %bias, %shape, %output) {kernel_shape = [5, 5], strides = [2, 2], padding_type = "SAME_PADDING", act_func = "ACT_RELU" } : (memref<2048xf16>, memref<2048xf16>, memref<2048xf16>, memref<7xi64>, memref<2048xf16>)-> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnn_conv2d(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_cond2d_relu_act() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_3_:%.+]] = llvm.getelementptr [[VAR_2_]]{{.}}[[VAR_0_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.ptrtoint [[VAR_3_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_6_:%.+]] = llvm.add [[VAR_4_]], [[VAR_5_]] : i64
// CHECK:           [[VAR_7_:%.+]] = llvm.call @malloc([[VAR_6_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.ptrtoint [[VAR_7_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_9_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_10_:%.+]] = llvm.sub [[VAR_5_]], [[VAR_9_]] : i64
// CHECK:           [[VAR_11_:%.+]] = llvm.add [[VAR_8_]], [[VAR_10_]] : i64
// CHECK:           [[VAR_12_:%.+]] = llvm.urem [[VAR_11_]], [[VAR_5_]] : i64
// CHECK:           [[VAR_13_:%.+]] = llvm.sub [[VAR_11_]], [[VAR_12_]] : i64
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.inttoptr [[VAR_13_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_7_]], [[VAR_15_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_14_]], [[VAR_16_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_18_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_18_]], [[VAR_17_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_19_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_22_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_23_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_25_:%.+]] = llvm.getelementptr [[VAR_24_]]{{.}}[[VAR_22_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.ptrtoint [[VAR_25_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_28_:%.+]] = llvm.add [[VAR_26_]], [[VAR_27_]] : i64
// CHECK:           [[VAR_29_:%.+]] = llvm.call @malloc([[VAR_28_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.ptrtoint [[VAR_29_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_32_:%.+]] = llvm.sub [[VAR_27_]], [[VAR_31_]] : i64
// CHECK:           [[VAR_33_:%.+]] = llvm.add [[VAR_30_]], [[VAR_32_]] : i64
// CHECK:           [[VAR_34_:%.+]] = llvm.urem [[VAR_33_]], [[VAR_27_]] : i64
// CHECK:           [[VAR_35_:%.+]] = llvm.sub [[VAR_33_]], [[VAR_34_]] : i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.inttoptr [[VAR_35_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_29_]], [[VAR_37_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_38_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_39_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_42_:%.+]] = llvm.insertvalue [[VAR_22_]], [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_23_]], [[VAR_42_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_44_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_45_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_46_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_47_:%.+]] = llvm.getelementptr [[VAR_46_]]{{.}}[[VAR_44_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_48_:%.+]] = llvm.ptrtoint [[VAR_47_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_50_:%.+]] = llvm.add [[VAR_48_]], [[VAR_49_]] : i64
// CHECK:           [[VAR_51_:%.+]] = llvm.call @malloc([[VAR_50_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.ptrtoint [[VAR_51_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_54_:%.+]] = llvm.sub [[VAR_49_]], [[VAR_53_]] : i64
// CHECK:           [[VAR_55_:%.+]] = llvm.add [[VAR_52_]], [[VAR_54_]] : i64
// CHECK:           [[VAR_56_:%.+]] = llvm.urem [[VAR_55_]], [[VAR_49_]] : i64
// CHECK:           [[VAR_57_:%.+]] = llvm.sub [[VAR_55_]], [[VAR_56_]] : i64
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.inttoptr [[VAR_57_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_59_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_60_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_59_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_61_:%.+]] = llvm.insertvalue [[VAR_58_]], [[VAR_60_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_62_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_63_:%.+]] = llvm.insertvalue [[VAR_62_]], [[VAR_61_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_64_:%.+]] = llvm.insertvalue [[VAR_44_]], [[VAR_63_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.insertvalue [[VAR_45_]], [[VAR_64_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_67_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_69_:%.+]] = llvm.getelementptr [[VAR_68_]]{{.}}[[VAR_66_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_70_:%.+]] = llvm.ptrtoint [[VAR_69_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_72_:%.+]] = llvm.add [[VAR_70_]], [[VAR_71_]] : i64
// CHECK:           [[VAR_73_:%.+]] = llvm.call @malloc([[VAR_72_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.ptrtoint [[VAR_73_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_76_:%.+]] = llvm.sub [[VAR_71_]], [[VAR_75_]] : i64
// CHECK:           [[VAR_77_:%.+]] = llvm.add [[VAR_74_]], [[VAR_76_]] : i64
// CHECK:           [[VAR_78_:%.+]] = llvm.urem [[VAR_77_]], [[VAR_71_]] : i64
// CHECK:           [[VAR_79_:%.+]] = llvm.sub [[VAR_77_]], [[VAR_78_]] : i64
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.inttoptr [[VAR_79_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_82_:%.+]] = llvm.insertvalue [[VAR_73_]], [[VAR_81_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.insertvalue [[VAR_80_]], [[VAR_82_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_85_:%.+]] = llvm.insertvalue [[VAR_84_]], [[VAR_83_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_86_:%.+]] = llvm.insertvalue [[VAR_66_]], [[VAR_85_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_87_:%.+]] = llvm.insertvalue [[VAR_67_]], [[VAR_86_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_88_:%.+]] = llvm.mlir.constant(7 : index) : i64
// CHECK-DAG:       [[VAR_89_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_90_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_91_:%.+]] = llvm.getelementptr [[VAR_90_]]{{.}}[[VAR_88_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_92_:%.+]] = llvm.ptrtoint [[VAR_91_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.call @malloc([[VAR_92_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_95_:%.+]] = llvm.insertvalue [[VAR_93_]], [[VAR_94_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.insertvalue [[VAR_93_]], [[VAR_95_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_98_:%.+]] = llvm.insertvalue [[VAR_97_]], [[VAR_96_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_99_:%.+]] = llvm.insertvalue [[VAR_88_]], [[VAR_98_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_100_:%.+]] = llvm.insertvalue [[VAR_89_]], [[VAR_99_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_101_:%.+]] = llvm.extractvalue [[VAR_100_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_102_:%.+]] = llvm.getelementptr [[VAR_101_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_102_MEM_:%.+]] = llvm.load [[VAR_102_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.getelementptr [[VAR_101_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_104_MEM_:%.+]] = llvm.load [[VAR_104_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.getelementptr [[VAR_101_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_106_MEM_:%.+]] = llvm.load [[VAR_106_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.getelementptr [[VAR_101_]][3] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_108_MEM_:%.+]] = llvm.load [[VAR_108_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_110_:%.+]] = llvm.getelementptr [[VAR_101_]][4] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_110_MEM_:%.+]] = llvm.load [[VAR_110_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_112_:%.+]] = llvm.getelementptr [[VAR_101_]][5] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_112_MEM_:%.+]] = llvm.load [[VAR_112_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_114_:%.+]] = llvm.getelementptr [[VAR_101_]][6] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_114_MEM_:%.+]] = llvm.load [[VAR_114_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.mlir.constant(5 : i64) : i64
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.mlir.constant(5 : i64) : i64
// CHECK-DAG:       [[VAR_118_:%.+]] = llvm.extractvalue [[VAR_21_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_119_:%.+]] = llvm.bitcast [[VAR_118_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_122_:%.+]] = llvm.alloca [[VAR_121_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_123_:%.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-DAG:       [[VAR_124_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_125_:%.+]] = llvm.bitcast [[VAR_122_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_123_]], [[VAR_124_]], [[VAR_125_]], [[LOAD_VAR_102_MEM_]], [[LOAD_VAR_106_MEM_]], [[LOAD_VAR_108_MEM_]], [[LOAD_VAR_104_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64, i64) -> ()
// CHECK:           [[VAR_126_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_127_:%.+]] = llvm.alloca [[VAR_126_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_128_:%.+]] = llvm.bitcast [[VAR_122_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_129_:%.+]] = llvm.bitcast [[VAR_127_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_130_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_128_]], [[VAR_129_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_131_:%.+]] = llvm.alloca [[VAR_120_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_132_:%.+]] = llvm.bitcast [[VAR_127_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_133_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_132_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_134_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_122_]], [[VAR_134_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_135_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_127_]], [[VAR_135_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_136_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_133_]], [[VAR_136_]] : i64, !llvm.ptr
// CHECK:           [[VAR_137_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_119_]], [[VAR_137_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_138_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_139_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_138_]], [[VAR_139_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_140_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_141_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_141_]], [[VAR_140_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_142_:%.+]] = llvm.getelementptr [[VAR_131_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_143_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_143_]], [[VAR_142_]] : f32, !llvm.ptr
// CHECK:           [[VAR_144_:%.+]] = llvm.extractvalue [[VAR_43_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_145_:%.+]] = llvm.bitcast [[VAR_144_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_146_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_147_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_148_:%.+]] = llvm.alloca [[VAR_147_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_149_:%.+]] = llvm.mlir.constant(11 : i64) : i64
// CHECK-DAG:       [[VAR_150_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_151_:%.+]] = llvm.bitcast [[VAR_148_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_149_]], [[VAR_150_]], [[VAR_151_]], [[VAR_116_]], [[VAR_117_]], [[LOAD_VAR_104_MEM_]], [[LOAD_VAR_110_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64, i64) -> ()
// CHECK:           [[VAR_152_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_153_:%.+]] = llvm.alloca [[VAR_152_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_154_:%.+]] = llvm.bitcast [[VAR_148_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_155_:%.+]] = llvm.bitcast [[VAR_153_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_156_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_154_]], [[VAR_155_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_157_:%.+]] = llvm.alloca [[VAR_146_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_158_:%.+]] = llvm.bitcast [[VAR_153_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_159_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_158_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_160_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_148_]], [[VAR_160_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_161_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_153_]], [[VAR_161_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_162_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_159_]], [[VAR_162_]] : i64, !llvm.ptr
// CHECK:           [[VAR_163_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_145_]], [[VAR_163_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_164_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_165_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_164_]], [[VAR_165_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_166_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_167_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_167_]], [[VAR_166_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_168_:%.+]] = llvm.getelementptr [[VAR_157_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_169_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_169_]], [[VAR_168_]] : f32, !llvm.ptr
// CHECK:           [[VAR_170_:%.+]] = llvm.extractvalue [[VAR_65_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_171_:%.+]] = llvm.bitcast [[VAR_170_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_172_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_173_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_174_:%.+]] = llvm.alloca [[VAR_173_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_175_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_176_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_177_:%.+]] = llvm.bitcast [[VAR_174_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_175_]], [[VAR_176_]], [[VAR_177_]], [[LOAD_VAR_110_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64) -> ()
// CHECK:           [[VAR_178_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_179_:%.+]] = llvm.alloca [[VAR_178_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_180_:%.+]] = llvm.bitcast [[VAR_174_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_181_:%.+]] = llvm.bitcast [[VAR_179_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_182_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_180_]], [[VAR_181_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_183_:%.+]] = llvm.alloca [[VAR_172_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_184_:%.+]] = llvm.bitcast [[VAR_179_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_185_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_184_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_186_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_174_]], [[VAR_186_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_187_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_179_]], [[VAR_187_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_188_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_185_]], [[VAR_188_]] : i64, !llvm.ptr
// CHECK:           [[VAR_189_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_171_]], [[VAR_189_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_190_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_191_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_190_]], [[VAR_191_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_192_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_193_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_193_]], [[VAR_192_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_194_:%.+]] = llvm.getelementptr [[VAR_183_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_195_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_195_]], [[VAR_194_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_196_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_197_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       [[VAR_198_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       [[VAR_199_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_200_:%.+]] = llvm.extractvalue [[VAR_87_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_201_:%.+]] = llvm.bitcast [[VAR_200_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_202_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_203_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_204_:%.+]] = llvm.alloca [[VAR_203_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_205_:%.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-DAG:       [[VAR_206_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_207_:%.+]] = llvm.bitcast [[VAR_204_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_205_]], [[VAR_206_]], [[VAR_207_]], [[LOAD_VAR_102_MEM_]], [[LOAD_VAR_112_MEM_]], [[LOAD_VAR_114_MEM_]], [[LOAD_VAR_110_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64, i64) -> ()
// CHECK:           [[VAR_208_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_209_:%.+]] = llvm.alloca [[VAR_208_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_210_:%.+]] = llvm.bitcast [[VAR_204_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_211_:%.+]] = llvm.bitcast [[VAR_209_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_212_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_210_]], [[VAR_211_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_213_:%.+]] = llvm.alloca [[VAR_202_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_214_:%.+]] = llvm.bitcast [[VAR_209_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_215_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_214_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_216_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_204_]], [[VAR_216_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_217_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_209_]], [[VAR_217_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_218_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_215_]], [[VAR_218_]] : i64, !llvm.ptr
// CHECK:           [[VAR_219_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_201_]], [[VAR_219_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_220_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_221_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_220_]], [[VAR_221_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_222_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_223_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_223_]], [[VAR_222_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_224_:%.+]] = llvm.getelementptr [[VAR_213_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_225_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_225_]], [[VAR_224_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_226_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:       [[VAR_227_:%.+]] = llvm.bitcast [[VAR_131_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_228_:%.+]] = llvm.bitcast [[VAR_157_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_229_:%.+]] = llvm.bitcast [[VAR_183_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_230_:%.+]] = llvm.bitcast [[VAR_213_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_231_:%.+]] = llvm.call @zdnn_conv2d([[VAR_227_]], [[VAR_228_]], [[VAR_229_]], [[VAR_196_]], [[VAR_197_]], [[VAR_198_]], [[VAR_199_]], [[VAR_226_]], [[VAR_230_]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_cond2d_relu_act() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_cond2d_relu_act() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether avgpool2d calls the correct zDNN API or not.
func.func @test_call_zdnn_avgpool2d() -> () {
  %input = memref.alloc() alignment = 4096 : memref<2048xf16>
  %output = memref.alloc() alignment = 4096 : memref<2048xf16>
  %shape = memref.alloc() : memref<6xi64>
  "zlow.avgpool2d"(%input, %shape, %output) {kernel_shape = [5, 5], strides = [2, 2], padding_type = "SAME_PADDING" } : (memref<2048xf16>, memref<6xi64>, memref<2048xf16>)-> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnn_avgpool2d(!llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_avgpool2d() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_3_:%.+]] = llvm.getelementptr [[VAR_2_]]{{.}}[[VAR_0_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.ptrtoint [[VAR_3_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_6_:%.+]] = llvm.add [[VAR_4_]], [[VAR_5_]] : i64
// CHECK:           [[VAR_7_:%.+]] = llvm.call @malloc([[VAR_6_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.ptrtoint [[VAR_7_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_9_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_10_:%.+]] = llvm.sub [[VAR_5_]], [[VAR_9_]] : i64
// CHECK:           [[VAR_11_:%.+]] = llvm.add [[VAR_8_]], [[VAR_10_]] : i64
// CHECK:           [[VAR_12_:%.+]] = llvm.urem [[VAR_11_]], [[VAR_5_]] : i64
// CHECK:           [[VAR_13_:%.+]] = llvm.sub [[VAR_11_]], [[VAR_12_]] : i64
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.inttoptr [[VAR_13_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_7_]], [[VAR_15_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_14_]], [[VAR_16_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_18_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_18_]], [[VAR_17_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_19_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_22_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_23_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_25_:%.+]] = llvm.getelementptr [[VAR_24_]]{{.}}[[VAR_22_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.ptrtoint [[VAR_25_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_28_:%.+]] = llvm.add [[VAR_26_]], [[VAR_27_]] : i64
// CHECK:           [[VAR_29_:%.+]] = llvm.call @malloc([[VAR_28_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.ptrtoint [[VAR_29_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_32_:%.+]] = llvm.sub [[VAR_27_]], [[VAR_31_]] : i64
// CHECK:           [[VAR_33_:%.+]] = llvm.add [[VAR_30_]], [[VAR_32_]] : i64
// CHECK:           [[VAR_34_:%.+]] = llvm.urem [[VAR_33_]], [[VAR_27_]] : i64
// CHECK:           [[VAR_35_:%.+]] = llvm.sub [[VAR_33_]], [[VAR_34_]] : i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.inttoptr [[VAR_35_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_29_]], [[VAR_37_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_38_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_39_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_42_:%.+]] = llvm.insertvalue [[VAR_22_]], [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_23_]], [[VAR_42_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_44_:%.+]] = llvm.mlir.constant(6 : index) : i64
// CHECK-DAG:       [[VAR_45_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_46_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_47_:%.+]] = llvm.getelementptr [[VAR_46_]]{{.}}[[VAR_44_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_48_:%.+]] = llvm.ptrtoint [[VAR_47_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.call @malloc([[VAR_48_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_50_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_51_:%.+]] = llvm.insertvalue [[VAR_49_]], [[VAR_50_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.insertvalue [[VAR_49_]], [[VAR_51_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_54_:%.+]] = llvm.insertvalue [[VAR_53_]], [[VAR_52_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_55_:%.+]] = llvm.insertvalue [[VAR_44_]], [[VAR_54_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_56_:%.+]] = llvm.insertvalue [[VAR_45_]], [[VAR_55_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_57_:%.+]] = llvm.extractvalue [[VAR_56_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_58_:%.+]] = llvm.getelementptr [[VAR_57_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_58_MEM_:%.+]] = llvm.load [[VAR_58_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_60_:%.+]] = llvm.getelementptr [[VAR_57_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_60_MEM_:%.+]] = llvm.load [[VAR_60_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_62_:%.+]] = llvm.getelementptr [[VAR_57_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_62_MEM_:%.+]] = llvm.load [[VAR_62_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_64_:%.+]] = llvm.getelementptr [[VAR_57_]][3] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_64_MEM_:%.+]] = llvm.load [[VAR_64_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.getelementptr [[VAR_57_]][4] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_66_MEM_:%.+]] = llvm.load [[VAR_66_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.getelementptr [[VAR_57_]][5] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_68_MEM_:%.+]] = llvm.load [[VAR_68_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_70_:%.+]] = llvm.mlir.constant(5 : i64) : i64
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.mlir.constant(5 : i64) : i64
// CHECK-DAG:       [[VAR_72_:%.+]] = llvm.extractvalue [[VAR_21_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_73_:%.+]] = llvm.bitcast [[VAR_72_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_76_:%.+]] = llvm.alloca [[VAR_75_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-DAG:       [[VAR_78_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_79_:%.+]] = llvm.bitcast [[VAR_76_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_77_]], [[VAR_78_]], [[VAR_79_]], [[LOAD_VAR_58_MEM_]], [[LOAD_VAR_62_MEM_]], [[LOAD_VAR_64_MEM_]], [[LOAD_VAR_60_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64, i64) -> ()
// CHECK:           [[VAR_80_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.alloca [[VAR_80_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_82_:%.+]] = llvm.bitcast [[VAR_76_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_83_:%.+]] = llvm.bitcast [[VAR_81_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_82_]], [[VAR_83_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_85_:%.+]] = llvm.alloca [[VAR_74_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_86_:%.+]] = llvm.bitcast [[VAR_81_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_87_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_86_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_88_:%.+]] = llvm.getelementptr [[VAR_85_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_76_]], [[VAR_88_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_89_:%.+]] = llvm.getelementptr [[VAR_85_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_81_]], [[VAR_89_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_90_:%.+]] = llvm.getelementptr [[VAR_85_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_87_]], [[VAR_90_]] : i64, !llvm.ptr
// CHECK:           [[VAR_91_:%.+]] = llvm.getelementptr [[VAR_85_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_73_]], [[VAR_91_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_92_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.getelementptr [[VAR_85_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_92_]], [[VAR_93_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.getelementptr [[VAR_85_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_95_]], [[VAR_94_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.getelementptr [[VAR_85_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_97_]], [[VAR_96_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_98_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_99_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       [[VAR_100_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       [[VAR_101_:%.+]] = llvm.extractvalue [[VAR_43_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.bitcast [[VAR_101_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_105_:%.+]] = llvm.alloca [[VAR_104_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-DAG:       [[VAR_107_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_108_:%.+]] = llvm.bitcast [[VAR_105_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_106_]], [[VAR_107_]], [[VAR_108_]], [[LOAD_VAR_58_MEM_]], [[LOAD_VAR_66_MEM_]], [[LOAD_VAR_68_MEM_]], [[LOAD_VAR_60_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64, i64) -> ()
// CHECK:           [[VAR_109_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_110_:%.+]] = llvm.alloca [[VAR_109_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_111_:%.+]] = llvm.bitcast [[VAR_105_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_112_:%.+]] = llvm.bitcast [[VAR_110_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_113_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_111_]], [[VAR_112_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_114_:%.+]] = llvm.alloca [[VAR_103_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_115_:%.+]] = llvm.bitcast [[VAR_110_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_115_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.getelementptr [[VAR_114_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_105_]], [[VAR_117_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_118_:%.+]] = llvm.getelementptr [[VAR_114_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_110_]], [[VAR_118_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_119_:%.+]] = llvm.getelementptr [[VAR_114_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_116_]], [[VAR_119_]] : i64, !llvm.ptr
// CHECK:           [[VAR_120_:%.+]] = llvm.getelementptr [[VAR_114_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_120_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_122_:%.+]] = llvm.getelementptr [[VAR_114_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_121_]], [[VAR_122_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_123_:%.+]] = llvm.getelementptr [[VAR_114_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_124_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_124_]], [[VAR_123_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_125_:%.+]] = llvm.getelementptr [[VAR_114_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_126_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_126_]], [[VAR_125_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_127_:%.+]] = llvm.bitcast [[VAR_85_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_128_:%.+]] = llvm.bitcast [[VAR_114_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_129_:%.+]] = llvm.call @zdnn_avgpool2d([[VAR_127_]], [[VAR_98_]], [[VAR_70_]], [[VAR_71_]], [[VAR_99_]], [[VAR_100_]], [[VAR_128_]]) : (!llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_avgpool2d() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_avgpool2d() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether maxpool2d calls the correct zDNN API or not.
func.func @test_call_zdnn_maxpool2d() -> () {
  %input = memref.alloc() alignment = 4096 : memref<2048xf16>
  %output = memref.alloc() alignment = 4096 : memref<2048xf16>
  %shape = memref.alloc() : memref<6xi64>
  "zlow.maxpool2d"(%input, %shape, %output) {kernel_shape = [5, 5], strides = [2, 2], padding_type = "SAME_PADDING" } : (memref<2048xf16>, memref<6xi64>, memref<2048xf16>)-> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnn_maxpool2d(!llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_maxpool2d() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_3_:%.+]] = llvm.getelementptr [[VAR_2_]]{{.}}[[VAR_0_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.ptrtoint [[VAR_3_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_6_:%.+]] = llvm.add [[VAR_4_]], [[VAR_5_]] : i64
// CHECK:           [[VAR_7_:%.+]] = llvm.call @malloc([[VAR_6_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.ptrtoint [[VAR_7_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_9_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_10_:%.+]] = llvm.sub [[VAR_5_]], [[VAR_9_]] : i64
// CHECK:           [[VAR_11_:%.+]] = llvm.add [[VAR_8_]], [[VAR_10_]] : i64
// CHECK:           [[VAR_12_:%.+]] = llvm.urem [[VAR_11_]], [[VAR_5_]] : i64
// CHECK:           [[VAR_13_:%.+]] = llvm.sub [[VAR_11_]], [[VAR_12_]] : i64
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.inttoptr [[VAR_13_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_7_]], [[VAR_15_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_14_]], [[VAR_16_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_18_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_18_]], [[VAR_17_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_19_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_22_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_23_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_25_:%.+]] = llvm.getelementptr [[VAR_24_]]{{.}}[[VAR_22_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.ptrtoint [[VAR_25_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_28_:%.+]] = llvm.add [[VAR_26_]], [[VAR_27_]] : i64
// CHECK:           [[VAR_29_:%.+]] = llvm.call @malloc([[VAR_28_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.ptrtoint [[VAR_29_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_32_:%.+]] = llvm.sub [[VAR_27_]], [[VAR_31_]] : i64
// CHECK:           [[VAR_33_:%.+]] = llvm.add [[VAR_30_]], [[VAR_32_]] : i64
// CHECK:           [[VAR_34_:%.+]] = llvm.urem [[VAR_33_]], [[VAR_27_]] : i64
// CHECK:           [[VAR_35_:%.+]] = llvm.sub [[VAR_33_]], [[VAR_34_]] : i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.inttoptr [[VAR_35_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_29_]], [[VAR_37_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_38_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_39_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_42_:%.+]] = llvm.insertvalue [[VAR_22_]], [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_23_]], [[VAR_42_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_44_:%.+]] = llvm.mlir.constant(6 : index) : i64
// CHECK-DAG:       [[VAR_45_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_46_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_47_:%.+]] = llvm.getelementptr [[VAR_46_]]{{.}}[[VAR_44_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_48_:%.+]] = llvm.ptrtoint [[VAR_47_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.call @malloc([[VAR_48_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_50_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_51_:%.+]] = llvm.insertvalue [[VAR_49_]], [[VAR_50_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.insertvalue [[VAR_49_]], [[VAR_51_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_54_:%.+]] = llvm.insertvalue [[VAR_53_]], [[VAR_52_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_55_:%.+]] = llvm.insertvalue [[VAR_44_]], [[VAR_54_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_56_:%.+]] = llvm.insertvalue [[VAR_45_]], [[VAR_55_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_57_:%.+]] = llvm.extractvalue [[VAR_56_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_58_:%.+]] = llvm.getelementptr [[VAR_57_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_58_MEM_:%.+]] = llvm.load [[VAR_58_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_60_:%.+]] = llvm.getelementptr [[VAR_57_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_60_MEM_:%.+]] = llvm.load [[VAR_60_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_62_:%.+]] = llvm.getelementptr [[VAR_57_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_62_MEM_:%.+]] = llvm.load [[VAR_62_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_64_:%.+]] = llvm.getelementptr [[VAR_57_]][3] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_64_MEM_:%.+]] = llvm.load [[VAR_64_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.getelementptr [[VAR_57_]][4] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_66_MEM_:%.+]] = llvm.load [[VAR_66_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.getelementptr [[VAR_57_]][5] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_68_MEM_:%.+]] = llvm.load [[VAR_68_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_70_:%.+]] = llvm.mlir.constant(5 : i64) : i64
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.mlir.constant(5 : i64) : i64
// CHECK-DAG:       [[VAR_72_:%.+]] = llvm.extractvalue [[VAR_21_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_73_:%.+]] = llvm.bitcast [[VAR_72_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_76_:%.+]] = llvm.alloca [[VAR_75_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-DAG:       [[VAR_78_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_79_:%.+]] = llvm.bitcast [[VAR_76_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_77_]], [[VAR_78_]], [[VAR_79_]], [[LOAD_VAR_58_MEM_]], [[LOAD_VAR_62_MEM_]], [[LOAD_VAR_64_MEM_]], [[LOAD_VAR_60_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64, i64) -> ()
// CHECK:           [[VAR_80_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.alloca [[VAR_80_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_82_:%.+]] = llvm.bitcast [[VAR_76_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_83_:%.+]] = llvm.bitcast [[VAR_81_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_82_]], [[VAR_83_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_85_:%.+]] = llvm.alloca [[VAR_74_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_86_:%.+]] = llvm.bitcast [[VAR_81_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_87_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_86_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_88_:%.+]] = llvm.getelementptr [[VAR_85_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_76_]], [[VAR_88_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_89_:%.+]] = llvm.getelementptr [[VAR_85_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_81_]], [[VAR_89_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_90_:%.+]] = llvm.getelementptr [[VAR_85_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_87_]], [[VAR_90_]] : i64, !llvm.ptr
// CHECK:           [[VAR_91_:%.+]] = llvm.getelementptr [[VAR_85_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_73_]], [[VAR_91_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_92_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.getelementptr [[VAR_85_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_92_]], [[VAR_93_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.getelementptr [[VAR_85_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_95_]], [[VAR_94_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.getelementptr [[VAR_85_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_97_]], [[VAR_96_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_98_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_99_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       [[VAR_100_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       [[VAR_101_:%.+]] = llvm.extractvalue [[VAR_43_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.bitcast [[VAR_101_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_105_:%.+]] = llvm.alloca [[VAR_104_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-DAG:       [[VAR_107_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_108_:%.+]] = llvm.bitcast [[VAR_105_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_106_]], [[VAR_107_]], [[VAR_108_]], [[LOAD_VAR_58_MEM_]], [[LOAD_VAR_66_MEM_]], [[LOAD_VAR_68_MEM_]], [[LOAD_VAR_60_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64, i64) -> ()
// CHECK:           [[VAR_109_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_110_:%.+]] = llvm.alloca [[VAR_109_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_111_:%.+]] = llvm.bitcast [[VAR_105_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_112_:%.+]] = llvm.bitcast [[VAR_110_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_113_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_111_]], [[VAR_112_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_114_:%.+]] = llvm.alloca [[VAR_103_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_115_:%.+]] = llvm.bitcast [[VAR_110_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_115_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.getelementptr [[VAR_114_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_105_]], [[VAR_117_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_118_:%.+]] = llvm.getelementptr [[VAR_114_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_110_]], [[VAR_118_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_119_:%.+]] = llvm.getelementptr [[VAR_114_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_116_]], [[VAR_119_]] : i64, !llvm.ptr
// CHECK:           [[VAR_120_:%.+]] = llvm.getelementptr [[VAR_114_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_120_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_122_:%.+]] = llvm.getelementptr [[VAR_114_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_121_]], [[VAR_122_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_123_:%.+]] = llvm.getelementptr [[VAR_114_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_124_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_124_]], [[VAR_123_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_125_:%.+]] = llvm.getelementptr [[VAR_114_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_126_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_126_]], [[VAR_125_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_127_:%.+]] = llvm.bitcast [[VAR_85_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_128_:%.+]] = llvm.bitcast [[VAR_114_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_129_:%.+]] = llvm.call @zdnn_maxpool2d([[VAR_127_]], [[VAR_98_]], [[VAR_70_]], [[VAR_71_]], [[VAR_99_]], [[VAR_100_]], [[VAR_128_]]) : (!llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_maxpool2d() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_maxpool2d() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether meanreduce2d calls the correct zDNN API or not.
func.func @test_call_zdnn_meanreduce2d() -> () {
  %input = memref.alloc() alignment = 4096 : memref<2048xf16>
  %output = memref.alloc() alignment = 4096 : memref<2048xf16>
  %shape = memref.alloc() : memref<4xi64>
  "zlow.meanreduce2d"(%input, %shape, %output) : (memref<2048xf16>, memref<4xi64>, memref<2048xf16>)-> ()
  return


// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnn_meanreduce2d(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_meanreduce2d() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_3_:%.+]] = llvm.getelementptr [[VAR_2_]]{{.}}[[VAR_0_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.ptrtoint [[VAR_3_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_6_:%.+]] = llvm.add [[VAR_4_]], [[VAR_5_]] : i64
// CHECK:           [[VAR_7_:%.+]] = llvm.call @malloc([[VAR_6_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.ptrtoint [[VAR_7_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_9_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_10_:%.+]] = llvm.sub [[VAR_5_]], [[VAR_9_]] : i64
// CHECK:           [[VAR_11_:%.+]] = llvm.add [[VAR_8_]], [[VAR_10_]] : i64
// CHECK:           [[VAR_12_:%.+]] = llvm.urem [[VAR_11_]], [[VAR_5_]] : i64
// CHECK:           [[VAR_13_:%.+]] = llvm.sub [[VAR_11_]], [[VAR_12_]] : i64
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.inttoptr [[VAR_13_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_7_]], [[VAR_15_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_14_]], [[VAR_16_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_18_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_18_]], [[VAR_17_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_19_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_22_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_23_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_25_:%.+]] = llvm.getelementptr [[VAR_24_]]{{.}}[[VAR_22_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.ptrtoint [[VAR_25_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_28_:%.+]] = llvm.add [[VAR_26_]], [[VAR_27_]] : i64
// CHECK:           [[VAR_29_:%.+]] = llvm.call @malloc([[VAR_28_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.ptrtoint [[VAR_29_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_32_:%.+]] = llvm.sub [[VAR_27_]], [[VAR_31_]] : i64
// CHECK:           [[VAR_33_:%.+]] = llvm.add [[VAR_30_]], [[VAR_32_]] : i64
// CHECK:           [[VAR_34_:%.+]] = llvm.urem [[VAR_33_]], [[VAR_27_]] : i64
// CHECK:           [[VAR_35_:%.+]] = llvm.sub [[VAR_33_]], [[VAR_34_]] : i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.inttoptr [[VAR_35_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_29_]], [[VAR_37_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_38_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_39_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_42_:%.+]] = llvm.insertvalue [[VAR_22_]], [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_23_]], [[VAR_42_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_44_:%.+]] = llvm.mlir.constant(4 : index) : i64
// CHECK-DAG:       [[VAR_45_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_46_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_47_:%.+]] = llvm.getelementptr [[VAR_46_]]{{.}}[[VAR_44_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_48_:%.+]] = llvm.ptrtoint [[VAR_47_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.call @malloc([[VAR_48_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_50_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_51_:%.+]] = llvm.insertvalue [[VAR_49_]], [[VAR_50_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.insertvalue [[VAR_49_]], [[VAR_51_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_54_:%.+]] = llvm.insertvalue [[VAR_53_]], [[VAR_52_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_55_:%.+]] = llvm.insertvalue [[VAR_44_]], [[VAR_54_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.insertvalue [[VAR_45_]], [[VAR_55_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_57_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_58_:%.+]] = llvm.extractvalue [[VAR_56_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_59_:%.+]] = llvm.getelementptr [[VAR_58_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_59_MEM_:%.+]] = llvm.load [[VAR_59_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_61_:%.+]] = llvm.getelementptr [[VAR_58_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_61_MEM_:%.+]] = llvm.load [[VAR_61_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_63_:%.+]] = llvm.getelementptr [[VAR_58_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_63_MEM_:%.+]] = llvm.load [[VAR_63_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.getelementptr [[VAR_58_]][3] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_65_MEM_:%.+]] = llvm.load [[VAR_65_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_67_:%.+]] = llvm.extractvalue [[VAR_21_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.bitcast [[VAR_67_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_69_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_70_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.alloca [[VAR_70_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_72_:%.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-DAG:       [[VAR_73_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_74_:%.+]] = llvm.bitcast [[VAR_71_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_72_]], [[VAR_73_]], [[VAR_74_]], [[LOAD_VAR_59_MEM_]], [[LOAD_VAR_61_MEM_]], [[LOAD_VAR_63_MEM_]], [[LOAD_VAR_65_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64, i64) -> ()
// CHECK:           [[VAR_75_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_76_:%.+]] = llvm.alloca [[VAR_75_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.bitcast [[VAR_71_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_78_:%.+]] = llvm.bitcast [[VAR_76_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_79_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_77_]], [[VAR_78_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.alloca [[VAR_69_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.bitcast [[VAR_76_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_82_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_81_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.getelementptr [[VAR_80_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_71_]], [[VAR_83_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_84_:%.+]] = llvm.getelementptr [[VAR_80_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_76_]], [[VAR_84_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_85_:%.+]] = llvm.getelementptr [[VAR_80_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_82_]], [[VAR_85_]] : i64, !llvm.ptr
// CHECK:           [[VAR_86_:%.+]] = llvm.getelementptr [[VAR_80_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_68_]], [[VAR_86_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_87_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_88_:%.+]] = llvm.getelementptr [[VAR_80_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_87_]], [[VAR_88_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_89_:%.+]] = llvm.getelementptr [[VAR_80_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_90_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_90_]], [[VAR_89_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_91_:%.+]] = llvm.getelementptr [[VAR_80_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_92_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_92_]], [[VAR_91_]] : f32, !llvm.ptr
// CHECK:           [[VAR_93_:%.+]] = llvm.extractvalue [[VAR_43_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.bitcast [[VAR_93_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.alloca [[VAR_96_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_98_:%.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-DAG:       [[VAR_99_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_100_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_98_]], [[VAR_99_]], [[VAR_100_]], [[LOAD_VAR_59_MEM_]], [[VAR_57_]], [[VAR_57_]], [[LOAD_VAR_65_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64, i64) -> ()
// CHECK:           [[VAR_101_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.alloca [[VAR_101_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.bitcast [[VAR_97_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_104_:%.+]] = llvm.bitcast [[VAR_102_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_105_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_103_]], [[VAR_104_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.alloca [[VAR_95_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_107_:%.+]] = llvm.bitcast [[VAR_102_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_107_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_97_]], [[VAR_109_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_110_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_102_]], [[VAR_110_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_111_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_111_]] : i64, !llvm.ptr
// CHECK:           [[VAR_112_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_94_]], [[VAR_112_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_113_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_114_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_113_]], [[VAR_114_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_115_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_116_]], [[VAR_115_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.getelementptr [[VAR_106_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_118_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_118_]], [[VAR_117_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_119_:%.+]] = llvm.bitcast [[VAR_80_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.bitcast [[VAR_106_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_121_:%.+]] = llvm.call @zdnn_meanreduce2d([[VAR_119_]], [[VAR_120_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_meanreduce2d() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_meanreduce2d() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

// -----

// Check whether batchnorm calls the correct zDNN API or not.
func.func @test_call_zdnn_batchnorm() -> () {
  %input = memref.alloc() alignment = 4096 : memref<2048xf16>
  %a = memref.alloc() alignment = 4096 : memref<2048xf16>
  %b = memref.alloc() alignment = 4096 : memref<2048xf16>
  %shape = memref.alloc() : memref<4xi64>
  %output = memref.alloc() alignment = 4096 : memref<2048xf16>
  "zlow.batchnorm"(%input, %a, %b, %shape, %output) : (memref<2048xf16>, memref<2048xf16>, memref<2048xf16>, memref<4xi64>, memref<2048xf16>)-> ()
  return

// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnn_batchnorm(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_call_zdnn_batchnorm() attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_3_:%.+]] = llvm.getelementptr [[VAR_2_]]{{.}}[[VAR_0_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.ptrtoint [[VAR_3_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_6_:%.+]] = llvm.add [[VAR_4_]], [[VAR_5_]] : i64
// CHECK:           [[VAR_7_:%.+]] = llvm.call @malloc([[VAR_6_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.ptrtoint [[VAR_7_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_9_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_10_:%.+]] = llvm.sub [[VAR_5_]], [[VAR_9_]] : i64
// CHECK:           [[VAR_11_:%.+]] = llvm.add [[VAR_8_]], [[VAR_10_]] : i64
// CHECK:           [[VAR_12_:%.+]] = llvm.urem [[VAR_11_]], [[VAR_5_]] : i64
// CHECK:           [[VAR_13_:%.+]] = llvm.sub [[VAR_11_]], [[VAR_12_]] : i64
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.inttoptr [[VAR_13_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_15_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_16_:%.+]] = llvm.insertvalue [[VAR_7_]], [[VAR_15_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_17_:%.+]] = llvm.insertvalue [[VAR_14_]], [[VAR_16_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_18_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_19_:%.+]] = llvm.insertvalue [[VAR_18_]], [[VAR_17_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_20_:%.+]] = llvm.insertvalue [[VAR_0_]], [[VAR_19_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_21_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_20_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_22_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_23_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_25_:%.+]] = llvm.getelementptr [[VAR_24_]]{{.}}[[VAR_22_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.ptrtoint [[VAR_25_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_28_:%.+]] = llvm.add [[VAR_26_]], [[VAR_27_]] : i64
// CHECK:           [[VAR_29_:%.+]] = llvm.call @malloc([[VAR_28_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_30_:%.+]] = llvm.ptrtoint [[VAR_29_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_32_:%.+]] = llvm.sub [[VAR_27_]], [[VAR_31_]] : i64
// CHECK:           [[VAR_33_:%.+]] = llvm.add [[VAR_30_]], [[VAR_32_]] : i64
// CHECK:           [[VAR_34_:%.+]] = llvm.urem [[VAR_33_]], [[VAR_27_]] : i64
// CHECK:           [[VAR_35_:%.+]] = llvm.sub [[VAR_33_]], [[VAR_34_]] : i64
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.inttoptr [[VAR_35_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_29_]], [[VAR_37_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_36_]], [[VAR_38_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_40_]], [[VAR_39_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_42_:%.+]] = llvm.insertvalue [[VAR_22_]], [[VAR_41_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_43_:%.+]] = llvm.insertvalue [[VAR_23_]], [[VAR_42_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_44_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_45_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_46_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_47_:%.+]] = llvm.getelementptr [[VAR_46_]]{{.}}[[VAR_44_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_48_:%.+]] = llvm.ptrtoint [[VAR_47_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_49_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_50_:%.+]] = llvm.add [[VAR_48_]], [[VAR_49_]] : i64
// CHECK:           [[VAR_51_:%.+]] = llvm.call @malloc([[VAR_50_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.ptrtoint [[VAR_51_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_54_:%.+]] = llvm.sub [[VAR_49_]], [[VAR_53_]] : i64
// CHECK:           [[VAR_55_:%.+]] = llvm.add [[VAR_52_]], [[VAR_54_]] : i64
// CHECK:           [[VAR_56_:%.+]] = llvm.urem [[VAR_55_]], [[VAR_49_]] : i64
// CHECK:           [[VAR_57_:%.+]] = llvm.sub [[VAR_55_]], [[VAR_56_]] : i64
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.inttoptr [[VAR_57_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_59_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_60_:%.+]] = llvm.insertvalue [[VAR_51_]], [[VAR_59_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_61_:%.+]] = llvm.insertvalue [[VAR_58_]], [[VAR_60_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_62_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_63_:%.+]] = llvm.insertvalue [[VAR_62_]], [[VAR_61_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_64_:%.+]] = llvm.insertvalue [[VAR_44_]], [[VAR_63_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.insertvalue [[VAR_45_]], [[VAR_64_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.mlir.constant(4 : index) : i64
// CHECK-DAG:       [[VAR_67_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_68_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_69_:%.+]] = llvm.getelementptr [[VAR_68_]]{{.}}[[VAR_66_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:           [[VAR_70_:%.+]] = llvm.ptrtoint [[VAR_69_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.call @malloc([[VAR_70_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_72_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_73_:%.+]] = llvm.insertvalue [[VAR_71_]], [[VAR_72_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.insertvalue [[VAR_71_]], [[VAR_73_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_76_:%.+]] = llvm.insertvalue [[VAR_75_]], [[VAR_74_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_77_:%.+]] = llvm.insertvalue [[VAR_66_]], [[VAR_76_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_78_:%.+]] = llvm.insertvalue [[VAR_67_]], [[VAR_77_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_79_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_82_:%.+]] = llvm.getelementptr [[VAR_81_]]{{.}}[[VAR_79_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.ptrtoint [[VAR_82_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_85_:%.+]] = llvm.add [[VAR_83_]], [[VAR_84_]] : i64
// CHECK:           [[VAR_86_:%.+]] = llvm.call @malloc([[VAR_85_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_87_:%.+]] = llvm.ptrtoint [[VAR_86_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_88_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_89_:%.+]] = llvm.sub [[VAR_84_]], [[VAR_88_]] : i64
// CHECK:           [[VAR_90_:%.+]] = llvm.add [[VAR_87_]], [[VAR_89_]] : i64
// CHECK:           [[VAR_91_:%.+]] = llvm.urem [[VAR_90_]], [[VAR_84_]] : i64
// CHECK:           [[VAR_92_:%.+]] = llvm.sub [[VAR_90_]], [[VAR_91_]] : i64
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.inttoptr [[VAR_92_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_95_:%.+]] = llvm.insertvalue [[VAR_86_]], [[VAR_94_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.insertvalue [[VAR_93_]], [[VAR_95_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_98_:%.+]] = llvm.insertvalue [[VAR_97_]], [[VAR_96_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_99_:%.+]] = llvm.insertvalue [[VAR_79_]], [[VAR_98_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_100_:%.+]] = llvm.insertvalue [[VAR_80_]], [[VAR_99_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_101_:%.+]] = llvm.extractvalue [[VAR_78_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_102_:%.+]] = llvm.getelementptr [[VAR_101_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_102_MEM_:%.+]] = llvm.load [[VAR_102_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.getelementptr [[VAR_101_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_104_MEM_:%.+]] = llvm.load [[VAR_104_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.getelementptr [[VAR_101_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_106_MEM_:%.+]] = llvm.load [[VAR_106_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.getelementptr [[VAR_101_]][3] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_108_MEM_:%.+]] = llvm.load [[VAR_108_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_110_:%.+]] = llvm.extractvalue [[VAR_21_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_111_:%.+]] = llvm.bitcast [[VAR_110_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_112_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_113_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_114_:%.+]] = llvm.alloca [[VAR_113_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_115_:%.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_117_:%.+]] = llvm.bitcast [[VAR_114_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_115_]], [[VAR_116_]], [[VAR_117_]], [[LOAD_VAR_102_MEM_]], [[LOAD_VAR_104_MEM_]], [[LOAD_VAR_106_MEM_]], [[LOAD_VAR_108_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64, i64) -> ()
// CHECK:           [[VAR_118_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_119_:%.+]] = llvm.alloca [[VAR_118_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.bitcast [[VAR_114_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_121_:%.+]] = llvm.bitcast [[VAR_119_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_122_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_120_]], [[VAR_121_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_123_:%.+]] = llvm.alloca [[VAR_112_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_124_:%.+]] = llvm.bitcast [[VAR_119_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_125_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_124_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_126_:%.+]] = llvm.getelementptr [[VAR_123_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_114_]], [[VAR_126_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_127_:%.+]] = llvm.getelementptr [[VAR_123_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_119_]], [[VAR_127_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_128_:%.+]] = llvm.getelementptr [[VAR_123_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_125_]], [[VAR_128_]] : i64, !llvm.ptr
// CHECK:           [[VAR_129_:%.+]] = llvm.getelementptr [[VAR_123_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_111_]], [[VAR_129_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_130_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_131_:%.+]] = llvm.getelementptr [[VAR_123_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_130_]], [[VAR_131_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_132_:%.+]] = llvm.getelementptr [[VAR_123_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_133_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_133_]], [[VAR_132_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_134_:%.+]] = llvm.getelementptr [[VAR_123_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_135_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_135_]], [[VAR_134_]] : f32, !llvm.ptr
// CHECK:           [[VAR_136_:%.+]] = llvm.extractvalue [[VAR_43_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_137_:%.+]] = llvm.bitcast [[VAR_136_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_138_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_139_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_140_:%.+]] = llvm.alloca [[VAR_139_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_141_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_142_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_143_:%.+]] = llvm.bitcast [[VAR_140_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_141_]], [[VAR_142_]], [[VAR_143_]], [[LOAD_VAR_108_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64) -> ()
// CHECK:           [[VAR_144_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_145_:%.+]] = llvm.alloca [[VAR_144_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_146_:%.+]] = llvm.bitcast [[VAR_140_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_147_:%.+]] = llvm.bitcast [[VAR_145_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_148_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_146_]], [[VAR_147_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_149_:%.+]] = llvm.alloca [[VAR_138_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_150_:%.+]] = llvm.bitcast [[VAR_145_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_151_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_150_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_152_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_140_]], [[VAR_152_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_153_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_145_]], [[VAR_153_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_154_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_151_]], [[VAR_154_]] : i64, !llvm.ptr
// CHECK:           [[VAR_155_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_137_]], [[VAR_155_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_156_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_157_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_156_]], [[VAR_157_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_158_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_159_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_159_]], [[VAR_158_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_160_:%.+]] = llvm.getelementptr [[VAR_149_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_161_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_161_]], [[VAR_160_]] : f32, !llvm.ptr
// CHECK:           [[VAR_162_:%.+]] = llvm.extractvalue [[VAR_65_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_163_:%.+]] = llvm.bitcast [[VAR_162_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_164_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_165_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_166_:%.+]] = llvm.alloca [[VAR_165_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_167_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_168_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_169_:%.+]] = llvm.bitcast [[VAR_166_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_167_]], [[VAR_168_]], [[VAR_169_]], [[LOAD_VAR_108_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64) -> ()
// CHECK:           [[VAR_170_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_171_:%.+]] = llvm.alloca [[VAR_170_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_172_:%.+]] = llvm.bitcast [[VAR_166_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_173_:%.+]] = llvm.bitcast [[VAR_171_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_174_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_172_]], [[VAR_173_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_175_:%.+]] = llvm.alloca [[VAR_164_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_176_:%.+]] = llvm.bitcast [[VAR_171_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_177_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_176_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_178_:%.+]] = llvm.getelementptr [[VAR_175_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_166_]], [[VAR_178_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_179_:%.+]] = llvm.getelementptr [[VAR_175_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_171_]], [[VAR_179_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_180_:%.+]] = llvm.getelementptr [[VAR_175_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_177_]], [[VAR_180_]] : i64, !llvm.ptr
// CHECK:           [[VAR_181_:%.+]] = llvm.getelementptr [[VAR_175_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_163_]], [[VAR_181_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_182_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_183_:%.+]] = llvm.getelementptr [[VAR_175_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_182_]], [[VAR_183_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_184_:%.+]] = llvm.getelementptr [[VAR_175_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_185_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_185_]], [[VAR_184_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_186_:%.+]] = llvm.getelementptr [[VAR_175_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_187_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_187_]], [[VAR_186_]] : f32, !llvm.ptr
// CHECK:           [[VAR_188_:%.+]] = llvm.extractvalue [[VAR_100_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_189_:%.+]] = llvm.bitcast [[VAR_188_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_190_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_191_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_192_:%.+]] = llvm.alloca [[VAR_191_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_193_:%.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-DAG:       [[VAR_194_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_195_:%.+]] = llvm.bitcast [[VAR_192_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_193_]], [[VAR_194_]], [[VAR_195_]], [[LOAD_VAR_102_MEM_]], [[LOAD_VAR_104_MEM_]], [[LOAD_VAR_106_MEM_]], [[LOAD_VAR_108_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64, i64) -> ()
// CHECK:           [[VAR_196_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_197_:%.+]] = llvm.alloca [[VAR_196_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_198_:%.+]] = llvm.bitcast [[VAR_192_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_199_:%.+]] = llvm.bitcast [[VAR_197_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_200_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_198_]], [[VAR_199_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_201_:%.+]] = llvm.alloca [[VAR_190_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_202_:%.+]] = llvm.bitcast [[VAR_197_]] : !llvm.ptr to !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_203_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_202_]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       [[VAR_204_:%.+]] = llvm.getelementptr [[VAR_201_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_192_]], [[VAR_204_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_205_:%.+]] = llvm.getelementptr [[VAR_201_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_197_]], [[VAR_205_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_206_:%.+]] = llvm.getelementptr [[VAR_201_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_203_]], [[VAR_206_]] : i64, !llvm.ptr
// CHECK:           [[VAR_207_:%.+]] = llvm.getelementptr [[VAR_201_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_189_]], [[VAR_207_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_208_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_209_:%.+]] = llvm.getelementptr [[VAR_201_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_208_]], [[VAR_209_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_210_:%.+]] = llvm.getelementptr [[VAR_201_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_211_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_211_]], [[VAR_210_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_212_:%.+]] = llvm.getelementptr [[VAR_201_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_213_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_213_]], [[VAR_212_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_214_:%.+]] = llvm.bitcast [[VAR_123_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_215_:%.+]] = llvm.bitcast [[VAR_149_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_216_:%.+]] = llvm.bitcast [[VAR_175_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_217_:%.+]] = llvm.bitcast [[VAR_201_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_218_:%.+]] = llvm.call @zdnn_batchnorm([[VAR_214_]], [[VAR_215_]], [[VAR_216_]], [[VAR_217_]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_call_zdnn_batchnorm() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_call_zdnn_batchnorm() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

