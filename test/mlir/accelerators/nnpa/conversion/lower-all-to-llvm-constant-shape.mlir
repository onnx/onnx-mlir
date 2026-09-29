// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-krnl-to-llvm -cse %s -split-input-file | FileCheck %s

// -----

// COM: Check the lowering of an zlow operation when its shape includes constant dims.
// COM: In this case, the constant values will be passed directly to
// COM: 'zdnn_init_pre_transformed_desc' that initializes a zTensor descriptor.
// COM: Using zlow.softmax as an example.
func.func @test_zlow_softmax_constant_shape() -> () {
  // %0 = "onnx.Softmax"(%arg0) : (memref<5x10xf32>) -> memref<5x10xf32>
  // "func.return"(%0) : (memref<5x10xf32>) -> ()
  %shape = "krnl.global"() {name = "constant_fold_std_alloc_0", shape = [3], value = dense<[1, 5, 10]> : tensor<3xi64>} : () -> memref<3xi64>
  %res = memref.alloc() alignment = 4096 : memref<1x1x1x1x32x64xf16>
  %input = memref.alloc() alignment = 4096 : memref<1x1x1x1x32x64xf16>
  %work_area = memref.alloc() alignment = 4096 : memref<8192xi8>
  "zlow.softmax"(%input, %work_area, %shape, %res) {act_func = "ACT_NONE"} : (memref<1x1x1x1x32x64xf16>, memref<8192xi8>, memref<3xi64>, memref<1x1x1x1x32x64xf16>) -> ()
  return

// ...

// ...

// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\0A\22compiler_version\22: \22\22,\0A\22compile_options\22: \22\22,\0A\22accelerators\22: {\22NNPA\22: {\22nnpa_level\22: \22--march=z16\22, \22zdnn_version\22: \221.0.1\22}},\0A\22op_stats\22: }\00") {addr_space = 0 : i32}
// CHECK:         llvm.func @zdnnx_softmax(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_getsize_ztensor(!llvm.ptr) -> i64
// CHECK:         llvm.func @zdnn_generate_transformed_desc(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func @zdnn_init_pre_transformed_desc(i64, i64, !llvm.ptr, ...)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.mlir.global internal constant @constant_fold_std_alloc_0(dense<[1, 5, 10]> : tensor<3xi64>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<3 x i64>
// CHECK:         llvm.func @test_zlow_softmax_constant_shape() attributes {llvm.emit_c_interface} {
// CHECK:           [[VAR_0_:%.+]] = llvm.mlir.addressof @constant_fold_std_alloc_0 : !llvm.ptr
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.bitcast [[VAR_0_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_3_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_2_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_3_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_4_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.insertvalue [[VAR_7_]], [[VAR_6_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_9_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_8_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG:       [[VAR_13_:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK-DAG:       [[VAR_14_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_15_:%.+]] = llvm.getelementptr [[VAR_14_]]{{.}}[[VAR_13_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-DAG:       [[VAR_16_:%.+]] = llvm.ptrtoint [[VAR_15_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_17_:%.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK:           [[VAR_18_:%.+]] = llvm.add [[VAR_16_]], [[VAR_17_]] : i64
// CHECK:           [[VAR_19_:%.+]] = llvm.call @malloc([[VAR_18_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_20_:%.+]] = llvm.ptrtoint [[VAR_19_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_21_:%.+]] = llvm.sub [[VAR_17_]], [[VAR_9_]] : i64
// CHECK:           [[VAR_22_:%.+]] = llvm.add [[VAR_20_]], [[VAR_21_]] : i64
// CHECK:           [[VAR_23_:%.+]] = llvm.urem [[VAR_22_]], [[VAR_17_]] : i64
// CHECK:           [[VAR_24_:%.+]] = llvm.sub [[VAR_22_]], [[VAR_23_]] : i64
// CHECK-DAG:       [[VAR_25_:%.+]] = llvm.inttoptr [[VAR_24_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_26_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_27_:%.+]] = llvm.insertvalue [[VAR_19_]], [[VAR_26_]][0] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_28_:%.+]] = llvm.insertvalue [[VAR_25_]], [[VAR_27_]][1] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_29_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_28_]][2] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_30_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_29_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_31_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_30_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_32_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_31_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_33_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_32_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_34_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_33_]][3, 4] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_35_:%.+]] = llvm.insertvalue [[VAR_12_]], [[VAR_34_]][3, 5] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_36_:%.+]] = llvm.insertvalue [[VAR_13_]], [[VAR_35_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_37_:%.+]] = llvm.insertvalue [[VAR_13_]], [[VAR_36_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_38_:%.+]] = llvm.insertvalue [[VAR_13_]], [[VAR_37_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_39_:%.+]] = llvm.insertvalue [[VAR_13_]], [[VAR_38_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_40_:%.+]] = llvm.insertvalue [[VAR_12_]], [[VAR_39_]][4, 4] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK-DAG:       [[VAR_41_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_40_]][4, 5] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK-DAG:       [[VAR_42_:%.+]] = llvm.call @malloc([[VAR_18_]]) : (i64) -> !llvm.ptr
// CHECK:           [[VAR_43_:%.+]] = llvm.ptrtoint [[VAR_42_]] : !llvm.ptr to i64
// CHECK:           [[VAR_44_:%.+]] = llvm.add [[VAR_43_]], [[VAR_21_]] : i64
// CHECK:           [[VAR_45_:%.+]] = llvm.urem [[VAR_44_]], [[VAR_17_]] : i64
// CHECK:           [[VAR_46_:%.+]] = llvm.sub [[VAR_44_]], [[VAR_45_]] : i64
// CHECK-DAG:       [[VAR_47_:%.+]] = llvm.inttoptr [[VAR_46_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_48_:%.+]] = llvm.insertvalue [[VAR_42_]], [[VAR_26_]][0] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_49_:%.+]] = llvm.insertvalue [[VAR_47_]], [[VAR_48_]][1] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_50_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_49_]][2] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_51_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_50_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_52_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_51_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_53_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_52_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_54_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_53_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_55_:%.+]] = llvm.insertvalue [[VAR_11_]], [[VAR_54_]][3, 4] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_56_:%.+]] = llvm.insertvalue [[VAR_12_]], [[VAR_55_]][3, 5] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_57_:%.+]] = llvm.insertvalue [[VAR_13_]], [[VAR_56_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_58_:%.+]] = llvm.insertvalue [[VAR_13_]], [[VAR_57_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_59_:%.+]] = llvm.insertvalue [[VAR_13_]], [[VAR_58_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_60_:%.+]] = llvm.insertvalue [[VAR_13_]], [[VAR_59_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK:           [[VAR_61_:%.+]] = llvm.insertvalue [[VAR_12_]], [[VAR_60_]][4, 4] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK-DAG:       [[VAR_62_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_61_]][4, 5] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK-DAG:       [[VAR_63_:%.+]] = llvm.mlir.constant(8192 : index) : i64
// CHECK:           [[VAR_64_:%.+]] = llvm.getelementptr [[VAR_14_]]{{.}}[[VAR_63_]]{{.}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK:           [[VAR_65_:%.+]] = llvm.ptrtoint [[VAR_64_]] : !llvm.ptr to i64
// CHECK:           [[VAR_66_:%.+]] = llvm.add [[VAR_65_]], [[VAR_17_]] : i64
// CHECK:           [[VAR_67_:%.+]] = llvm.call @malloc([[VAR_66_]]) : (i64) -> !llvm.ptr
// CHECK:           [[VAR_68_:%.+]] = llvm.ptrtoint [[VAR_67_]] : !llvm.ptr to i64
// CHECK:           [[VAR_69_:%.+]] = llvm.add [[VAR_68_]], [[VAR_21_]] : i64
// CHECK:           [[VAR_70_:%.+]] = llvm.urem [[VAR_69_]], [[VAR_17_]] : i64
// CHECK:           [[VAR_71_:%.+]] = llvm.sub [[VAR_69_]], [[VAR_70_]] : i64
// CHECK-DAG:       [[VAR_72_:%.+]] = llvm.inttoptr [[VAR_71_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_73_:%.+]] = llvm.insertvalue [[VAR_67_]], [[VAR_2_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_74_:%.+]] = llvm.insertvalue [[VAR_72_]], [[VAR_73_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_75_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_74_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_76_:%.+]] = llvm.insertvalue [[VAR_63_]], [[VAR_75_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_76_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_78_:%.+]] = llvm.extractvalue [[VAR_10_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_79_:%.+]] = llvm.getelementptr [[VAR_78_]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-DAG:       [[LOAD_VAR_79_MEM_:%.+]] = llvm.load [[VAR_79_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.getelementptr [[VAR_78_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_81_MEM_:%.+]] = llvm.load [[VAR_81_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.getelementptr [[VAR_78_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_83_MEM_:%.+]] = llvm.load [[VAR_83_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_85_:%.+]] = llvm.extractvalue [[VAR_62_]][1] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_86_:%.+]] = llvm.bitcast [[VAR_85_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_87_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_88_:%.+]] = llvm.alloca [[VAR_87_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_89_:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:       [[VAR_90_:%.+]] = llvm.mlir.constant(254 : i64) : i64
// CHECK:           [[VAR_91_:%.+]] = llvm.bitcast [[VAR_88_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.call @zdnn_init_pre_transformed_desc([[VAR_89_]], [[VAR_90_]], [[VAR_91_]], [[LOAD_VAR_79_MEM_]], [[LOAD_VAR_81_MEM_]], [[LOAD_VAR_83_MEM_]]) vararg(!llvm.func<void (i64, i64, ptr, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64) -> ()
// CHECK:           [[VAR_92_:%.+]] = llvm.alloca [[VAR_87_]] x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_93_:%.+]] = llvm.bitcast [[VAR_92_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_94_:%.+]] = llvm.call @zdnn_generate_transformed_desc([[VAR_91_]], [[VAR_93_]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.alloca [[VAR_87_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_96_:%.+]] = llvm.call @zdnn_getsize_ztensor([[VAR_93_]]) : (!llvm.ptr) -> i64
// CHECK:           [[VAR_97_:%.+]] = llvm.getelementptr [[VAR_95_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_88_]], [[VAR_97_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_98_:%.+]] = llvm.getelementptr [[VAR_95_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_92_]], [[VAR_98_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_99_:%.+]] = llvm.getelementptr [[VAR_95_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_96_]], [[VAR_99_]] : i64, !llvm.ptr
// CHECK:           [[VAR_100_:%.+]] = llvm.getelementptr [[VAR_95_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_86_]], [[VAR_100_]] : !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[VAR_101_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_102_:%.+]] = llvm.getelementptr [[VAR_95_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_101_]], [[VAR_102_]] : i1, !llvm.ptr
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.getelementptr [[VAR_95_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:           llvm.store [[VAR_104_]], [[VAR_103_]] : f32, !llvm.ptr
// CHECK:           [[VAR_105_:%.+]] = llvm.getelementptr [[VAR_95_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_104_]], [[VAR_105_]] : f32, !llvm.ptr
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_107_:%.+]] = llvm.extractvalue [[VAR_41_]][1] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_108_:%.+]] = llvm.bitcast [[VAR_107_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_109_:%.+]] = llvm.alloca [[VAR_87_]] x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
// CHECK:           [[VAR_110_:%.+]] = llvm.getelementptr [[VAR_109_]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_88_]], [[VAR_110_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_111_:%.+]] = llvm.getelementptr [[VAR_109_]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_92_]], [[VAR_111_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_112_:%.+]] = llvm.getelementptr [[VAR_109_]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_96_]], [[VAR_112_]] : i64, !llvm.ptr
// CHECK:           [[VAR_113_:%.+]] = llvm.getelementptr [[VAR_109_]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_108_]], [[VAR_113_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_114_:%.+]] = llvm.getelementptr [[VAR_109_]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_101_]], [[VAR_114_]] : i1, !llvm.ptr
// CHECK:           [[VAR_115_:%.+]] = llvm.getelementptr [[VAR_109_]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_104_]], [[VAR_115_]] : f32, !llvm.ptr
// CHECK:           [[VAR_116_:%.+]] = llvm.getelementptr [[VAR_109_]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
// CHECK:           llvm.store [[VAR_104_]], [[VAR_116_]] : f32, !llvm.ptr
// CHECK:           [[VAR_117_:%.+]] = llvm.extractvalue [[VAR_77_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_118_:%.+]] = llvm.bitcast [[VAR_117_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_119_:%.+]] = llvm.bitcast [[VAR_95_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.bitcast [[VAR_109_]] : !llvm.ptr to !llvm.ptr
// CHECK:           [[VAR_121_:%.+]] = llvm.call @zdnnx_softmax([[VAR_119_]], [[VAR_118_]], [[VAR_106_]], [[VAR_120_]]) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @_mlir_ciface_test_zlow_softmax_constant_shape() attributes {llvm.emit_c_interface} {
// CHECK:           llvm.call @test_zlow_softmax_constant_shape() : () -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
// CHECK:           llvm.return [[VAR_1_1_]] : !llvm.ptr
// CHECK:         }

}

