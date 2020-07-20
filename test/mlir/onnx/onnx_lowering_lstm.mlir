// RUN: onnx-mlir-opt --shape-inference --lower-frontend %s -split-input-file | FileCheck %s

func @test_lstm_general_computation(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

  // CHECK-DAG: [[ACCESS_BY_OFFSET_MAP:#.+]] = affine_map<(d0)[s0, s1] -> (d0 + s0 * s1)>

  // CHECK-LABEL: @test_lstm_general_computation

  // CHECK:  [[CELL_STATE:%.+]] = alloc() : memref<1x3x3xf32>
  // CHECK:  [[HIDDEN_STATE:%.+]] = alloc() : memref<1x3x3xf32>
  // CHECK:  {{.*}} = constant unit

  // CHECK:  [[INITIAL_VALUE:%.+]] = constant 0.000000e+00 : f32
  // CHECK:  [[INITIALIZE_LOOPS:%.+]]:3 = krnl.define_loops 3
  // CHECK:  krnl.iterate([[INITIALIZE_LOOPS]]#0, [[INITIALIZE_LOOPS]]#1, [[INITIALIZE_LOOPS]]#2) with ([[INITIALIZE_LOOPS]]#0 -> %arg3 = 0 to 1, [[INITIALIZE_LOOPS]]#1 -> %arg4 = 0 to 3, [[INITIALIZE_LOOPS]]#2 -> %arg5 = 0 to 3) {
  // CHECK:    affine.store [[INITIAL_VALUE]], [[HIDDEN_STATE]][%arg3, %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:    affine.store [[INITIAL_VALUE]], [[CELL_STATE]][%arg3, %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:  }

  // CHECK:  [[SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:  krnl.iterate([[SEQUENCE_LOOPS]]) with ([[SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:    {{.*}} = constant 0 : index
  // CHECK:    {{.*}} = constant 3 : index
  // CHECK:    {{.*}} = constant 0 : index
  // CHECK:    {{.*}} = constant 1 : index
  // CHECK:    {{.*}} = constant 2 : index
  // CHECK:    {{.*}} = constant 3 : index
  // CHECK:    {{.*}} = constant 4 : index
  // CHECK:    {{.*}} = constant 5 : index
  // CHECK:    {{.*}} = constant 6 : index
  // CHECK:    {{.*}} = constant 7 : index
  // CHECK:    [[DATA_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:    krnl.iterate([[DATA_LOOPS]]#0, [[DATA_LOOPS]]#1) with ([[DATA_LOOPS]]#0 -> %arg4 = 0 to 3, [[DATA_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:      [[hCt:%.+]] = alloc() : memref<f32>
  // CHECK:      [[Ot:%.+]] = alloc() : memref<f32>
  // CHECK:      [[ct:%.+]] = alloc() : memref<f32>
  // CHECK:      [[Ft:%.+]] = alloc() : memref<f32>
  // CHECK:      [[It:%.+]] = alloc() : memref<f32>
  // CHECK:      [[Ht1_LOAD:%.+]] = affine.load [[HIDDEN_STATE]][%c0, %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      [[Ct1_LOAD:%.+]] = affine.load [[CELL_STATE]][%c0, %arg4, %arg5] : memref<1x3x3xf32>

  // CHECK:      [[ZERO_FLOAT:%.+]] = constant 0.000000e+00 : f32
  // CHECK:      [[XtWi_GEMM:%.+]] = alloc() : memref<f32>
  // CHECK:      affine.store [[ZERO_FLOAT]], [[XtWi_GEMM]][] : memref<f32>
  // CHECK:      [[Ht1Ri_GEMM:%.+]] = alloc() : memref<f32>
  // CHECK:      affine.store [[ZERO_FLOAT]], [[Ht1Ri_GEMM]][] : memref<f32>
  // CHECK:      [[XtWo_GEMM:%.+]] = alloc() : memref<f32>
  // CHECK:      affine.store [[ZERO_FLOAT]], [[XtWo_GEMM]][] : memref<f32>
  // CHECK:      [[Ht1Ro_GEMM:%.+]] = alloc() : memref<f32>
  // CHECK:      affine.store [[ZERO_FLOAT]], [[Ht1Ro_GEMM]][] : memref<f32>
  // CHECK:      [[XtWf_GEMM:%.+]] = alloc() : memref<f32>
  // CHECK:      affine.store [[ZERO_FLOAT]], [[XtWf_GEMM]][] : memref<f32>
  // CHECK:      [[Ht1Rf_GEMM:%.+]] = alloc() : memref<f32>
  // CHECK:      affine.store [[ZERO_FLOAT]], [[Ht1Rf_GEMM]][] : memref<f32>
  // CHECK:      [[XtWc_GEMM:%.+]] = alloc() : memref<f32>
  // CHECK:      affine.store [[ZERO_FLOAT]], [[XtWc_GEMM]][] : memref<f32>
  // CHECK:      [[Ht1Rc_GEMM:%.+]] = alloc() : memref<f32>
  // CHECK:      affine.store [[ZERO_FLOAT]], [[Ht1Rc_GEMM]][] : memref<f32>

  // CHECK:      [[REDUCTION_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:      krnl.iterate([[REDUCTION_LOOPS]]) with ([[REDUCTION_LOOPS]] -> %arg6 = 0 to 2) {
  // CHECK:        [[INPUT_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[%c0_1, %c3]
  // CHECK:        [[OUTPUT_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[%c1, %c3]
  // CHECK:        [[FORGET_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[%c2, %c3]
  // CHECK:        [[CELL_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[%c3_2, %c3]
  // CHECK:        [[Xt_LOAD:%.+]] = affine.load %arg0[%arg3, %arg4, %arg6] : memref<4x3x2xf32>

  // CHECK:        [[Wi_LOAD:%.+]] = affine.load %arg1[%c0, [[INPUT_HIDDEN_INDEX]], %arg6] : memref<1x12x2xf32>
  // CHECK:        {{.*}} = mulf [[Xt_LOAD]], [[Wi_LOAD]] : f32
  // CHECK:        {{.*}} = affine.load [[XtWi_GEMM]][] : memref<f32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        affine.store {{.*}}, [[XtWi_GEMM]][] : memref<f32>

  // CHECK:        [[Ri_LOAD:%.+]] = affine.load %arg2[%c0, [[INPUT_HIDDEN_INDEX]], %arg6] : memref<1x12x3xf32>
  // CHECK:        {{.*}} = mulf [[Ht1_LOAD]], [[Ri_LOAD]] : f32
  // CHECK:        {{.*}} = affine.load [[Ht1Ri_GEMM]][] : memref<f32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        affine.store {{.*}}, [[Ht1Ri_GEMM]][] : memref<f32>

  // CHECK:        [[Wo_LOAD:%.+]] = affine.load %arg1[%c0, [[OUTPUT_HIDDEN_INDEX]], %arg6] : memref<1x12x2xf32>
  // CHECK:        {{.*}} = mulf [[Xt_LOAD]], [[Wo_LOAD]] : f32
  // CHECK:        {{.*}} = affine.load [[XtWo_GEMM]][] : memref<f32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        affine.store {{.*}}, [[XtWo_GEMM]][] : memref<f32>

  // CHECK:        [[Ro_LOAD:%.+]] = affine.load %arg2[%c0, [[OUTPUT_HIDDEN_INDEX]], %arg6] : memref<1x12x3xf32>
  // CHECK:        {{.*}} = mulf [[Ht1_LOAD]], [[Ro_LOAD]] : f32
  // CHECK:        {{.*}} = affine.load [[Ht1Ro_GEMM]][] : memref<f32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        affine.store {{.*}}, [[Ht1Ro_GEMM]][] : memref<f32>

  // CHECK:        [[Wf_LOAD:%.+]] = affine.load %arg1[%c0, [[FORGET_HIDDEN_INDEX]], %arg6] : memref<1x12x2xf32>
  // CHECK:        {{.*}} = mulf [[Xt_LOAD]], [[Wf_LOAD]] : f32
  // CHECK:        {{.*}} = affine.load [[XtWf_GEMM]][] : memref<f32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        affine.store {{.*}}, [[XtWf_GEMM]][] : memref<f32>

  // CHECK:        [[Rf_LOAD:%.+]] = affine.load %arg2[%c0, [[FORGET_HIDDEN_INDEX]], %arg6] : memref<1x12x3xf32>
  // CHECK:        {{.*}} = mulf [[Ht1_LOAD]], [[Rf_LOAD]] : f32
  // CHECK:        {{.*}} = affine.load [[Ht1Rf_GEMM]][] : memref<f32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        affine.store {{.*}}, [[Ht1Rf_GEMM]][] : memref<f32>

  // CHECK:        [[Wc_LOAD:%.+]] = affine.load %arg1[%c0, [[CELL_HIDDEN_INDEX]], %arg6] : memref<1x12x2xf32>
  // CHECK:        {{.*}} = mulf [[Xt_LOAD]], [[Wc_LOAD]] : f32
  // CHECK:        {{.*}} = affine.load [[XtWc_GEMM]][] : memref<f32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        affine.store {{.*}}, [[XtWc_GEMM]][] : memref<f32>

  // CHECK:        [[Rc_LOAD:%.+]] = affine.load %arg2[%c0, [[CELL_HIDDEN_INDEX]], %arg6] : memref<1x12x3xf32>
  // CHECK:        {{.*}} = mulf [[Ht1_LOAD]], [[Rc_LOAD]] : f32
  // CHECK:        {{.*}} = affine.load [[Ht1Rc_GEMM]][] : memref<f32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        affine.store {{.*}}, [[Ht1Rc_GEMM]][] : memref<f32>
  // CHECK:      }

  // CHECK:      [[XtWi_LOAD:%.+]] = affine.load [[XtWi_GEMM]][] : memref<f32>
  // CHECK:      [[Ht1Ri_LOAD:%.+]] = affine.load [[Ht1Ri_GEMM]][] : memref<f32>
  // CHECK:      [[It_OUTPUT:%.+]] = addf [[XtWi_LOAD]], [[Ht1Ri_LOAD]] : f32

  // CHECK:      [[SIGMOID_INPUT:%.+]] = alloc() : memref<f32>
  // CHECK:      affine.store [[It_OUTPUT]], [[SIGMOID_INPUT]][] : memref<f32>
  // CHECK:      {{.*}} = affine.load [[SIGMOID_INPUT]][] : memref<f32>
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = constant 1.000000e+00 : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}}: f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      affine.store {{.*}}, [[It]][] : memref<f32>
  // CHECK:      [[It_LOAD:%.+]] = affine.load [[It]][] : memref<f32>

  // CHECK:      [[XtWf_LOAD:%.+]] = affine.load [[XtWf_GEMM]][] : memref<f32>
  // CHECK:      [[Ht1Rf_LOAD:%.+]] = affine.load [[Ht1Rf_GEMM]][] : memref<f32>
  // CHECK:      [[Ft_OUTPUT:%.+]] = addf [[XtWf_LOAD]], [[Ht1Rf_LOAD]] : f32

  // CHECK:      [[SIGMOID_FORGET:%.+]] = alloc() : memref<f32>
  // CHECK:      affine.store [[Ft_OUTPUT]], [[SIGMOID_FORGET]][] : memref<f32>
  // CHECK:      {{.*}} = affine.load [[SIGMOID_FORGET]][] : memref<f32>
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = constant 1.000000e+00 : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}}: f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      affine.store {{.*}}, [[Ft]][] : memref<f32>
  // CHECK:      [[Ft_LOAD:%.+]] = affine.load [[Ft]][] : memref<f32>

  // CHECK:      [[XtWc_LOAD:%.+]] = affine.load [[XtWc_GEMM]][] : memref<f32>
  // CHECK:      [[Ht1Rc_LOAD:%.+]] = affine.load [[Ht1Rc_GEMM]][] : memref<f32>
  // CHECK:      [[ct_OUTPUT:%.+]] = addf [[XtWc_LOAD]], [[Ht1Rc_LOAD]] : f32

  // CHECK:      [[TANH_CELL:%.+]] = alloc() : memref<f32>
  // CHECK:      affine.store [[ct_OUTPUT]], [[TANH_CELL]][] : memref<f32>
  // CHECK:      {{.*}} = affine.load [[TANH_CELL]][] : memref<f32>
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      affine.store {{.*}}, [[ct]][] : memref<f32>
  // CHECK:      [[ct_LOAD:%.+]] = affine.load [[ct]][] : memref<f32>

  // CHECK:      [[FtCt1:%.+]] = mulf [[Ft_LOAD]], [[Ct1_LOAD]] : f32
  // CHECK:      [[Itct:%.+]] = mulf [[It_LOAD]], [[ct_LOAD]] : f32
  // CHECK:      [[Ct:%.+]] = addf [[FtCt1]], [[Itct]] : f32
  // CHECK:      affine.store [[Ct]], [[CELL_STATE]][%c0, %arg4, %arg5] : memref<1x3x3xf32>

  // CHECK:      [[XtWo_LOAD:%.+]] = affine.load [[XtWo_GEMM]][] : memref<f32>
  // CHECK:      [[Ht1Ro_LOAD:%.+]] = affine.load [[Ht1Ro_GEMM]][] : memref<f32>
  // CHECK:      [[Ot_OUTPUT:%.+]] = addf [[XtWo_LOAD]], [[Ht1Ro_LOAD]] : f32

  // CHECK:      [[SIGMOID_OUTPUT:%.+]] = alloc() : memref<f32>
  // CHECK:      affine.store [[Ot_OUTPUT]], [[SIGMOID_OUTPUT]][] : memref<f32>
  // CHECK:      {{.*}} = affine.load [[SIGMOID_OUTPUT]][] : memref<f32>
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = constant 1.000000e+00 : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}}: f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      affine.store {{.*}}, [[Ot]][] : memref<f32>
  // CHECK:      [[Ot_LOAD:%.+]] = affine.load [[Ot]][] : memref<f32>

  // CHECK:      [[TANH_HIDDEN:%.+]] = alloc() : memref<f32>
  // CHECK:      affine.store [[Ct]], [[TANH_HIDDEN]][] : memref<f32>
  // CHECK:      {{.*}} = affine.load [[TANH_HIDDEN]][] : memref<f32>
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      affine.store {{.*}}, [[hCt]][] : memref<f32>
  // CHECK:      [[hCt_LOAD:%.+]] = affine.load [[hCt]][] : memref<f32>

  // CHECK:      [[Ht:%.+]] = mulf [[Ot_LOAD]], [[hCt_LOAD]] : f32
  // CHECK:      affine.store [[Ht]], [[HIDDEN_STATE]][%c0, %arg4, %arg5] : memref<1x3x3xf32>

  // CHECK:      dealloc [[XtWi_GEMM]] : memref<f32>
  // CHECK:      dealloc [[XtWo_GEMM]] : memref<f32>
  // CHECK:      dealloc [[XtWf_GEMM]] : memref<f32>
  // CHECK:      dealloc [[XtWc_GEMM]] : memref<f32>
  // CHECK:      dealloc [[Ht1Ri_GEMM]] : memref<f32>
  // CHECK:      dealloc [[Ht1Ro_GEMM]] : memref<f32>
  // CHECK:      dealloc [[Ht1Rf_GEMM]] : memref<f32>
  // CHECK:      dealloc [[Ht1Rc_GEMM]] : memref<f32>
  // CHECK:      dealloc [[It]] : memref<f32>
  // CHECK:      dealloc [[Ft]] : memref<f32>
  // CHECK:      dealloc [[ct]] : memref<f32>
  // CHECK:      dealloc [[Ot]] : memref<f32>
  // CHECK:      dealloc [[hCt]] : memref<f32>
  // CHECK:    }
  // CHECK:  }
  // CHECK:  dealloc [[CELL_STATE]] : memref<1x3x3xf32>
  // CHECK:  return [[HIDDEN_STATE]] : memref<1x3x3xf32>
}

// -----

func @test_lstm_reverse_mode(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : i64, direction = "reverse"} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

  // CHECK-DAG: [[REVERSE_IV_MAP1:#.+]] = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>

  // CHECK-LABEL: @test_lstm_reverse_mode

  // CHECK:  [[REVERSE_SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:  krnl.iterate([[REVERSE_SEQUENCE_LOOPS]]) with ([[REVERSE_SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:  %[[SEQUENCE_LEN:.+]] = constant 4 : index
  // CHECK:  %[[REVERSE_SEQUENCE_IV:.+]] = affine.apply [[REVERSE_IV_MAP1]](%arg3)[%[[SEQUENCE_LEN]]{{]}}
  // CHECK:  [[Xt_LOAD:%.+]] = affine.load %arg0[%[[REVERSE_SEQUENCE_IV]], {{.*}}, {{.*}}] : memref<4x3x2xf32>
}

// -----

func @test_lstm_bidirectional_mode(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : i64, direction = "bidirectional"} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

  // CHECK-DAG: [[REVERSE_IV_MAP1:#.+]] = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>

  // CHECK-LABEL: @test_lstm_bidirectional_mode

  // CHECK:  [[SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:  krnl.iterate([[SEQUENCE_LOOPS]]) with ([[SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:  [[Xt_LOAD:%.+]] = affine.load %arg0[%arg3, {{.*}}, {{.*}}] : memref<4x3x2xf32>

  // CHECK:  [[REVERSE_SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:  krnl.iterate([[REVERSE_SEQUENCE_LOOPS]]) with ([[REVERSE_SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:  %[[SEQUENCE_LEN:.+]] = constant 4 : index
  // CHECK:  %[[REVERSE_SEQUENCE_IV:.+]] = affine.apply [[REVERSE_IV_MAP1]](%arg3)[%[[SEQUENCE_LEN]]{{]}}
  // CHECK:  [[Xt_LOAD:%.+]] = affine.load %arg0[%[[REVERSE_SEQUENCE_IV]], {{.*}}, {{.*}}] : memref<4x3x2xf32>
}
