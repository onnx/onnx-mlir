// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file

// FIXME: turn this on
// | FileCheck %s

/// Check GRU with three required inputs (X, W, R). The optional inputs are default.
/// Also check the equation for 'ht' when linear_before_reset = 0 (default)
func private @test_gru_general_computation(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x9x2xf32>, %arg2: tensor<1x9x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x9x2xf32>, tensor<1x9x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_gru_general_computation
  // CHECK: [[RES:%.+]] = memref.alloc() : memref<1x3x3xf32>

  /// Check initialize loop.
  // CHECK: [[INITIAL_VAL:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[DEF_LOOPS_INIT:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS_INIT]]#0, [[DEF_LOOPS_INIT]]#1, [[DEF_LOOPS_INIT]]#2) with ([[DEF_LOOPS_INIT]]#0 -> %arg3 = 0 to 1, [[DEF_LOOPS_INIT]]#1 -> %arg4 = 0 to 3, [[DEF_LOOPS_INIT]]#2 -> %arg5 = 0 to 3) {
  // CHECK:   krnl.store [[INITIAL_VAL]], [[RES]][%arg3, %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK: }

  /// Check main loop.
  // CHECK: [[SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK: krnl.iterate([[SEQUENCE_LOOPS]]) with ([[SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:   [[rhrHMemRef:%.+]] = memref.alloc() : memref<3x3xf32>
  // CHECK:   [[rhMemRef:%.+]] = memref.alloc() : memref<3x3xf32>
  // CHECK:   [[xwHMemRef:%.+]] = memref.alloc() : memref<3x3xf32>
  // CHECK:   [[ztMemRef:%.+]] = memref.alloc() : memref<3x3xf32>
  // CHECK:   [[htMemRef:%.+]] = memref.alloc() : memref<3x3xf32>
  // CHECK:   [[ZERO_INDEX:%.+]] = constant 0 : index
  // CHECK:   [[INDEX_3:%.+]] = constant 3 : index
  // CHECK:   [[INDEX_0:%.+]] = constant 0 : index
  // CHECK:   [[INDEX_1:%.+]] = constant 1 : index
  // CHECK:   [[INDEX_2:%.+]] = constant 2 : index
  // CHECK:   [[DATA_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[DATA_LOOPS]]#0, [[DATA_LOOPS]]#1) with ([[DATA_LOOPS]]#0 -> %arg4 = 0 to 3, [[DATA_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:     [[rt:%.+]] = memref.alloc() : memref<f32>
  // CHECK:     [[zt:%.+]] = memref.alloc() : memref<f32>

  // CHECK:     [[INITIAL_VAL_0:%.+]] = constant 0.000000e+00 : f32
  // CHECK:     [[XWZt:%.+]] = memref.alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[XWZt]][] : memref<f32>
  // CHECK:     [[HRZt:%.+]] = memref.alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[HRZt]][] : memref<f32>
  // CHECK:     [[XWRt:%.+]] = memref.alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[XWRt]][] : memref<f32>
  // CHECK:     [[HRRt:%.+]] = memref.alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[HRRt]][] : memref<f32>

  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[xwHMemRef]][%arg4, %arg5] : memref<3x3xf32>

  // CHECK:     [[REDUCTION_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[REDUCTION_LOOPS]]) with ([[REDUCTION_LOOPS]] -> %arg6 = 0 to 2) {
  // CHECK:       [[BIAS_MAP_FOR_Z:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_0]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_R:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_1]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_H:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_2]], [[INDEX_3]]]
  // CHECK:       [[Xt:%.+]] = krnl.load %arg0[%arg3, %arg4, %arg6] : memref<4x3x2xf32>

  /// compute Xt*(Wz^T)
  // CHECK:       [[WZt:%.+]] = krnl.load %arg1{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_Z]], %arg6] : memref<1x9x2xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[Xt]], [[WZt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[XWZt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[XWZt]][] : memref<f32>

  /// compute Xt*(Wr^T)
  // CHECK:       [[WRt:%.+]] = krnl.load %arg1{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_R]], %arg6] : memref<1x9x2xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[Xt]], [[WRt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[XWRt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[XWRt]][] : memref<f32>

  /// compute Xt*(Wh^T)
  // CHECK:       [[WHt:%.+]] = krnl.load %arg1{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_H]], %arg6] : memref<1x9x2xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[Xt]], [[WHt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[xwHMemRef]][%arg4, %arg5] : memref<3x3xf32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[xwHMemRef]][%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     }

  // CHECK:     [[REDUCTION_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[REDUCTION_LOOPS]]) with ([[REDUCTION_LOOPS]] -> %arg6 = 0 to 3) {
  // CHECK:       [[BIAS_MAP_FOR_Z:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_0]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_R:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_1]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_H:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_2]], [[INDEX_3]]]
  // CHECK:       [[PREVIOUS_Ht:%.+]] = krnl.load [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg6] : memref<1x3x3xf32>
  /// compute Ht-1*(Rz^T)
  // CHECK:       [[RZt:%.+]] = krnl.load %arg2{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_Z]], %arg6] : memref<1x9x3xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[PREVIOUS_Ht]], [[RZt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[HRZt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[HRZt]][] : memref<f32>

  /// compute Ht-1*(Rr^T)
  // CHECK:       [[RRt:%.+]] = krnl.load %arg2{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_R]], %arg6] : memref<1x9x3xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[PREVIOUS_Ht]], [[RRt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[HRRt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[HRRt]][] : memref<f32>
  // CHECK:     }

  /// compute zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
  // CHECK:     [[LOAD_XWZt:%.+]] = krnl.load [[XWZt]][] : memref<f32>
  // CHECK:     [[LOAD_HRZt:%.+]] = krnl.load [[HRZt]][] : memref<f32>
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_XWZt]], [[LOAD_HRZt]] : f32
  /// apply activation f = sigmoid
  // CHECK:     {{.*}} = memref.alloc() : memref<f32>
  // CHECK:     krnl.store [[ADD]], {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = krnl.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = constant 1.000000e+00 : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = math.exp {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     krnl.store {{.*}}, [[zt]][] : memref<f32>
  // CHECK:     krnl.store {{.*}}, [[ztMemRef]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  /// compute rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
  // CHECK:     [[LOAD_XWRt:%.+]] = krnl.load [[XWRt]][] : memref<f32>
  // CHECK:     [[LOAD_HRRt:%.+]] = krnl.load [[HRRt]][] : memref<f32>
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_XWRt]], [[LOAD_HRRt]] : f32
  /// apply activation f = sigmoid
  // CHECK:     {{.*}} = memref.alloc() : memref<f32>
  // CHECK:     krnl.store [[ADD]], {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = krnl.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = constant 1.000000e+00 : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = math.exp {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     krnl.store {{.*}}, [[rt]][] : memref<f32>
  // CHECK:     [[LOAD_rt:%.+]] = krnl.load [[rt]][] : memref<f32>

  // COM: 'rt (.) Ht-1'
  // CHECK:     [[LOAD_ht:%.+]] = krnl.load [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     [[RtHt:%.+]] = mulf [[LOAD_rt]], [[LOAD_ht]] : f32
  // CHECK:     krnl.store [[RtHt]], [[rhMemRef]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  // CHECK:     memref.dealloc [[XWZt]] : memref<f32>
  // CHECK:     memref.dealloc [[XWRt]] : memref<f32>
  // CHECK:     memref.dealloc [[HRZt]] : memref<f32>
  // CHECK:     memref.dealloc [[HRRt]] : memref<f32>
  // CHECK:   }

  // COM: compute '(rt (.) Ht-1)*(Rh^T)'
  // CHECK:   [[HT_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[HT_LOOPS]]#0, [[HT_LOOPS]]#1) with ([[HT_LOOPS]]#0 -> %arg4 = 0 to 3, [[HT_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:     [[BIAS_MAP_FOR_H:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_2]], [[INDEX_3]]]
  // CHECK:     [[INITIAL_VAL:%.+]] = constant 0.000000e+00 : f32
  // CHECK:     krnl.store [[INITIAL_VAL]], [[rhrHMemRef]][%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     [[REDUCTION_LOOPS_1:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[REDUCTION_LOOPS_1]]) with ([[REDUCTION_LOOPS_1]] -> %arg6 = 0 to 3) {
  // CHECK:       [[LOAD_RtHt:%.+]] = krnl.load [[rhMemRef]][%arg4, %arg6] : memref<3x3xf32>
  // CHECK:       [[LOAD_RHt:%.+]] = krnl.load %arg2{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_H]], %arg6] : memref<1x9x3xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[LOAD_RtHt]], [[LOAD_RHt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[rhrHMemRef]][%arg4, %arg5] : memref<3x3xf32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[rhrHMemRef]][%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     }
  // CHECK:   }

  // CHECK:   [[GATE_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[GATE_LOOPS]]#0, [[GATE_LOOPS]]#1) with ([[GATE_LOOPS]]#0 -> %arg4 = 0 to 3, [[GATE_LOOPS]]#1 -> %arg5 = 0 to 3) {

  // COM: compute  ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) since linear_before_reset = 0 (default)
  // CHECK:     [[ht:%.+]] = memref.alloc() : memref<f32>
  // CHECK:     [[LOAD_XWHt:%.+]] = krnl.load [[xwHMemRef]][%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     [[LOAD_HRHt:%.+]] = krnl.load [[rhrHMemRef]][%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_XWHt]], [[LOAD_HRHt]] : f32
  /// apply activation g = tanh
  // CHECK:     krnl.store [[ADD]], {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = krnl.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 1.000000e+00 : f32
  // CHECK:     {{.*}} = constant 2.000000e+00 : f32
  // CHECK:     {{.*}} = mulf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = negf {{.*}} : f32
  // CHECK:     {{.*}} = math.exp {{.*}} : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = math.exp {{.*}} : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = cmpf oge, {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = select {{.*}}, {{.*}}, {{.*}} : f32
  // CHECK:     krnl.store {{.*}}, [[ht]][] : memref<f32>
  // CHECK:     [[LOAD_ht:%.+]] = krnl.load [[ht]][] : memref<f32>

  // COM: compute  Ht = (1 - zt) (.) ht + zt (.) Ht-1
  // CHECK:     [[LOAD_zt:%.+]] = krnl.load [[ztMemRef]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     [[PREVIOUS_Ht:%.+]] = krnl.load [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     [[ONE:%.+]] = constant 1.000000e+00 : f32
  // CHECK:     [[SUB:%.+]] = subf [[ONE]], [[LOAD_zt]] : f32
  // CHECK:     [[MUL:%.+]] = mulf [[SUB]], [[LOAD_ht]] : f32
  // CHECK:     [[MUL_1:%.+]] = mulf [[LOAD_zt]], [[PREVIOUS_Ht]] : f32
  // CHECK:     [[ADD:%.+]] = addf [[MUL]], [[MUL_1]] : f32
  // CHECK:     krnl.store [[ADD]], [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:   }
  // CHECK:   memref.dealloc [[htMemRef]] : memref<3x3xf32>
  // CHECK:   memref.dealloc [[ztMemRef]] : memref<3x3xf32>
  // CHECK:   memref.dealloc [[xwHMemRef]] : memref<3x3xf32>
  // CHECK:   memref.dealloc [[rhMemRef]] : memref<3x3xf32>
  // CHECK:   memref.dealloc [[rhrHMemRef]] : memref<3x3xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<1x3x3xf32>
}

// -----

/// GRU with three required inputs (X, W, R). The optional inputs are default.
/// Check the equation for 'ht' when linear_before_reset !=0.
func private @test_gru_linear_before_reset(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x9x2xf32>, %arg2: tensor<1x9x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64, linear_before_reset = 1 : si64} : (tensor<4x3x2xf32>, tensor<1x9x2xf32>, tensor<1x9x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_gru_linear_before_reset
  // CHECK: [[RES:%.+]] = memref.alloc() : memref<1x3x3xf32>

  /// Check initialize loop.
  // CHECK: [[INITIAL_VAL:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[DEF_LOOPS_INIT:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS_INIT]]#0, [[DEF_LOOPS_INIT]]#1, [[DEF_LOOPS_INIT]]#2) with ([[DEF_LOOPS_INIT]]#0 -> %arg3 = 0 to 1, [[DEF_LOOPS_INIT]]#1 -> %arg4 = 0 to 3, [[DEF_LOOPS_INIT]]#2 -> %arg5 = 0 to 3) {
  // CHECK:   krnl.store [[INITIAL_VAL]], [[RES]][%arg3, %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK: }

  /// Check main loop.
  // CHECK: [[SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK: krnl.iterate([[SEQUENCE_LOOPS]]) with ([[SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:   [[ztMemRef:%.+]] = memref.alloc() : memref<3x3xf32>
  // CHECK:   [[htMemRef:%.+]] = memref.alloc() : memref<3x3xf32>
  // CHECK:   [[ZERO_INDEX:%.+]] = constant 0 : index
  // CHECK:   [[INDEX_3:%.+]] = constant 3 : index
  // CHECK:   [[INDEX_0:%.+]] = constant 0 : index
  // CHECK:   [[INDEX_1:%.+]] = constant 1 : index
  // CHECK:   [[INDEX_2:%.+]] = constant 2 : index
  // CHECK:   [[DATA_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[DATA_LOOPS]]#0, [[DATA_LOOPS]]#1) with ([[DATA_LOOPS]]#0 -> %arg4 = 0 to 3, [[DATA_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:     [[ht:%.+]] = memref.alloc() : memref<f32>
  // CHECK:     [[rt:%.+]] = memref.alloc() : memref<f32>
  // CHECK:     [[zt:%.+]] = memref.alloc() : memref<f32>

  // CHECK:     [[INITIAL_VAL_0:%.+]] = constant 0.000000e+00 : f32
  // CHECK:     [[XWZt:%.+]] = memref.alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[XWZt]][] : memref<f32>
  // CHECK:     [[HRZt:%.+]] = memref.alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[HRZt]][] : memref<f32>
  // CHECK:     [[XWRt:%.+]] = memref.alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[XWRt]][] : memref<f32>
  // CHECK:     [[HRRt:%.+]] = memref.alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[HRRt]][] : memref<f32>
  // CHECK:     [[XWHt:%.+]] = memref.alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[XWHt]][] : memref<f32>
  // CHECK:     [[HRHt:%.+]] = memref.alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[HRHt]][] : memref<f32>

  // CHECK:     [[REDUCTION_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[REDUCTION_LOOPS]]) with ([[REDUCTION_LOOPS]] -> %arg6 = 0 to 2) {
  // CHECK:       [[BIAS_MAP_FOR_Z:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_0]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_R:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_1]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_H:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_2]], [[INDEX_3]]]
  // CHECK:       [[Xt:%.+]] = krnl.load %arg0[%arg3, %arg4, %arg6] : memref<4x3x2xf32>

  /// compute Xt*(Wz^T)
  // CHECK:       [[WZt:%.+]] = krnl.load %arg1{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_Z]], %arg6] : memref<1x9x2xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[Xt]], [[WZt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[XWZt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[XWZt]][] : memref<f32>

  /// compute Xt*(Wr^T)
  // CHECK:       [[WRt:%.+]] = krnl.load %arg1{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_R]], %arg6] : memref<1x9x2xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[Xt]], [[WRt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[XWRt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[XWRt]][] : memref<f32>

  /// compute Xt*(Wh^T)
  // CHECK:       [[WHt:%.+]] = krnl.load %arg1{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_H]], %arg6] : memref<1x9x2xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[Xt]], [[WHt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[XWHt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[XWHt]][] : memref<f32>
  // CHECK:     }

  // CHECK:     [[REDUCTION_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[REDUCTION_LOOPS]]) with ([[REDUCTION_LOOPS]] -> %arg6 = 0 to 3) {
  // CHECK:       [[BIAS_MAP_FOR_Z:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_0]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_R:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_1]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_H:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_2]], [[INDEX_3]]]
  // CHECK:       [[PREVIOUS_Ht:%.+]] = krnl.load [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg6] : memref<1x3x3xf32>
  /// compute Ht-1*(Rz^T)
  // CHECK:       [[RZt:%.+]] = krnl.load %arg2{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_Z]], %arg6] : memref<1x9x3xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[PREVIOUS_Ht]], [[RZt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[HRZt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[HRZt]][] : memref<f32>

  /// compute Ht-1*(Rr^T)
  // CHECK:       [[RRt:%.+]] = krnl.load %arg2{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_R]], %arg6] : memref<1x9x3xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[PREVIOUS_Ht]], [[RRt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[HRRt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[HRRt]][] : memref<f32>

  /// compute Ht-1*(Rh^T)
  // CHECK:       [[RHt:%.+]] = krnl.load %arg2{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_H]], %arg6] : memref<1x9x3xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[PREVIOUS_Ht]], [[RHt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[HRHt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[HRHt]][] : memref<f32>
  // CHECK:     }

  /// compute zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
  // CHECK:     [[LOAD_XWZt:%.+]] = krnl.load [[XWZt]][] : memref<f32>
  // CHECK:     [[LOAD_HRZt:%.+]] = krnl.load [[HRZt]][] : memref<f32>
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_XWZt]], [[LOAD_HRZt]] : f32
  /// apply activation f = sigmoid
  // CHECK:     {{.*}} = memref.alloc() : memref<f32>
  // CHECK:     krnl.store [[ADD]], {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = krnl.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = constant 1.000000e+00 : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = math.exp {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     krnl.store {{.*}}, [[zt]][] : memref<f32>
  // CHECK:     krnl.store {{.*}}, [[ztMemRef]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  /// compute rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
  // CHECK:     [[LOAD_XWRt:%.+]] = krnl.load [[XWRt]][] : memref<f32>
  // CHECK:     [[LOAD_HRRt:%.+]] = krnl.load [[HRRt]][] : memref<f32>
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_XWRt]], [[LOAD_HRRt]] : f32
  /// apply activation f = sigmoid
  // CHECK:     {{.*}} = memref.alloc() : memref<f32>
  // CHECK:     krnl.store [[ADD]], {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = krnl.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = constant 1.000000e+00 : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = math.exp {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     krnl.store {{.*}}, [[rt]][] : memref<f32>
  // CHECK:     [[LOAD_rt:%.+]] = krnl.load [[rt]][] : memref<f32>

  /// compute ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) since linear_before_reset != 0
  // CHECK:     [[LOAD_XWHt:%.+]] = krnl.load [[XWHt]][] : memref<f32>
  // CHECK:     [[LOAD_HRHt:%.+]] = krnl.load [[HRHt]][] : memref<f32>
  // CHECK:     [[MUL_rt_HRHt:%.+]] = mulf [[LOAD_rt]], [[LOAD_HRHt]] : f32
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_XWHt]], [[MUL_rt_HRHt]] : f32
  /// apply activation g = tanh
  // CHECK:     {{.*}} = memref.alloc() : memref<f32>
  // CHECK:     krnl.store [[ADD]], {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = krnl.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 1.000000e+00 : f32
  // CHECK:     {{.*}} = constant 2.000000e+00 : f32
  // CHECK:     {{.*}} = mulf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = negf {{.*}} : f32
  // CHECK:     {{.*}} = math.exp {{.*}} : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = math.exp {{.*}} : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = cmpf oge, {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = select {{.*}}, {{.*}}, {{.*}} : f32
  // CHECK:     krnl.store {{.*}}, [[ht]][] : memref<f32>
  // CHECK:     [[LOAD_ht:%.+]] = krnl.load [[ht]][] : memref<f32>
  // CHECK:     krnl.store [[LOAD_ht]], [[htMemRef]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  // CHECK:     memref.dealloc [[XWZt]] : memref<f32>
  // CHECK:     memref.dealloc [[XWRt]] : memref<f32>
  // CHECK:     memref.dealloc [[XWHt]] : memref<f32>
  // CHECK:     memref.dealloc [[HRZt]] : memref<f32>
  // CHECK:     memref.dealloc [[HRRt]] : memref<f32>
  // CHECK:     memref.dealloc [[HRHt]] : memref<f32>
  // CHECK:     memref.dealloc [[zt]] : memref<f32>
  // CHECK:     memref.dealloc [[rt]] : memref<f32>
  // CHECK:     memref.dealloc [[ht]] : memref<f32>
  // CHECK:   }

  // CHECK:   [[GATE_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[GATE_LOOPS]]#0, [[GATE_LOOPS]]#1) with ([[GATE_LOOPS]]#0 -> %arg4 = 0 to 3, [[GATE_LOOPS]]#1 -> %arg5 = 0 to 3) {
  /// compute  Ht = (1 - zt) (.) ht + zt (.) Ht-1
  // CHECK:     [[LOAD_ht:%.+]] = krnl.load [[htMemRef]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     [[LOAD_zt:%.+]] = krnl.load [[ztMemRef]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     [[PREVIOUS_Ht:%.+]] = krnl.load [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     [[ONE:%.+]] = constant 1.000000e+00 : f32
  // CHECK:     [[SUB:%.+]] = subf [[ONE]], [[LOAD_zt]] : f32
  // CHECK:     [[MUL:%.+]] = mulf [[SUB]], [[LOAD_ht]] : f32
  // CHECK:     [[MUL_1:%.+]] = mulf [[LOAD_zt]], [[PREVIOUS_Ht]] : f32
  // CHECK:     [[ADD:%.+]] = addf [[MUL]], [[MUL_1]] : f32
  // CHECK:     krnl.store [[ADD]], [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:   }
  // CHECK:    memref.dealloc [[htMemRef]] : memref<3x3xf32>
  // CHECK:    memref.dealloc [[ztMemRef]] : memref<3x3xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<1x3x3xf32>
}

// -----

/// Check GRU with three required inputs (X, W, R), and bias input.
func private @test_gru_with_bias(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x9x2xf32>, %arg2: tensor<1x9x3xf32>, %arg3: tensor<1x18xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x9x2xf32>, tensor<1x9x3xf32>, tensor<1x18xf32>, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_gru_with_bias

  // CHECK: [[LOAD_WZ_BIAS:%.+]] = krnl.load %arg3[{{.*}}, {{.*}}] : memref<1x18xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_WZ_BIAS]] : f32
  // CHECK: [[LOAD_RZ_BIAS:%.+]] = krnl.load %arg3[{{.*}}, {{.*}}] : memref<1x18xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_RZ_BIAS]] : f32

  // CHECK: [[LOAD_WR_BIAS:%.+]] = krnl.load %arg3[{{.*}}, {{.*}}] : memref<1x18xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_WR_BIAS]] : f32
  // CHECK: [[LOAD_RR_BIAS:%.+]] = krnl.load %arg3[{{.*}}, {{.*}}] : memref<1x18xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_RR_BIAS]] : f32

  // CHECK: [[LOAD_WH_BIAS:%.+]] = krnl.load %arg3[{{.*}}, {{.*}}] : memref<1x18xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_WH_BIAS]] : f32
  // CHECK: [[LOAD_RH_BIAS:%.+]] = krnl.load %arg3[{{.*}}, {{.*}}] : memref<1x18xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_RH_BIAS]] : f32
}

// -----

// Check handling unknown dimensions for GRU by checking the
// correctness of allocating and deallocating memory.
func private @test_gru_unkown_dims_allocation(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x9x?xf32>, %arg2: tensor<1x9x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<?x?x?xf32>, tensor<1x9x?xf32>, tensor<1x9x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: @test_gru_unkown_dims_allocation

  // allocate memory for Hidden (Y_h).
  // CHECK: [[C1_0:%.+]] = constant 1 : index
  // CHECK: [[BATCH_SIZE:%.+]] = memref.dim %arg0, [[C1_0]] : memref<?x?x?xf32>
  // CHECK: [[Y_h:%.+]] = memref.alloc([[BATCH_SIZE]]) : memref<1x?x3xf32>

  // CHECK: return [[Y_h]] : memref<1x?x3xf32>
}

// -----

func private @test_lstm_general_computation(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

  // CHECK-DAG: [[ACCESS_BY_OFFSET_MAP:#.+]] = affine_map<(d0)[s0, s1] -> (d0 + s0 * s1)>
  // CHECK-LABEL: @test_lstm_general_computation

  // CHECK:  [[CELL_STATE:%.+]] = memref.alloc() : memref<1x3x3xf32>
  // CHECK:  [[HIDDEN_STATE:%.+]] = memref.alloc() : memref<1x3x3xf32>
  // CHECK:  {{.*}} = constant unit

  // CHECK:  [[INITIAL_VALUE:%.+]] = constant 0.000000e+00 : f32
  // CHECK:  [[INITIALIZE_LOOPS:%.+]]:3 = krnl.define_loops 3
  // CHECK:  krnl.iterate([[INITIALIZE_LOOPS]]#0, [[INITIALIZE_LOOPS]]#1, [[INITIALIZE_LOOPS]]#2) with ([[INITIALIZE_LOOPS]]#0 -> %arg3 = 0 to 1, [[INITIALIZE_LOOPS]]#1 -> %arg4 = 0 to 3, [[INITIALIZE_LOOPS]]#2 -> %arg5 = 0 to 3) {
  // CHECK:    krnl.store [[INITIAL_VALUE]], [[HIDDEN_STATE]][%arg3, %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:    krnl.store [[INITIAL_VALUE]], [[CELL_STATE]][%arg3, %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:  }

  // CHECK:  [[SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:  krnl.iterate([[SEQUENCE_LOOPS]]) with ([[SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {

  // CHECK:    [[HtRc_GEMM:%.+]] = memref.alloc() : memref<3x3xf32>
  // CHECK:    [[XtWc_GEMM:%.+]] = memref.alloc() : memref<3x3xf32>
  // CHECK:    [[HtRf_GEMM:%.+]] = memref.alloc() : memref<3x3xf32>
  // CHECK:    [[XtWf_GEMM:%.+]] = memref.alloc() : memref<3x3xf32>
  // CHECK:    [[HtRo_GEMM:%.+]] = memref.alloc() : memref<3x3xf32>
  // CHECK:    [[XtWo_GEMM:%.+]] = memref.alloc() : memref<3x3xf32>
  // CHECK:    [[HtRi_GEMM:%.+]] = memref.alloc() : memref<3x3xf32>
  // CHECK:    [[XtWi_GEMM:%.+]] = memref.alloc() : memref<3x3xf32>

  // CHECK:    [[C0_INDEX:%.+]] = constant 0 : index
  // CHECK:    {{.*}} = constant 3 : index
  // CHECK:    {{.*}} = constant 0 : index
  // CHECK:    {{.*}} = constant 1 : index
  // CHECK:    {{.*}} = constant 2 : index
  // CHECK:    {{.*}} = constant 3 : index
  // CHECK:    {{.*}} = constant 4 : index
  // CHECK:    {{.*}} = constant 5 : index
  // CHECK:    {{.*}} = constant 6 : index
  // CHECK:    {{.*}} = constant 7 : index
  // CHECK:    [[MATRIX_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:    krnl.iterate([[MATRIX_LOOPS]]#0, [[MATRIX_LOOPS]]#1) with ([[MATRIX_LOOPS]]#0 -> %arg4 = 0 to 3, [[MATRIX_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:      [[CST0:%.+]] = constant 0.000000e+00 : f32
  // CHECK:      krnl.store [[CST0]], [[XtWi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      krnl.store [[CST0]], [[HtRi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      krnl.store [[CST0]], [[XtWo_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      krnl.store [[CST0]], [[HtRo_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      krnl.store [[CST0]], [[XtWf_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      krnl.store [[CST0]], [[HtRf_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      krnl.store [[CST0]], [[XtWc_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      krnl.store [[CST0]], [[HtRc_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      [[XW_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:      krnl.iterate([[XW_LOOPS]]) with ([[XW_LOOPS]] -> %arg6 = 0 to 2) {
  // CHECK:        [[INPUT_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]
  // CHECK:        [[OUTPUT_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]
  // CHECK:        [[FORGET_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]
  // CHECK:        [[CELL_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]

  // CHECK:        [[Xt_LOAD:%.+]] = krnl.load %arg0[%arg3, %arg4, %arg6] : memref<4x3x2xf32>
  // CHECK:        [[Wi_LOAD:%.+]] = krnl.load %arg1{{\[}}[[C0_INDEX]], [[INPUT_HIDDEN_INDEX]], %arg6] : memref<1x12x2xf32>
  // CHECK:        {{.*}} = mulf [[Xt_LOAD]], [[Wi_LOAD]] : f32
  // CHECK:        {{.*}} = krnl.load [[XtWi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        krnl.store {{.*}}, [[XtWi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  // CHECK:        [[Wo_LOAD:%.+]] = krnl.load %arg1{{\[}}[[C0_INDEX]], [[OUTPUT_HIDDEN_INDEX]], %arg6] : memref<1x12x2xf32>
  // CHECK:        {{.*}} = mulf [[Xt_LOAD]], [[Wo_LOAD]] : f32
  // CHECK:        {{.*}} = krnl.load [[XtWo_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, %26 : f32
  // CHECK:        krnl.store {{.*}}, [[XtWo_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  // CHECK:        [[Wf_LOAD:%.+]] = krnl.load %arg1{{\[}}[[C0_INDEX]], [[FORGET_HIDDEN_INDEX]], %arg6] : memref<1x12x2xf32>
  // CHECK:        {{.*}} = mulf [[Xt_LOAD]], [[Wf_LOAD]] : f32
  // CHECK:        {{.*}} = krnl.load [[XtWf_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, %30 : f32
  // CHECK:        krnl.store {{.*}}, [[XtWf_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  // CHECK:        [[Wc_LOAD:%.+]] = krnl.load %arg1{{\[}}[[C0_INDEX]], [[CELL_HIDDEN_INDEX]], %arg6] : memref<1x12x2xf32>
  // CHECK:        {{.*}} = mulf [[Xt_LOAD]], [[Wc_LOAD]] : f32
  // CHECK:        {{.*}} = krnl.load [[XtWc_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, %34 : f32
  // CHECK:        krnl.store {{.*}}, [[XtWc_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      }
  // CHECK:      [[HR_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:      krnl.iterate([[HR_LOOPS]]) with ([[HR_LOOPS]] -> %arg6 = 0 to 3) {
  // CHECK:        [[INPUT_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]
  // CHECK:        [[OUTPUT_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]
  // CHECK:        [[FORGET_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]
  // CHECK:        [[CELL_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]

  // CHECK:        [[Ht_LOAD:%.+]] = krnl.load %1{{\[}}[[C0_INDEX]], %arg4, %arg6] : memref<1x3x3xf32>

  // CHECK:        [[Ri_LOAD:%.+]] = krnl.load %arg2{{\[}}[[C0_INDEX]], [[INPUT_HIDDEN_INDEX]], %arg6] : memref<1x12x3xf32>
  // CHECK:        {{.*}} = mulf [[Ht_LOAD]], [[Ri_LOAD]] : f32
  // CHECK:        {{.*}} = krnl.load [[HtRi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        krnl.store {{.*}}, [[HtRi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  // CHECK:        [[Ro_LOAD:%.+]] = krnl.load %arg2{{\[}}[[C0_INDEX]], [[OUTPUT_HIDDEN_INDEX]], %arg6] : memref<1x12x3xf32>
  // CHECK:        {{.*}} = mulf [[Ht_LOAD]], [[Ro_LOAD]] : f32
  // CHECK:        {{.*}} = krnl.load [[HtRo_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        krnl.store {{.*}}, [[HtRo_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  // CHECK:        [[Rf_LOAD:%.+]] = krnl.load %arg2{{\[}}[[C0_INDEX]], [[FORGET_HIDDEN_INDEX]], %arg6] : memref<1x12x3xf32>
  // CHECK:        {{.*}} = mulf [[Ht_LOAD]], [[Rf_LOAD]] : f32
  // CHECK:        {{.*}} = krnl.load [[HtRf_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        krnl.store {{.*}}, [[HtRf_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  // CHECK:        [[Rc_LOAD:%.+]] = krnl.load %arg2{{\[}}[[C0_INDEX]], [[CELL_HIDDEN_INDEX]], %arg6] : memref<1x12x3xf32>
  // CHECK:        {{.*}} = mulf [[Ht_LOAD]], [[Rc_LOAD]] : f32
  // CHECK:        {{.*}} = krnl.load [[HtRc_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        krnl.store {{.*}}, [[HtRc_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      } 
  // CHECK:    }

  // CHECK:    [[DATA_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:    krnl.iterate([[DATA_LOOPS]]#0, [[DATA_LOOPS]]#1) with ([[DATA_LOOPS]]#0 -> %arg4 = 0 to 3, [[DATA_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:      [[hCt:%.+]] = memref.alloc() : memref<f32>
  // CHECK:      [[Ot:%.+]] = memref.alloc() : memref<f32>
  // CHECK:      [[ct:%.+]] = memref.alloc() : memref<f32>
  // CHECK:      [[Ft:%.+]] = memref.alloc() : memref<f32>
  // CHECK:      [[It:%.+]] = memref.alloc() : memref<f32>

  // CHECK:      [[Ct1_LOAD:%.+]] = krnl.load [[CELL_STATE]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      [[XtWi_LOAD:%.+]] = krnl.load [[XtWi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      [[HtRi_LOAD:%.+]] = krnl.load [[HtRi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      [[It_OUTPUT:%.+]] = addf [[XtWi_LOAD]], [[HtRi_LOAD]] : f32

  // CHECK:      [[SIGMOID_INPUT:%.+]] = memref.alloc() : memref<f32>
  // CHECK:      krnl.store [[It_OUTPUT]], [[SIGMOID_INPUT]][] : memref<f32>
  // CHECK:      {{.*}} = krnl.load [[SIGMOID_INPUT]][] : memref<f32>
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = constant 1.000000e+00 : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}}: f32
  // CHECK:      {{.*}} = math.exp {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      krnl.store {{.*}}, [[It]][] : memref<f32>
  // CHECK:      [[It_LOAD:%.+]] = krnl.load [[It]][] : memref<f32>

  // CHECK:      [[XtWf_LOAD:%.+]] = krnl.load [[XtWf_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      [[HtRf_LOAD:%.+]] = krnl.load [[HtRf_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      [[Ft_OUTPUT:%.+]] = addf [[XtWf_LOAD]], [[HtRf_LOAD]] : f32

  // CHECK:      [[SIGMOID_FORGET:%.+]] = memref.alloc() : memref<f32>
  // CHECK:      krnl.store [[Ft_OUTPUT]], [[SIGMOID_FORGET]][] : memref<f32>
  // CHECK:      {{.*}} = krnl.load [[SIGMOID_FORGET]][] : memref<f32>
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = constant 1.000000e+00 : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}}: f32
  // CHECK:      {{.*}} = math.exp {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      krnl.store {{.*}}, [[Ft]][] : memref<f32>
  // CHECK:      [[Ft_LOAD:%.+]] = krnl.load [[Ft]][] : memref<f32>

  // CHECK:      [[XtWc_LOAD:%.+]] = krnl.load [[XtWc_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      [[HtRc_LOAD:%.+]] = krnl.load [[HtRc_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      [[ct_OUTPUT:%.+]] = addf [[XtWc_LOAD]], [[HtRc_LOAD]] : f32

  // CHECK:      [[TANH_CELL:%.+]] = memref.alloc() : memref<f32>
  // CHECK:      krnl.store [[ct_OUTPUT]], [[TANH_CELL]][] : memref<f32>
  // CHECK:      {{.*}} = krnl.load [[TANH_CELL]][] : memref<f32>
  // CHECK:      {{.*}} = constant 1.000000e+00 : f32
  // CHECK:      {{.*}} = constant 2.000000e+00 : f32
  // CHECK:      {{.*}} = mulf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = negf {{.*}} : f32
  // CHECK:      {{.*}} = math.exp {{.*}} : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = math.exp {{.*}} : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = cmpf oge, {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = select {{.*}}, {{.*}}, {{.*}} : f32
  // CHECK:      krnl.store {{.*}}, [[ct]][] : memref<f32>
  // CHECK:      [[ct_LOAD:%.+]] = krnl.load [[ct]][] : memref<f32>

  // CHECK:      [[FtCt1:%.+]] = mulf [[Ft_LOAD]], [[Ct1_LOAD]] : f32
  // CHECK:      [[Itct:%.+]] = mulf [[It_LOAD]], [[ct_LOAD]] : f32
  // CHECK:      [[Ct:%.+]] = addf [[FtCt1]], [[Itct]] : f32
  // CHECK:      krnl.store [[Ct]], [[CELL_STATE]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>

  // CHECK:      [[XtWo_LOAD:%.+]] = krnl.load [[XtWo_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      [[HtRo_LOAD:%.+]] = krnl.load [[HtRo_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      [[Ot_OUTPUT:%.+]] = addf [[XtWo_LOAD]], [[HtRo_LOAD]] : f32

  // CHECK:      [[SIGMOID_OUTPUT:%.+]] = memref.alloc() : memref<f32>
  // CHECK:      krnl.store [[Ot_OUTPUT]], [[SIGMOID_OUTPUT]][] : memref<f32>
  // CHECK:      {{.*}} = krnl.load [[SIGMOID_OUTPUT]][] : memref<f32>
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = constant 1.000000e+00 : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}}: f32
  // CHECK:      {{.*}} = math.exp {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      krnl.store {{.*}}, [[Ot]][] : memref<f32>
  // CHECK:      [[Ot_LOAD:%.+]] = krnl.load [[Ot]][] : memref<f32>

  // CHECK:      [[TANH_HIDDEN:%.+]] = memref.alloc() : memref<f32>
  // CHECK:      krnl.store [[Ct]], [[TANH_HIDDEN]][] : memref<f32>
  // CHECK:      {{.*}} = krnl.load [[TANH_HIDDEN]][] : memref<f32>
  // CHECK:      {{.*}} = constant 1.000000e+00 : f32
  // CHECK:      {{.*}} = constant 2.000000e+00 : f32
  // CHECK:      {{.*}} = mulf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = negf {{.*}} : f32
  // CHECK:      {{.*}} = math.exp {{.*}} : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = math.exp {{.*}} : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = cmpf oge, {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = select {{.*}}, {{.*}}, {{.*}} : f32
  // CHECK:      krnl.store {{.*}}, [[hCt]][] : memref<f32>
  // CHECK:      [[hCt_LOAD:%.+]] = krnl.load [[hCt]][] : memref<f32>

  // CHECK:      [[Ht:%.+]] = mulf [[Ot_LOAD]], [[hCt_LOAD]] : f32
  // CHECK:      krnl.store [[Ht]], [[HIDDEN_STATE]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>

  // CHECK:      memref.dealloc [[It]] : memref<f32>
  // CHECK:      memref.dealloc [[Ft]] : memref<f32>
  // CHECK:      memref.dealloc [[ct]] : memref<f32>
  // CHECK:      memref.dealloc [[Ot]] : memref<f32>
  // CHECK:      memref.dealloc [[hCt]] : memref<f32>
  // CHECK:    }
  // CHECK:    memref.dealloc [[XtWi_GEMM]] : memref<3x3xf32>
  // CHECK:    memref.dealloc [[XtWo_GEMM]] : memref<3x3xf32>
  // CHECK:    memref.dealloc [[XtWf_GEMM]] : memref<3x3xf32>
  // CHECK:    memref.dealloc [[XtWc_GEMM]] : memref<3x3xf32>
  // CHECK:    memref.dealloc [[HtRi_GEMM]] : memref<3x3xf32>
  // CHECK:    memref.dealloc [[HtRo_GEMM]] : memref<3x3xf32>
  // CHECK:    memref.dealloc [[HtRf_GEMM]] : memref<3x3xf32>
  // CHECK:    memref.dealloc [[HtRc_GEMM]] : memref<3x3xf32>
 
  // CHECK:  }
  // CHECK:  memref.dealloc [[CELL_STATE]] : memref<1x3x3xf32>
  // CHECK:  return [[HIDDEN_STATE]] : memref<1x3x3xf32>
}

// -----

func private @test_lstm_reverse_mode(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64, direction = "reverse"} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

  // CHECK: [[REVERSE_IV_MAP:#.+]] = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
  // CHECK-LABEL: @test_lstm_reverse_mode

  // CHECK:  [[REVERSE_SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:  krnl.iterate([[REVERSE_SEQUENCE_LOOPS]]) with ([[REVERSE_SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:  %[[SEQUENCE_LEN:.+]] = constant 4 : index
  // CHECK:  %[[REVERSE_SEQUENCE_IV:.+]] = affine.apply [[REVERSE_IV_MAP]](%arg3)[%[[SEQUENCE_LEN]]{{]}}
  // CHECK:  [[Xt_LOAD:%.+]] = krnl.load %arg0[%[[REVERSE_SEQUENCE_IV]], {{.*}}, {{.*}}] : memref<4x3x2xf32>
}

// -----

func private @test_lstm_bidirectional_mode(%arg0: tensor<4x3x2xf32>, %arg1: tensor<2x12x2xf32>, %arg2: tensor<2x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64, direction = "bidirectional"} : (tensor<4x3x2xf32>, tensor<2x12x2xf32>, tensor<2x12x3xf32>, none, none, none, none, none) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

  // CHECK: [[REVERSE_IV_MAP:#.+]] = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
  // CHECK-LABEL: @test_lstm_bidirectional_mode

  // CHECK:  [[SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:  krnl.iterate([[SEQUENCE_LOOPS]]) with ([[SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:  {{.*}} = krnl.define_loops 2
  // CHECK:  {{.*}} = krnl.define_loops 1
  // CHECK:  [[Xt_LOAD:%.+]] = krnl.load %arg0[%arg3, {{.*}}, {{.*}}] : memref<4x3x2xf32>
  // CHECK:  {{.*}} = krnl.define_loops 1
  // CHECK:  {{.*}} = krnl.define_loops 2

  // CHECK:  [[REVERSE_SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:  krnl.iterate([[REVERSE_SEQUENCE_LOOPS]]) with ([[REVERSE_SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:  %[[SEQUENCE_LEN:.+]] = constant 4 : index
  // CHECK:  %[[REVERSE_SEQUENCE_IV:.+]] = affine.apply [[REVERSE_IV_MAP]](%arg3)[%[[SEQUENCE_LEN]]{{]}}
  // CHECK:  {{.*}} = krnl.define_loops 2
  // CHECK:  {{.*}} = krnl.define_loops 1
  // CHECK:  [[Xt_LOAD:%.+]] = krnl.load %arg0[%[[REVERSE_SEQUENCE_IV]], {{.*}}, {{.*}}] : memref<4x3x2xf32>
  // CHECK:  {{.*}} = krnl.define_loops 1
  // CHECK:  {{.*}} = krnl.define_loops 2
}

// -----

// Check handling unknown dimensions for LSTM by checking the
// correctness of allocating and deallocating memory.
func private @test_lstm_unkown_dims_allocation(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x12x?xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<?x?x?xf32>, tensor<1x12x?xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: @test_lstm_unkown_dims_allocation

  // allocate memory for all Hidden (Y).
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[SEQUENCE_LENGTH:%.+]] = memref.dim %arg0, [[C0]] : memref<?x?x?xf32>
  // CHECK: [[C1:%.+]] = constant 1 : index
  // CHECK: [[BATCH_SIZE:%.+]] = memref.dim %arg0, [[C1]] : memref<?x?x?xf32>
  // CHECK: [[Y:%.+]] = memref.alloc([[SEQUENCE_LENGTH]], [[BATCH_SIZE]]) : memref<?x1x?x3xf32>

  // allocate memory for Hidden (Y_h).
  // CHECK: [[C1_0:%.+]] = constant 1 : index
  // CHECK: [[BATCH_SIZE:%.+]] = memref.dim %arg0, [[C1_0]] : memref<?x?x?xf32>
  // CHECK: [[Y_h:%.+]] = memref.alloc([[BATCH_SIZE]]) : memref<1x?x3xf32>

  // allocate memory for Cell (Y_c).
  // CHECK: [[C1_1:%.+]] = constant 1 : index
  // CHECK: [[BATCH_SIZE:%.+]] = memref.dim %arg0, [[C1_1]] : memref<?x?x?xf32>
  // CHECK: [[Y_c:%.+]] = memref.alloc([[BATCH_SIZE]]) : memref<1x?x3xf32>

  // deallocate Y since there is no operation consuming it.
  // CHECK: memref.dealloc [[Y]] : memref<?x1x?x3xf32>
  // deallocate Y_c since it is not a return value.
  // CHECK: memref.dealloc [[Y_c]] : memref<1x?x3xf32>
  // CHECK: return [[Y_h]] : memref<1x?x3xf32>
}

// -----

/// Check RNN with three required inputs (X, W, R). The optional inputs are default.
func private @test_rnn_general_computation(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x3x2xf32>, %arg2: tensor<1x3x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x3x2xf32>, tensor<1x3x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_rnn_general_computation
  // CHECK: [[RES:%.+]] = memref.alloc() : memref<1x3x3xf32>

  /// Check initialize loop.
  // CHECK: [[INITIAL_VAL:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[DEF_LOOPS_INIT:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS_INIT]]#0, [[DEF_LOOPS_INIT]]#1, [[DEF_LOOPS_INIT]]#2) with ([[DEF_LOOPS_INIT]]#0 -> %arg3 = 0 to 1, [[DEF_LOOPS_INIT]]#1 -> %arg4 = 0 to 3, [[DEF_LOOPS_INIT]]#2 -> %arg5 = 0 to 3) {
  // CHECK:   krnl.store [[INITIAL_VAL]], [[RES]][%arg3, %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK: }

  /// Check main loop.
  // CHECK: [[SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK: krnl.iterate([[SEQUENCE_LOOPS]]) with ([[SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:   [[HtRi_GEMM:%.+]] = memref.alloc() : memref<3x3xf32>
  // CHECK:   [[XtWi_GEMM:%.+]] = memref.alloc() : memref<3x3xf32>
  // CHECK:   [[ZERO_INDEX:%.+]] = constant 0 : index
  // CHECK:   {{.*}} = constant 3 : index
  // CHECK:   {{.*}} = constant 0 : index
  // CHECK:   {{.*}} = constant 1 : index

  /// Check reduction loop to compute matrix multiplication for 'Xt*(Wi^T)' and 'Ht-1*(Ri^T)'
  // CHECK:   [[MATRIX_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[MATRIX_LOOPS]]#0, [[MATRIX_LOOPS]]#1) with ([[MATRIX_LOOPS]]#0 -> %arg4 = 0 to 3, [[MATRIX_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:     [[CST0:%.+]] = constant 0.000000e+00 : f32
  // CHECK:     krnl.store [[CST0]], [[XtWi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     krnl.store [[CST0]], [[HtRi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     [[XW_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[XW_LOOPS]]) with ([[XW_LOOPS]] -> %arg6 = 0 to 2) {
  // CHECK:       [[Xt_LOAD:%.+]] = krnl.load %arg0[%arg3, %arg4, %arg6] : memref<4x3x2xf32>
  // CHECK:       [[Wi_LOAD:%.+]] = krnl.load %arg1{{\[}}[[ZERO_INDEX]], %arg5, %arg6] : memref<1x3x2xf32>
  // CHECK:       {{.*}} = mulf [[Xt_LOAD]], [[Wi_LOAD]] : f32
  // CHECK:       {{.*}} = krnl.load [[XtWi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:       {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:       krnl.store {{.*}}, [[XtWi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     }
  // CHECK:     [[HR_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[HR_LOOPS]]) with ([[HR_LOOPS]] -> %arg6 = 0 to 3) {
  // CHECK:       [[Ht_LOAD:%.+]] = krnl.load %0{{\[}}[[ZERO_INDEX]], %arg4, %arg6] : memref<1x3x3xf32>
  // CHECK:       [[Ri_LOAD:%.+]] = krnl.load %arg2{{\[}}[[ZERO_INDEX]], %arg5, %arg6] : memref<1x3x3xf32>
  // CHECK:       {{.*}} = mulf [[Ht_LOAD]], [[Ri_LOAD]] : f32
  // CHECK:       {{.*}} = krnl.load [[HtRi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:       {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:       krnl.store {{.*}}, [[HtRi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     }
  // CHECK:   }
 
  // CHECK:   [[DATA_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[DATA_LOOPS]]#0, [[DATA_LOOPS]]#1) with ([[DATA_LOOPS]]#0 -> %arg4 = 0 to 3, [[DATA_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:     [[Ht:%.+]] = memref.alloc() : memref<f32>

  /// Check 'Xt*(Wi^T) + Ht-1*(Ri^T)'
  // CHECK:     [[LOAD_XWi:%.+]] = krnl.load [[XtWi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     [[LOAD_HRi:%.+]] = krnl.load [[HtRi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     [[XWi_PLUS_HRi:%.+]] = addf [[LOAD_XWi]], [[LOAD_HRi]] : f32

  /// Check calling 'Tanh'
  // CHECK:     {{.*}} = memref.alloc() : memref<f32>
  // CHECK:     krnl.store [[XWi_PLUS_HRi]], {{.*}} : memref<f32>
  // CHECK:     {{.*}} = krnl.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 1.000000e+00 : f32
  // CHECK:     {{.*}} = constant 2.000000e+00 : f32
  // CHECK:     {{.*}} = mulf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = negf {{.*}} : f32
  // CHECK:     {{.*}} = math.exp {{.*}} : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = math.exp {{.*}} : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = cmpf oge, {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = select {{.*}}, {{.*}}, {{.*}} : f32
  // CHECK:     krnl.store {{.*}}, [[Ht]][] : memref<f32>

  /// Check storing the result.
  // CHECK:     [[NEW_Ht_LOAD:%.+]] = krnl.load [[Ht]][] : memref<f32>
  // CHECK:     krnl.store [[NEW_Ht_LOAD]], [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     memref.dealloc [[Ht]] : memref<f32>
  // CHECK:   }
  // CHECK:   memref.dealloc [[XtWi_GEMM]] : memref<3x3xf32>
  // CHECK:   memref.dealloc [[HtRi_GEMM]] : memref<3x3xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<1x3x3xf32>
}

// -----

/// Check RNN with three required inputs (X, W, R), and bias input.
func private @test_rnn_with_bias(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x3x2xf32>, %arg2: tensor<1x3x3xf32>, %arg3: tensor<1x6xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x3x2xf32>, tensor<1x3x3xf32>, tensor<1x6xf32>, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_rnn_with_bias
  // CHECK: [[LOAD_W_BIAS:%.+]] = krnl.load %arg3[{{.*}}, {{.*}}] : memref<1x6xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_W_BIAS]] : f32
  // CHECK: [[LOAD_R_BIAS:%.+]] = krnl.load %arg3[{{.*}}, {{.*}}] : memref<1x6xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_R_BIAS]] : f32
}

// -----

// Check handling unknown dimensions for RNN by checking the
// correctness of allocating and deallocating memory.
func private @test_rnn_unkown_dims_allocation(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x3x?xf32>, %arg2: tensor<1x3x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<?x?x?xf32>, tensor<1x3x?xf32>, tensor<1x3x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: @test_rnn_unkown_dims_allocation

  // allocate memory for Hidden (Y_h).
  // CHECK: [[C1_0:%.+]] = constant 1 : index
  // CHECK: [[BATCH_SIZE:%.+]] = memref.dim %arg0, [[C1_0]] : memref<?x?x?xf32>
  // CHECK: [[Y_h:%.+]] = memref.alloc([[BATCH_SIZE]]) : memref<1x?x3xf32>

  // CHECK: return [[Y_h]] : memref<1x?x3xf32>
}

